// d3d12_backend.cpp — the Windows D3D12 + DirectML backend (OpenDLSS-NR MetalFX).
//
// Host structure mirrors platforms/macos/src/metal_backend.mm one-for-one and
// platforms/linux/src/vulkan_backend.cpp for the block/transition driver; the
// DirectML GEMM path plays the role MetalFX plays on macOS — an accelerator
// layered over our own HLSL compute kernels, never a requirement.
//
// Device / command plumbing:
//   * debug layer opt-in (OPENDLSS_D3D12_DEBUG=1 or D3D12BackendOptions)
//   * hardware adapter walk; WARP fallback (OPENDLSS_D3D12_WARP=1 forces it)
//   * one DIRECT command queue, ONE command allocator reused with a fence +
//     event sync after every submission (correctness-first, mirrors the
//     Vulkan backend's per-frame submit+wait)
//
// Shaders are compiled at runtime with D3DCompileFromFile (cs_5_0; the
// sources are SM 6.x compatible and can be precompiled with DXC — see
// CMakeLists). Shader directory resolution:
//     $OPENDLSS_D3D_SHADER_DIR  ->  "shaders_hlsl" next to the executable
//     (CMake copies the .hlsl tree there)  ->  the configure-time build-dir
//     copy (OPENDLSS_D3D_SHADER_BUILD_DIR)  ->  "shaders_hlsl" (cwd)
//
// Root signature (one unified signature for every kernel, following the
// .hlsl conventions exactly — all parameters are descriptor tables or the
// b0 root-constant block):
//     table 0: SRVs  t0..t7   (Texture2D or ByteAddressBuffer per kernel)
//     table 1: UAVs  u0..u4   (RWTexture2D<float4|float2> or RWByteAddressBuffer)
//     root constants b0: 64 dwords = uint4 P[16] (graph) / float4 C[8]+pad (media)
// Every PSO is cached by kernel name in psoCache_. Descriptor heap slots are
// handed out in spans per dispatch from a 512-slot shader-visible
// CBV_SRV_UAV heap and recycled after every fence wait (the Vulkan backend's
// descriptor-pool-reset semantics); when a frame would outgrow the heap the
// dispatcher syncs mid-frame and keeps recording on the fresh list.
//
// Resources:
//   * token tensors live in byte-address buffers as f16 pairs (two halfs per
//     dword, little-endian: lane 2*i in bits 0..15, lane 2*i+1 in bits
//     16..31) — one consistent layout for every kernel, matching
//     Common.hlsli's OdlLoadHalf / OdlStoreHalf2
//   * writable DEFAULT-heap buffers get BOTH views: a raw byte-address view
//     (DXGI_FORMAT_R32_TYPELESS + D3D12_BUFFER_UAV_FLAG_RAW — the
//     ByteAddressBuffer style the kernels bind) and a typed
//     DXGI_FORMAT_R32_UINT UAV used with ClearUnorderedAccessViewUint to
//     guarantee zero-initialized scratch (committed resources are zero by
//     spec; the clear is belt-and-braces)
//   * media textures are committed RGBA32F (D3D12_RESOURCE_FLAG_ALLOW_
//     UNORDERED_ACCESS | ALLOW_SIMULTANEOUS_ACCESS), uploaded/read back
//     through pitched staging buffers (256-byte row pitch, 512-byte offset
//     alignment). GpuTex tracks its D3D12_RESOURCE_STATES and every change
//     goes through transition() (COMMON -> UNORDERED_ACCESS -> COPY_SOURCE /
//     COPY_DEST); simultaneous-access textures decay back to COMMON when the
//     command list finishes, which flush() folds back into the tracking.
//
// DirectML:
//   * DirectML.dll is loaded dynamically (LoadLibraryW); DMLCreateDevice1 is
//     resolved with GetProcAddress — no directml.lib link dependency
//   * requires feature level 2_0 (FLOAT16 tensors); otherwise the DML path
//     stays disabled and every GEMM runs in the odl_channel_gemm HLSL kernel
//   * accelerates channel GEMMs whose output is E4M3-published (pubMode 0)
//     with >= 4096 weight elements: a DML GEMM (A [tokens x inCh] x
//     B [outCh x inCh]ᵀ, both FLOAT16) writes raw f16, then the same
//     odl_channel_gemm kernel in mode 2 publishes the output onto the E4M3
//     grid — the quantization grid NEVER depends on DML. Raw-f16 GEMMs
//     (pubMode 1: ViT qkv, attention proj, vitout.down, trans*.down) always
//     run in the HLSL kernel so their rounding matches the CPU reference
//     exactly. Known deviation: DML accumulates in f32 but rounds through an
//     f16 intermediate before our publish pass (the reference publishes
//     straight from the f32 accumulator).
//   * every compiled GEMM is initialized once (IDMLOperatorInitializer) and
//     cached per shape; execute binds A/B/output through DML_BUFFER_BINDING
//     against the very buffers the HLSL kernels use
//
// Game mode uses our Lanczos + reprojection + temporal-blend fallback (the
// MTLFXTemporalScaler role; DirectML has no temporal scaler). Every op
// checks ok_ and reports failure (or the identity) when no device exists;
// the factory still returns the instance with info().kind == D3D12 so the
// CLI can report it — see d3d12_backend.h.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.

#include "d3d12_backend.h"

#include "opendlss/image.h"     // image_downscale_box (game feedback path)
#include "opendlss/logging.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>

// directml.h is optional at build time; everything DML is compiled out
// without it (the HLSL GEMM fallback carries the full pipeline either way).
#if defined(__has_include)
#  if __has_include(<directml.h>)
#    include <directml.h>
#    define OPENDLSS_D3D12_HAS_DML 1
#  endif
#endif

// IID_PPV_ARGS shim for MinGW toolchains whose headers predate the macro.
#ifndef IID_PPV_ARGS
namespace odl_win {
template <typename T> inline void** iid_ppv_args_helper(T** pp) {
    return reinterpret_cast<void**>(pp);
}
} // namespace odl_win
#define IID_PPV_ARGS(pp) __uuidof(**(pp)), odl_win::iid_ppv_args_helper(pp)
#endif

namespace opendlss {
namespace {

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------
constexpr uint32_t kCsuHeapSlots     = 512;   // shader-visible CBV_SRV_UAV heap
constexpr uint32_t kSrvTableRegs     = 8;     // t0..t7
constexpr uint32_t kUavTableRegs     = 9;     // u0..u4
constexpr uint32_t kRootConstWords   = 64;    // uint4 P[16] / float4 C[8] + pad
constexpr uint64_t kDmlMinWeightElems = 4096; // DML threshold (weight elements)
constexpr float    kGameMaxBlend     = 0.85f; // temporal cap for the game fallback
constexpr uint32_t kTexRowAlign      = 256;   // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
constexpr uint32_t kTexOffsetAlign   = 512;   // D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT

inline uint32_t divUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }
inline uint32_t rowPitch(uint32_t widthBytes) { return alignUp(widthBytes, kTexRowAlign); }

std::string narrow(const wchar_t* w, int len = -1) {
    if (!w) return std::string();
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return std::string();
    std::string s(size_t(n), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w, len, &s[0], n, nullptr, nullptr);
    return s;
}

std::wstring widen(const std::string& s) {
    const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w(size_t(n), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), &w[0], n);
    return w;
}

bool dirHasShaders(const std::string& dir) {
    const std::string probe = dir + "/Graph.hlsli";
    const DWORD attr = ::GetFileAttributesA(probe.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

// ---------------------------------------------------------------------------
// shader directory resolution:
//   $OPENDLSS_D3D_SHADER_DIR -> "shaders_hlsl" next to the executable ->
//   the configure-time build-dir copy -> "shaders_hlsl" (cwd)
// ---------------------------------------------------------------------------
std::string resolveShaderDir() {
    if (const char* env = std::getenv("OPENDLSS_D3D_SHADER_DIR"))
        if (env[0] != '\0') return env;

    wchar_t buf[MAX_PATH];
    const DWORD n = ::GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::wstring exe(buf, n);
        const size_t slash = exe.find_last_of(L"\\/");
        if (slash != std::wstring::npos) {
            std::string dir = narrow(exe.substr(0, slash + 1).c_str()) + "shaders_hlsl";
            if (dirHasShaders(dir)) return dir;
        }
    }
#ifdef OPENDLSS_D3D_SHADER_BUILD_DIR
    {
        std::string dir = OPENDLSS_D3D_SHADER_BUILD_DIR;
        if (dirHasShaders(dir)) return dir;
    }
#endif
    return "shaders_hlsl";
}

// ---------------------------------------------------------------------------
// kernel -> source file / entry point map (platforms/windows/shaders/*.hlsl)
// The root signature is unified (t0..t7 table, u0..u4 table, b0 constants),
// so only the source location differs per kernel.
// ---------------------------------------------------------------------------
struct KernelSpec {
    const char* name;
    const char* file;
    const char* entry;
};

const KernelSpec kKernels[] = {
    // Preprocess.hlsli
    { "odl_preprocess",          "Preprocess.hlsli", "odl_preprocess" },
    { "odl_input_embed",         "Preprocess.hlsli", "odl_input_embed" },
    // Graph.hlsli
    { "odl_ffn",                 "Graph.hlsli", "odl_ffn" },
    { "odl_window_attention",    "Graph.hlsli", "odl_window_attention" },
    { "odl_global_attention",    "Graph.hlsli", "odl_global_attention" },
    { "odl_block_skip",          "Graph.hlsli", "odl_block_skip" },
    { "odl_block_epilogue",      "Graph.hlsli", "odl_block_epilogue" },
    { "odl_pool2x2",             "Graph.hlsli", "odl_pool2x2" },
    { "odl_channel_gemm",        "Graph.hlsli", "odl_channel_gemm" },
    { "odl_decoder_skip",        "Graph.hlsli", "odl_decoder_skip" },
    { "odl_nearest_upsample2x",  "Graph.hlsli", "odl_nearest_upsample2x" },
    { "odl_head",                "Graph.hlsli", "odl_head" },
    // Media.hlsli
    { "odl_denoise",             "Media.hlsli", "odl_denoise" },
    { "odl_upscale_h",           "Media.hlsli", "odl_upscale_h" },
    { "odl_upscale_v",           "Media.hlsli", "odl_upscale_v" },
    { "odl_sharpen",             "Media.hlsli", "odl_sharpen" },
    { "odl_motion",              "Media.hlsli", "odl_motion" },
    { "odl_reproject",           "Media.hlsli", "odl_reproject" },
    { "odl_temporal_blend",      "Media.hlsli", "odl_temporal_blend" },
    { "odl_head_composite",      "Media.hlsli", "odl_head_composite" },
};

const KernelSpec* findKernel(const std::string& name) {
    for (const KernelSpec& k : kKernels)
        if (name == k.name) return &k;
    return nullptr;
}

// ---------------------------------------------------------------------------
// resource wrappers
// ---------------------------------------------------------------------------
struct GpuBuffer {
    ComPtr<ID3D12Resource> res;
    void*  map   = nullptr;             // persistent map (upload/readback heaps)
    size_t bytes = 0;
    bool   upload = false;              // UPLOAD heap, GENERIC_READ, host-writable
    bool   writable = false;            // DEFAULT heap with UAV flag
    bool valid() const { return res != nullptr; }
};

struct GpuTex {
    ComPtr<ID3D12Resource> res;
    uint32_t w = 0, h = 0;
    DXGI_FORMAT fmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;  // tracked
    bool valid() const { return res != nullptr; }
};

// one kernel argument: a register slot bound to a buffer or a texture
struct BindArg {
    uint32_t reg = 0;
    const GpuBuffer* buf = nullptr;
    const GpuTex*    tex = nullptr;
};
inline BindArg bindBuf(uint32_t reg, const GpuBuffer& b) { return BindArg{ reg, &b, nullptr }; }
inline BindArg bindTex(uint32_t reg, const GpuTex& t)    { return BindArg{ reg, nullptr, &t }; }

// push-constant helpers (kernels share the 64-word b0 block)
inline void pcU(uint32_t* pc, uint32_t i, uint32_t v) { pc[i] = v; }
inline void pcF(uint32_t* pc, uint32_t i, float v)    { std::memcpy(&pc[i], &v, 4); }

} // namespace

// ---------------------------------------------------------------------------
// D3D12Backend
// ---------------------------------------------------------------------------
class D3D12Backend final : public IBackend {
public:
    explicit D3D12Backend(const D3D12BackendOptions& options) : opt_(options) {
        // env overrides (see d3d12_backend.h)
        if (const char* e = std::getenv("OPENDLSS_D3D12_DEBUG")) opt_.enableDebugLayer = std::strcmp(e, "0") != 0;
        if (const char* e = std::getenv("OPENDLSS_D3D12_WARP"))  opt_.preferWarp       = std::strcmp(e, "0") != 0;
        if (const char* e = std::getenv("OPENDLSS_D3D12_DML"))   opt_.enableDirectML   = std::strcmp(e, "0") != 0;

        if (!createDevice()) { log_error("d3d12: no D3D12 device (backend inert)"); return; }
        if (!createCommandResources()) { log_error("d3d12: command resources failed"); return; }
        if (!createRootSignature())    { log_error("d3d12: root signature failed"); return; }
        if (!createDummies())          { log_error("d3d12: dummy resources failed"); return; }

        shaderDir_ = resolveShaderDir();

        info_.kind = BackendKind::D3D12;
        info_.name = "d3d12";            // becomes "d3d12+directml" when DML is live
        info_.neuralGraph = true;
        info_.fp16Storage = false;       // f16 storage, f32 arithmetic (as the shaders say)
        info_.e4m3Quant = true;          // kernels publish through the E4M3 grid
        info_.maxTextureDim = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;

        if (initDml()) {
            info_.name = "d3d12+directml";
            info_.details = "D3D12 compute graph (HLSL, runtime-compiled) + DirectML f16 GEMMs, "
                            "Lanczos spatial scaler + temporal-blend game fallback";
        } else {
            info_.details = "D3D12 compute graph (HLSL, runtime-compiled), Lanczos spatial "
                            "scaler + temporal-blend game fallback (DirectML unavailable)";
        }
        info_.details += "; shaders: ";
        info_.details += shaderDir_;
        ok_ = true;
        log_info("d3d12: %s, shaders: %s", info_.deviceName.c_str(), shaderDir_.c_str());
    }

    ~D3D12Backend() override {
        // wait for any outstanding GPU work before tearing down
        if (queue_ && fence_) {
            constexpr uint64_t kTeardownValue = 0xDEADBEEFull;
            queue_->Signal(fence_.Get(), kTeardownValue);
            if (fence_->GetCompletedValue() < kTeardownValue) {
                fence_->SetEventOnCompletion(kTeardownValue, fenceEvent_);
                ::WaitForSingleObject(fenceEvent_, 10000);
            }
        }
        for (auto& kv : tensorBufs_) releaseBuffer(kv.second);
        tensorBufs_.clear();
        releaseBuffer(dummyBuf_);
        releaseBuffer(zeroBuf_);
        releaseBuffer(upStage_);
        releaseBuffer(downStage_);
        releaseBuffer(featBuf_);
        releaseBuffer(stateA_);
        releaseBuffer(stateB_);
        releaseBuffer(ffnOut_);
        releaseBuffer(projIn_);
        releaseBuffer(qkvBuf_);
        releaseBuffer(attnOut_);
        releaseBuffer(projOut_);
        releaseBuffer(headBuf_);
        releaseBuffer(headReadback_);
        for (int i = 0; i < 3; ++i) releaseBuffer(scratch_[i]);
        releaseTexture(dummyTex_);
        releaseTexture(dummyTex2_);
        releaseTexture(gColor_);
        releaseTexture(gHist_);
        releaseTexture(gMv_);
        releaseTexture(gRp_);
        releaseTexture(gCf_);
        releaseTexture(gBlend_);
        releaseTexture(gTmp_);
        releaseTexture(gOut_);
        texRegistry_.clear();
        if (fenceEvent_) { ::CloseHandle(fenceEvent_); fenceEvent_ = nullptr; }
    }

    const BackendInfo& info() const override { return info_; }

    // ---------------- image ops ----------------
    bool denoiseSpatial(const Image& input, Image& output, float strength) override {
        if (!ok_ || strength <= 0.0f || input.empty()) { output = input; return true; }

        GpuTex src, dst;
        if (!allocTexture(input.width, input.height, DXGI_FORMAT_R32G32B32A32_FLOAT, src)) return false;
        if (!allocTexture(input.width, input.height, DXGI_FORMAT_R32G32B32A32_FLOAT, dst)) {
            releaseTexture(src); return false;
        }

        bool ok = false;
        if (beginFrame()) {
            uint32_t pc[kRootConstWords] = {};
            pcF(pc, 0, strength);
            pcU(pc, 4, input.width);
            pcU(pc, 5, input.height);
            ok = uploadImage(input, src) &&
                 dispatch("odl_denoise", pc,
                          { bindTex(0, src), bindBuf(0, dst) },
                          divUp(input.width, 8), divUp(input.height, 8), 1);
            size_t off = 0;
            if (ok) ok = downloadImage(dst, off);
            if (ok && flush()) ok = finishDownload(dst, output, off);
            else ok = false;
        }
        releaseTexture(src);
        releaseTexture(dst);
        return ok;
    }

    bool upscaleSpatial(const Image& input, Image& output, uint32_t factor, float sharpen) override {
        if (!ok_) { output = input; return false; }
        if (factor == 0 || input.empty()) { output = input; return false; }
        const uint32_t OW = input.width * factor, OH = input.height * factor;
        output = Image(OW, OH);

        GpuTex src, tmp, up, fin;
        bool ok = false;
        do {
            if (!allocTexture(input.width, input.height, DXGI_FORMAT_R32G32B32A32_FLOAT, src)) break;
            if (!allocTexture(OW, input.height, DXGI_FORMAT_R32G32B32A32_FLOAT, tmp)) break;
            if (!allocTexture(OW, OH, DXGI_FORMAT_R32G32B32A32_FLOAT, up)) break;
            const bool doSharpen = sharpen > 0.0f;
            if (doSharpen && !allocTexture(OW, OH, DXGI_FORMAT_R32G32B32A32_FLOAT, fin)) break;

            if (!beginFrame()) break;
            uint32_t pc[kRootConstWords] = {};
            ok = uploadImage(input, src);
            if (ok) {   // 1. Lanczos3 horizontal pass
                pcF(pc, 0, float(OW));
                pcU(pc, 4, input.width);            // C[1].x = srcW
                pcU(pc, 5, input.height);           // C[1].y = srcH
                pcU(pc, 6, input.height);           // C[1].z = dstH (unchanged rows)
                ok = dispatch("odl_upscale_h", pc, { bindTex(0, src), bindBuf(0, tmp) },
                              divUp(OW, 8), divUp(input.height, 8), 1);
            }
            if (ok) {   // 2. Lanczos3 vertical pass
                std::memset(pc, 0, sizeof(pc));
                pcF(pc, 0, float(OH));
                pcU(pc, 4, OW);                     // C[1].x = tmpW
                pcU(pc, 5, input.height);           // C[1].y = tmpH
                pcU(pc, 6, OW);                     // C[1].z = dstW
                ok = dispatch("odl_upscale_v", pc, { bindTex(0, tmp), bindBuf(0, up) },
                              divUp(OW, 8), divUp(OH, 8), 1);
            }
            const GpuTex* finalTex = &up;
            if (ok && doSharpen) {   // 3. adaptive sharpen at output res
                std::memset(pc, 0, sizeof(pc));
                pcF(pc, 0, sharpen);
                pcU(pc, 4, OW);
                pcU(pc, 5, OH);
                ok = dispatch("odl_sharpen", pc, { bindTex(0, up), bindBuf(0, fin) },
                              divUp(OW, 8), divUp(OH, 8), 1);
                finalTex = &fin;
            }
            size_t off = 0;
            if (ok) ok = downloadImage(*finalTex, off);
            if (ok && flush()) ok = finishDownload(*finalTex, output, off);
            else ok = false;
        } while (false);

        releaseTexture(src);
        releaseTexture(tmp);
        releaseTexture(up);
        releaseTexture(fin);
        return ok;
    }

    // ---------------- neural graph ----------------
    bool loadModel(const std::string& modelDir) override {
        if (!ok_) return false;
        auto m = Model::load(modelDir);
        if (!m) { log_error("d3d12: model load failed (%s)", modelDir.c_str()); return false; }
        model_ = std::move(m);
        // weights upload lazily: every model_->tensor(name) the driver touches is
        // materialized into a packed-f16 UPLOAD buffer on first use (ensureTensor)
        for (auto& kv : tensorBufs_) releaseBuffer(kv.second);
        tensorBufs_.clear();
        log_info("d3d12: model loaded (%zu blocks, levels %u)", size_t(model_->schedule().size()),
                 model_->config().levels);
        return true;
    }

    bool hasNeuralGraph() const override { return ok_ && model_ != nullptr; }
    const Model* model() const override { return model_.get(); }

    bool runNeuralGraph(const Geometry& geom, const uint16_t* features,
                        float* head, uint64_t frameSeed) override {
        (void)frameSeed;
        if (!ok_ || !model_) return false;
        const auto& sched = model_->schedule();
        if (sched.empty()) { log_error("d3d12: empty block schedule"); return false; }
        if (!model_->tensor("input_proj") || !model_->tensor("head.w")) {
            log_error("d3d12: model missing input_proj / head.w");
            return false;
        }
        const uint32_t FW = geom.fieldWidth, FH = geom.fieldHeight;
        const uint32_t nLevels = model_->config().levels;
        if (FW == 0 || FH == 0 || !features || !head) return false;

        // resolve per-block geometry (the CPU executor's convention)
        std::vector<BlockScheduleEntry> entries = sched;
        for (auto& e : entries) {
            if (e.onField) { e.levelWidth = FW; e.levelHeight = FH; }
            else {
                const Level& lv = geom.levels[std::min(e.level, nLevels - 1)];
                e.levelWidth = lv.width; e.levelHeight = lv.height;
            }
        }

        // ---- scratch sizing (bytes; f16 pairs = 2 bytes per lane) ----
        size_t maxBytes    = size_t(FW) * FH * 32 * 2;    // Metal's baseline
        size_t maxQkvBytes = maxBytes;
        for (const BlockScheduleEntry& e : entries) {
            const size_t tokens = size_t(e.levelWidth) * e.levelHeight;
            const size_t padded = (tokens + 63u) & ~size_t(63u);   // ViT qkv padding
            maxBytes    = std::max(maxBytes,    tokens * e.channels * 2);
            maxBytes    = std::max(maxBytes,    padded * e.channels * 2);
            maxQkvBytes = std::max(maxQkvBytes, padded * e.channels * 3 * 2);
        }

        // ---- features upload: [fieldTokens][16] u16 lanes -> f16 pairs ----
        const size_t featLanes = size_t(FW) * FH * 16;
        if (!ensureUpload(featBuf_, featLanes * 2)) return false;
        {
            uint32_t* w = static_cast<uint32_t*>(featBuf_.map);
            for (size_t d = 0; d < featLanes / 2; ++d)
                w[d] = uint32_t(features[2 * d]) | (uint32_t(features[2 * d + 1]) << 16);
        }
        if (!ensureReadback(headReadback_, size_t(FW) * FH * 4 * 4)) return false;

        if (!beginFrame()) { log_error("d3d12: begin command list failed"); return false; }
        if (!ensureGpu(stateA_, maxBytes) || !ensureGpu(stateB_, maxBytes) ||
            !ensureGpu(ffnOut_, maxBytes) || !ensureGpu(projIn_, maxBytes) ||
            !ensureGpu(qkvBuf_, maxQkvBytes) || !ensureGpu(attnOut_, maxBytes) ||
            !ensureGpu(projOut_, maxBytes) ||
            !ensureGpu(headBuf_, size_t(FW) * FH * 4 * 4)) {
            abortFrame();
            return false;
        }
        for (int i = 0; i < 3; ++i) scratchUsed_[i] = false;

        GpuBuffer* state = &stateA_;      // live token state ([tokens][C] f16 pairs)
        uint32_t liveW = FW, liveH = FH, liveC = 0;
        GpuBuffer block0Buf;              // field-level skip (aliases stateA/B)
        bool hasBlock0 = false;
        GpuBuffer skipBufs[9];            // [0..7] levels, [8] field
        bool hasSkip[9] = {};

        for (size_t bi = 0; bi < entries.size(); ++bi) {
            const BlockScheduleEntry& e = entries[bi];
            const uint32_t bw = e.levelWidth, bh = e.levelHeight;
            const uint32_t tokens = bw * bh;
            const uint32_t C = e.channels;
            if (tokens == 0 || C == 0) { abortFrame(); return false; }

            if (bi == 0) {
                liveC = C;
                if (!runInputEmbed(*state, FW, FH, C)) { abortFrame(); return false; }
            } else {
                const BlockScheduleEntry& p = entries[bi - 1];
                const int32_t effE = e.onField ? -1 : int32_t(e.level);
                const int32_t effP = p.onField ? -1 : int32_t(p.level);
                GpuBuffer& other = (state == &stateA_) ? stateB : stateA;
                if (effE > effP) {
                    // encoder transition: 2x2 box pool -> C -> 2C up-GEMM
                    const std::string lvl = p.onField ? "trans_f" : "trans" + std::to_string(p.level);
                    GpuBuffer& pooled = takeScratch(size_t(bw) * bh * p.channels * 2);
                    if (!runPool(*state, liveW, liveH, bw, bh, p.channels, pooled) ||
                        !runGemm(pooled, lvl + ".up", bw * bh, C, p.channels, 0, other)) {
                        abortFrame(); return false;
                    }
                    state = &other;
                    liveC = C;
                } else if (effE < effP) {
                    // decoder transition: down-GEMM -> 2x upsample -> skip FMA
                    const std::string lvl = p.onField ? "trans_f" : "trans" + std::to_string(p.level);
                    GpuBuffer& proj = takeScratch(size_t(liveW) * liveH * C * 2);
                    GpuBuffer& up   = takeScratch(size_t(bw) * bh * C * 2);
                    if (!runGemm(*state, lvl + ".down", liveW * liveH, C, p.channels, 1, proj) ||
                        !runUpsample(proj, liveW, liveH, bw, bh, C, up)) {
                        abortFrame(); return false;
                    }
                    const uint32_t slot = std::min(e.level, 8u);
                    const bool hasSk = e.onField ? hasBlock0 : hasSkip[slot];
                    const GpuBuffer sk = hasSk ? (e.onField ? block0Buf : skipBufs[slot]) : GpuBuffer();
                    const GpuBuffer* scaleT   = tensorPtr(lvl + ".scale");
                    const GpuBuffer* inScaleT = tensorPtr(lvl + ".in_scale");
                    if (!runDecoderSkip(up, sk, hasSk, scaleT, inScaleT, bw * bh, C, other)) {
                        abortFrame(); return false;
                    }
                    state = &other;
                    liveC = C;
                } else if (p.channels * 2 == C) {
                    // ViT entry: channel expansion GEMM (E4M3 published)
                    if (!runGemm(*state, "vitin.up", tokens, C, p.channels, 0, other)) {
                        abortFrame(); return false;
                    }
                    state = &other;
                    liveC = C;
                } else if (p.channels == C * 2) {
                    // ViT exit: channel contraction GEMM (raw f16 published)
                    if (!runGemm(*state, "vitout.down", tokens, C, p.channels, 1, other)) {
                        abortFrame(); return false;
                    }
                    state = &other;
                    liveC = C;
                }
            }
            liveW = bw; liveH = bh;

            if (!runBlock(e, *state)) { abortFrame(); return false; }
            if (bi == 0) { block0Buf = *state; hasBlock0 = true; }
            // save encoder skips (field-level skip lives in slot 8)
            if (bi + 1 < entries.size()) {
                const BlockScheduleEntry& nx = entries[bi + 1];
                const int32_t effE2 = e.onField ? -1 : int32_t(e.level);
                const int32_t effN = nx.onField ? -1 : int32_t(nx.level);
                if (e.onField) { skipBufs[8] = *state; hasSkip[8] = true; }
                else if (effN > effE2 && e.level < 8) { skipBufs[e.level] = *state; hasSkip[e.level] = true; }
            }
        }

        // ---- head: C -> 4 f32 per token ----
        {
            const uint32_t tokens = liveW * liveH;
            uint32_t pc[kRootConstWords] = {};
            pcU(pc, 0, tokens);
            pcU(pc, 1, liveC);
            const GpuBuffer* bias = tensorPtr("head.b");
            std::vector<BindArg> args = {
                bindBuf(0, *state),
                bindBuf(1, *tensorPtr("head.w")),
                bindBuf(2, headOutBuf(*state)),   // placeholder replaced below
            };
            args[2] = bias ? bindBuf(2, *bias) : BindArg{ 2, nullptr, nullptr };
            args.push_back(bindBuf(0, headBuf_));      // u0 (register 0 on the UAV side)
            if (!dispatch("odl_head", pc, args, divUp(tokens, 64), 1, 1)) {
                abortFrame();
                return false;
            }
        }
        list_->CopyBufferRegion(headReadback_.res.Get(), 0, headBuf_.res.Get(), 0,
                                uint64_t(liveW) * liveH * 4 * 4);

        if (!flush()) { log_error("d3d12: graph submit failed"); return false; }
        std::memcpy(head, headReadback_.map, size_t(FW) * FH * 4 * sizeof(float));
        return true;
    }

    // ---------------- video / game temporal ops ----------------
    bool estimateMotion(const Image& prevLuma, const Image& currLuma, MotionField& mv) override {
        if (!ok_ || prevLuma.empty() || currLuma.empty()) return false;
        mv.width = currLuma.width;
        mv.height = currLuma.height;
        mv.xy.assign(size_t(mv.width) * mv.height * 2, 0.f);

        GpuTex prev, curr, mvTex;
        bool ok = false;
        do {
            if (!allocTexture(prevLuma.width, prevLuma.height, DXGI_FORMAT_R32G32B32A32_FLOAT, prev)) break;
            if (!allocTexture(currLuma.width, currLuma.height, DXGI_FORMAT_R32G32B32A32_FLOAT, curr)) break;
            if (!allocTexture(currLuma.width, currLuma.height, DXGI_FORMAT_R32G32_FLOAT, mvTex)) break;
            if (!beginFrame()) break;
            ok = uploadImage(prevLuma, prev) && uploadImage(currLuma, curr);
            if (ok) {
                uint32_t pc[kRootConstWords] = {};
                pcU(pc, 4, currLuma.width);            // C[1].x = currW
                pcU(pc, 5, currLuma.height);           // C[1].y = currH
                pcU(pc, 6, prevLuma.width);            // C[1].z = prevW
                pcU(pc, 7, prevLuma.height);           // C[1].w = prevH
                // one thread per 16x16 block, numthreads(8,8)
                const uint32_t bx = divUp(mv.width, 16), by = divUp(mv.height, 16);
                ok = dispatch("odl_motion", pc,
                              { bindTex(0, prev), bindTex(1, curr), bindBuf(4, mvTex) },
                              divUp(bx, 8), divUp(by, 8), 1);
            }
            size_t off = 0;
            if (ok) ok = downloadImage(mvTex, off);
            if (ok && flush()) {
                ok = finishDownloadRG(mvTex, mv.xy, off);
            } else ok = false;
        } while (false);
        releaseTexture(prev);
        releaseTexture(curr);
        releaseTexture(mvTex);
        return ok;
    }

    bool reprojectHistory(const Image& historyColor, const MotionField& mv,
                          Image& reproj, Image& confidence) override {
        if (!ok_ || mv.width == 0 || mv.height == 0) return false;
        const bool hasHistory = !historyColor.empty();
        reproj = Image(mv.width, mv.height);
        confidence = Image(mv.width, mv.height);

        // motion field -> rgba32f (dx, dy, 0, 1); the kernel reads Tex1 .rg
        Image mvImg(mv.width, mv.height);
        for (size_t i = 0; i < size_t(mv.width) * mv.height; ++i) {
            mvImg.pixels[i * 4 + 0] = mv.xy[i * 2 + 0];
            mvImg.pixels[i * 4 + 1] = mv.xy[i * 2 + 1];
            mvImg.pixels[i * 4 + 2] = 0.f;
            mvImg.pixels[i * 4 + 3] = 1.f;
        }

        GpuTex hist, mvTex, rp, cf;
        bool ok = false;
        do {
            if (hasHistory && !allocTexture(historyColor.width, historyColor.height,
                                            DXGI_FORMAT_R32G32B32A32_FLOAT, hist)) break;
            if (!allocTexture(mv.width, mv.height, DXGI_FORMAT_R32G32B32A32_FLOAT, mvTex)) break;
            if (!allocTexture(mv.width, mv.height, DXGI_FORMAT_R32G32B32A32_FLOAT, rp)) break;
            if (!allocTexture(mv.width, mv.height, DXGI_FORMAT_R32G32B32A32_FLOAT, cf)) break;
            if (!beginFrame()) break;
            uint32_t pc[kRootConstWords] = {};
            pcF(pc, 0, hasHistory ? 1.0f : 0.0f);  // C[0].x = hasHistory
            pcU(pc, 4, mv.width);
            pcU(pc, 5, mv.height);
            pcU(pc, 6, hasHistory ? historyColor.width : 1u);
            pcU(pc, 7, hasHistory ? historyColor.height : 1u);
            const GpuTex* histPtr = hasHistory ? &hist : &dummyTex_;
            ok = uploadImage(mvImg, mvTex) &&
                 (hasHistory ? uploadImage(historyColor, hist) : true) &&
                 dispatch("odl_reproject", pc,
                          { bindTex(0, *histPtr), bindTex(1, mvTex),
                            bindBuf(2, rp), bindBuf(3, cf) },
                          divUp(mv.width, 8), divUp(mv.height, 8), 1);
            size_t offRp = 0, offCf = 0;
            if (ok) ok = downloadImage(rp, offRp) && downloadImage(cf, offCf);
            if (ok && flush()) {
                ok = finishDownload(rp, reproj, offRp) &&
                     finishDownload(cf, confidence, offCf);
            } else ok = false;
        } while (false);
        releaseTexture(hist);
        releaseTexture(mvTex);
        releaseTexture(rp);
        releaseTexture(cf);
        return ok;
    }

    bool temporalBlend(const Image& current, const Image& reproj,
                       const Image& confidence, float maxBlend, Image& out) override {
        if (!ok_ || current.empty()) return false;
        out = Image(current.width, current.height);

        GpuTex cur, rp, cf, dst;
        bool ok = false;
        do {
            if (!allocTexture(current.width, current.height, DXGI_FORMAT_R32G32B32A32_FLOAT, cur)) break;
            if (!allocTexture(reproj.width, reproj.height, DXGI_FORMAT_R32G32B32A32_FLOAT, rp)) break;
            if (!allocTexture(confidence.width, confidence.height, DXGI_FORMAT_R32G32B32A32_FLOAT, cf)) break;
            if (!allocTexture(current.width, current.height, DXGI_FORMAT_R32G32B32A32_FLOAT, dst)) break;
            if (!beginFrame()) break;
            uint32_t pc[kRootConstWords] = {};
            pcF(pc, 0, maxBlend);
            pcU(pc, 4, current.width);
            pcU(pc, 5, current.height);
            ok = uploadImage(current, cur) && uploadImage(reproj, rp) && uploadImage(confidence, cf) &&
                 dispatch("odl_temporal_blend", pc,
                          { bindTex(0, cur), bindTex(1, rp), bindTex(2, cf), bindBuf(0, dst) },
                          divUp(current.width, 8), divUp(current.height, 8), 1);
            size_t off = 0;
            if (ok) ok = downloadImage(dst, off);
            if (ok && flush()) ok = finishDownload(dst, out, off);
            else ok = false;
        } while (false);
        releaseTexture(cur);
        releaseTexture(rp);
        releaseTexture(cf);
        releaseTexture(dst);
        return ok;
    }

    // ---------------- game mode (Lanczos + temporal-blend fallback) ----------------
    bool beginGame(uint32_t renderW, uint32_t renderH, uint32_t outputW, uint32_t outputH) override {
        endGame();
        gRenderW_ = renderW; gRenderH_ = renderH;
        gOutW_ = outputW;    gOutH_ = outputH;
        if (!ok_) return true;      // mirror Metal: surface success, frames will fail
        if (renderW == 0 || renderH == 0 || outputW == 0 || outputH == 0) return false;
        if (!allocTexture(renderW, renderH, DXGI_FORMAT_R32G32B32A32_FLOAT, gColor_) ||
            !allocTexture(renderW, renderH, DXGI_FORMAT_R32G32B32A32_FLOAT, gHist_)  ||
            !allocTexture(renderW, renderH, DXGI_FORMAT_R32G32B32A32_FLOAT, gMv_)    ||
            !allocTexture(renderW, renderH, DXGI_FORMAT_R32G32B32A32_FLOAT, gRp_)    ||
            !allocTexture(renderW, renderH, DXGI_FORMAT_R32G32B32A32_FLOAT, gCf_)    ||
            !allocTexture(renderW, renderH, DXGI_FORMAT_R32G32B32A32_FLOAT, gBlend_) ||
            !allocTexture(outputW, renderH, DXGI_FORMAT_R32G32B32A32_FLOAT, gTmp_)   ||
            !allocTexture(outputW, outputH, DXGI_FORMAT_R32G32B32A32_FLOAT, gOut_)) {
            endGame();
            return false;
        }
        gHasOut_ = false;
        prevOut_ = Image();
        return true;
    }

    bool submitGameFrame(const GameFrameInput& in, GameFrameOutput& out) override {
        if (!ok_ || !in.color || !gOut_.valid() || !gColor_.valid()) return false;
        if (in.color->width != gRenderW_ || in.color->height != gRenderH_) return false;
        const auto t0 = std::chrono::steady_clock::now();

        // feedback: previous output downscaled to render resolution (host-side,
        // exactly the Vulkan fallback's contract)
        bool temporal = false;
        Image temporalHist;
        if (!in.resetHistory && gHasOut_ && in.motion &&
            in.motion->width == gRenderW_ && in.motion->height == gRenderH_ &&
            !prevOut_.empty() && gOutW_ > 0 && gRenderW_ > 0 && gOutW_ % gRenderW_ == 0) {
            const uint32_t factor = gOutW_ / gRenderW_;
            if (factor > 1) {
                auto low = image_downscale_box(prevOut_, factor);
                if (low) temporalHist = std::move(*low);
            } else {
                temporalHist = prevOut_;
            }
            temporal = (temporalHist.width == gRenderW_ && temporalHist.height == gRenderH_);
        }

        bool ok = false;
        if (beginFrame()) {
            uint32_t pc[kRootConstWords] = {};
            ok = uploadImage(*in.color, gColor_);
            const GpuTex* srcTex = &gColor_;
            if (ok && temporal) {
                Image mvImg(gRenderW_, gRenderH_);
                for (size_t i = 0; i < size_t(gRenderW_) * gRenderH_; ++i) {
                    mvImg.pixels[i * 4 + 0] = in.motion->xy[i * 2 + 0];
                    mvImg.pixels[i * 4 + 1] = in.motion->xy[i * 2 + 1];
                    mvImg.pixels[i * 4 + 2] = 0.f;
                    mvImg.pixels[i * 4 + 3] = 1.f;
                }
                pcF(pc, 0, 1.0f);                  // hasHistory
                pcU(pc, 4, gRenderW_);
                pcU(pc, 5, gRenderH_);
                pcU(pc, 6, gRenderW_);
                pcU(pc, 7, gRenderH_);
                ok = uploadImage(temporalHist, gHist_) && uploadImage(mvImg, gMv_) &&
                     dispatch("odl_reproject", pc,
                              { bindTex(0, gHist_), bindTex(1, gMv_),
                                bindBuf(2, gRp_), bindBuf(3, gCf_) },
                              divUp(gRenderW_, 8), divUp(gRenderH_, 8), 1);
                if (ok) {
                    std::memset(pc, 0, sizeof(pc));
                    pcF(pc, 0, kGameMaxBlend);
                    pcU(pc, 4, gRenderW_);
                    pcU(pc, 5, gRenderH_);
                    ok = dispatch("odl_temporal_blend", pc,
                                  { bindTex(0, gColor_), bindTex(1, gRp_),
                                    bindTex(2, gCf_), bindBuf(0, gBlend_) },
                                  divUp(gRenderW_, 8), divUp(gRenderH_, 8), 1);
                }
                srcTex = &gBlend_;
            }
            if (ok) {   // Lanczos3 to output resolution
                std::memset(pc, 0, sizeof(pc));
                pcF(pc, 0, float(gOutW_));
                pcU(pc, 4, gRenderW_);
                pcU(pc, 5, gRenderH_);
                pcU(pc, 6, gRenderH_);
                ok = dispatch("odl_upscale_h", pc, { bindTex(0, *srcTex), bindBuf(0, gTmp_) },
                              divUp(gOutW_, 8), divUp(gRenderH_, 8), 1);
            }
            if (ok) {
                std::memset(pc, 0, sizeof(pc));
                pcF(pc, 0, float(gOutH_));
                pcU(pc, 4, gOutW_);
                pcU(pc, 5, gRenderH_);
                pcU(pc, 6, gOutW_);
                ok = dispatch("odl_upscale_v", pc, { bindTex(0, gTmp_), bindBuf(0, gOut_) },
                              divUp(gOutW_, 8), divUp(gOutH_, 8), 1);
            }
            size_t off = 0;
            if (ok) ok = downloadImage(gOut_, off);
            if (ok && flush()) ok = finishDownload(gOut_, out.upscaled, off);
            else ok = false;
        }
        if (ok) {
            out.gpuMs = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            prevOut_ = out.upscaled;
            gHasOut_ = true;
        }
        return ok;
    }

    void endGame() override {
        releaseTexture(gColor_);
        releaseTexture(gHist_);
        releaseTexture(gMv_);
        releaseTexture(gRp_);
        releaseTexture(gCf_);
        releaseTexture(gBlend_);
        releaseTexture(gTmp_);
        releaseTexture(gOut_);
        gRenderW_ = gRenderH_ = gOutW_ = gOutH_ = 0;
        gHasOut_ = false;
        prevOut_ = Image();
    }

private:
    // =====================================================================
    // device + queue
    // =====================================================================
    bool createDevice() {
        if (opt_.enableDebugLayer) {
            ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
                debug->EnableDebugLayer();
        }

        HRESULT hr = ::CreateDXGIFactory1(IID_PPV_ARGS(&factory_));
        if (FAILED(hr)) { log_error("d3d12: CreateDXGIFactory1 failed (hr=0x%08lX)", unsigned long(hr)); return false; }

        auto makeDevice = [&](IDXGIAdapter1* ad) {
            return D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_));
        };

        if (opt_.preferWarp) {
            ComPtr<IDXGIAdapter1> warp;
            if (SUCCEEDED(factory_->EnumWarpAdapter(IID_PPV_ARGS(&warp))) &&
                SUCCEEDED(makeDevice(warp.Get()))) {
                adapter_ = warp;
                DXGI_ADAPTER_DESC1 desc{};
                warp->GetDesc1(&desc);
                info_.deviceName = narrow(desc.Description) + " (WARP)";
            }
        }
        if (!device_) {
            ComPtr<IDXGIAdapter1> best;
            DXGI_ADAPTER_DESC1 bestDesc{};
            for (UINT i = 0; factory_->EnumAdapters1(i, &best) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 d{};
                best->GetDesc1(&d);
                if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                if (SUCCEEDED(makeDevice(best.Get()))) { adapter_ = best; bestDesc = d; break; }
                best.Reset();
            }
            if (device_) info_.deviceName = narrow(bestDesc.Description);
        }
        if (!device_) {
            // WARP fallback (no hardware adapter could create a device)
            ComPtr<IDXGIAdapter1> warp;
            if (SUCCEEDED(factory_->EnumWarpAdapter(IID_PPV_ARGS(&warp))) &&
                SUCCEEDED(makeDevice(warp.Get()))) {
                adapter_ = warp;
                DXGI_ADAPTER_DESC1 desc{};
                warp->GetDesc1(&desc);
                info_.deviceName = narrow(desc.Description) + " (WARP)";
                log_warn("d3d12: falling back to the WARP software adapter");
            }
        }
        if (!device_) return false;

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)))) return false;
        return true;
    }

    bool createCommandResources() {
        if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                   IID_PPV_ARGS(&allocator_)))) return false;
        if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(),
                                              nullptr, IID_PPV_ARGS(&list_)))) return false;
        // a freshly created list is in the recording state; close it so the
        // first beginFrame() can Reset() it
        if (FAILED(list_->Close())) return false;
        if (FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) return false;
        fenceEvent_ = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return fenceEvent_ != nullptr;
    }

    // One unified root signature for every kernel (see the file header):
    //   param 0: descriptor table SRV t0..t7
    //   param 1: descriptor table UAV u0..u4
    //   param 2: root constants b0 (64 dwords)
    bool createRootSignature() {
        D3D12_DESCRIPTOR_RANGE srvRange{};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = kSrvTableRegs;
        srvRange.BaseShaderRegister = 0;
        srvRange.RegisterSpace = 0;
        srvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE uavRange{};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = kUavTableRegs;
        uavRange.BaseShaderRegister = 0;
        uavRange.RegisterSpace = 0;
        uavRange.OffsetInDescriptorsFromTableStart = 0;

        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable.NumDescriptorRanges = 1;
        params[0].DescriptorTable.pDescriptorRanges = &srvRange;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 1;
        params[1].DescriptorTable.pDescriptorRanges = &uavRange;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants.ShaderRegister = 0;      // b0
        params[2].Constants.RegisterSpace = 0;
        params[2].Constants.Num32BitValues = kRootConstWords;
        params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rsDesc{};
        rsDesc.NumParameters = 3;
        rsDesc.pParameters = params;
        rsDesc.NumStaticSamplers = 0;
        rsDesc.pStaticSamplers = nullptr;
        rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ComPtr<ID3DBlob> blob, err;
        if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                               &blob, &err))) {
            if (err) log_error("d3d12: root signature serialize: %s",
                               static_cast<const char*>(err->GetBufferPointer()));
            return false;
        }
        return SUCCEEDED(device_->CreateRootSignature(0, blob->GetBufferPointer(),
                                                      blob->GetBufferSize(),
                                                      IID_PPV_ARGS(&rootSig_)));
    }

    bool createDummies() {
        // 4-byte dummy buffer (SRV/UAV filler for unbound slots — the kernels'
        // hasX flags guarantee those slots are never dereferenced)
        if (!allocBuffer(4, D3D12_HEAP_TYPE_UPLOAD, dummyBuf_, false)) return false;
        std::memset(dummyBuf_.map, 0, 4);
        if (!allocTexture(1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, dummyTex_)) return false;
        if (!allocTexture(1, 1, DXGI_FORMAT_R32G32_FLOAT, dummyTex2_)) return false;

        // zero source for region clears (ViT qkv padding); committed DEFAULT
        // resources are zero-initialized — grow it inside a throwaway frame
        if (beginFrame()) {
            if (!ensureGpu(zeroBuf_, 65536)) { abortFrame(); return false; }
            if (!flush()) return false;
        } else return false;
        return true;
    }

    // =====================================================================
    // DirectML (dynamic load; GEMM accelerator, never a requirement)
    // =====================================================================
#ifdef OPENDLSS_D3D12_HAS_DML
    using DmlCreateDevice1Fn = HRESULT (WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS,
                                                 DML_FEATURE_LEVEL, REFIID, void**);
    struct DmlGemm {
        ComPtr<IDMLCompiledOperator> op;
        GpuBuffer persistent;      // compiled-op scratch that survives dispatches
        GpuBuffer temporary;       // per-execute scratch
        GpuBuffer initTemp;        // initializer-only scratch
        uint32_t descs = 0;        // RequiredDescriptorCount at execute
    };

    bool initDml() {
        if (!opt_.enableDirectML) return false;
        dmlLib_ = ::LoadLibraryW(L"DirectML.dll");
        if (!dmlLib_) { log_info("d3d12: DirectML.dll not found — GEMMs run in HLSL"); return false; }
        auto create = reinterpret_cast<DmlCreateDevice1Fn>(
            reinterpret_cast<void*>(::GetProcAddress(dmlLib_, "DMLCreateDevice1")));
        if (!create) { log_warn("d3d12: DirectML.dll has no DMLCreateDevice1"); return false; }
        // FLOAT16 tensors (our packed f16 buffers) need feature level 2_0
        HRESULT hr = create(device_.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                            DML_FEATURE_LEVEL_2_0, IID_PPV_ARGS(&dmlDevice_));
        if (FAILED(hr)) {
            log_info("d3d12: DMLCreateDevice1(DML_FEATURE_LEVEL_2_0) failed (hr=0x%08lX) — GEMMs run in HLSL",
                     unsigned long(hr));
            return false;
        }
        DML_FEATURE_QUERY_FEATURE_LEVELS q{};
        q.RequestedFeatureLevel = DML_FEATURE_LEVEL_2_0;
        if (FAILED(dmlDevice_->CheckFeatureSupport(DML_FEATURE_FEATURE_LEVELS, &q, sizeof(q))) ||
            q.SupportedFeatureLevel < DML_FEATURE_LEVEL_2_0) {
            log_warn("d3d12: DML device lacks feature level 2_0 — GEMMs run in HLSL");
            dmlDevice_.Reset();
            return false;
        }
        if (FAILED(dmlDevice_->CreateCommandRecorder(IID_PPV_ARGS(&dmlRecorder_)))) {
            dmlDevice_.Reset();
            return false;
        }
        return true;
    }

    // compiles + initializes a GEMM for one shape; the initializer dispatch is
    // recorded into the CURRENT command list (the caller guarantees a frame is
    // open) so it always precedes the first execute
    bool createGemmOp(DmlGemm& g, uint32_t tokens, uint32_t outCh, uint32_t inCh) {
        uint32_t aSizes[4] = { tokens, inCh, 1, 1 };
        uint32_t bSizes[4] = { outCh, inCh, 1, 1 };
        uint32_t oSizes[4] = { tokens, outCh, 1, 1 };
        DML_BUFFER_TENSOR_DESC a{}, b{}, o{};
        a.DataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        a.Sizes = aSizes; a.Strides = nullptr;             // row-major, dense
        a.TotalTensorSizeInBytes = uint64_t(tokens) * inCh * 2;
        a.Flags = DML_TENSOR_FLAGS_NONE;
        b.DataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        b.Sizes = bSizes; b.Strides = nullptr;             // weights are [outCh][inCh]
        b.TotalTensorSizeInBytes = uint64_t(outCh) * inCh * 2;
        b.Flags = DML_TENSOR_FLAGS_NONE;
        o.DataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        o.Sizes = oSizes; o.Strides = nullptr;
        o.TotalTensorSizeInBytes = uint64_t(tokens) * outCh * 2;
        o.Flags = DML_TENSOR_FLAGS_NONE;
        DML_TENSOR_DESC aD{ DML_TENSOR_TYPE_BUFFER, &a };
        DML_TENSOR_DESC bD{ DML_TENSOR_TYPE_BUFFER, &b };
        DML_TENSOR_DESC oD{ DML_TENSOR_TYPE_BUFFER, &o };

        DML_GEMM_OPERATOR_DESC gemm{};
        gemm.ATensor = &aD;
        gemm.BTensor = &bD;
        gemm.CTensor = nullptr;                            // beta = 0
        gemm.OutputTensor = &oD;
        gemm.TransA = DML_MATRIX_TRANSFORM_NONE;
        gemm.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;      // out = x * wᵀ
        gemm.Alpha = 1.0f;
        gemm.Beta = 0.0f;
        DML_OPERATOR_DESC opDesc{ DML_OPERATOR_GEMM, &gemm };

        ComPtr<IDMLOperator> op;
        if (FAILED(dmlDevice_->CreateOperator(&opDesc, IID_PPV_ARGS(&op)))) return false;
        if (FAILED(op->Compile(nullptr, 0, DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&g.op)))) return false;

        DML_BINDING_PROPERTIES props = g.op->GetBindingProperties();
        if (props.PersistentResourceSize &&
            !allocBuffer(props.PersistentResourceSize, D3D12_HEAP_TYPE_DEFAULT, g.persistent, true))
            return false;
        if (props.TemporaryResourceSize &&
            !allocBuffer(props.TemporaryResourceSize, D3D12_HEAP_TYPE_DEFAULT, g.temporary, true))
            return false;

        // one-time initialization of the compiled operator
        ComPtr<IDMLOperatorInitializer> init;
        if (FAILED(dmlDevice_->CreateOperatorInitializer(1, g.op.GetAddressOf(),
                                                         IID_PPV_ARGS(&init)))) return false;
        DML_BINDING_PROPERTIES iprops = init->GetBindingProperties();
        if (iprops.TemporaryResourceSize &&
            !allocBuffer(iprops.TemporaryResourceSize, D3D12_HEAP_TYPE_DEFAULT, g.initTemp, true))
            return false;

        uint32_t base = 0;
        if (!allocSlots(iprops.RequiredDescriptorCount, base)) return false;
        DML_BINDING_TABLE_DESC td{};
        td.Dispatchable = init.Get();
        td.CPUDescriptorHandle = slotCpu(base);
        td.GPUDescriptorHandle = slotGpu(base);
        td.SizeInDescriptors = iprops.RequiredDescriptorCount;
        ComPtr<IDMLBindingTable> table;
        if (FAILED(dmlDevice_->CreateBindingTable(&td, IID_PPV_ARGS(&table)))) return false;
        list_->SetDescriptorHeaps(1, csuHeap_.GetAddressOf());
        if (g.initTemp.valid()) {
            DML_BUFFER_BINDING tb{ g.initTemp.res.Get(), 0, g.initTemp.bytes };
            table->BindTemporaryResource(&tb);
        }
        if (g.persistent.valid()) {
            DML_BUFFER_BINDING pb{ g.persistent.res.Get(), 0, g.persistent.bytes };
            table->BindPersistentResource(&pb);
        }
        dmlRecorder_->RecordDispatch(list_.Get(), init.Get(), table.Get());
        uavBarrier();
        g.descs = props.RequiredDescriptorCount;
        return true;
    }

    // x [tokens x inCh] * wᵀ -> out [tokens x outCh], raw f16 (published by
    // the caller's odl_channel_gemm mode-2 pass)
    bool runGemmDml(const GpuBuffer& x, const GpuBuffer& w, GpuBuffer& out,
                    uint32_t tokens, uint32_t outCh, uint32_t inCh) {
        const std::string key = std::to_string(tokens) + "x" + std::to_string(outCh) +
                                "x" + std::to_string(inCh);
        auto it = dmlOps_.find(key);
        if (it == dmlOps_.end()) {
            DmlGemm g;
            if (!createGemmOp(g, tokens, outCh, inCh)) return false;
            it = dmlOps_.emplace(key, std::move(g)).first;
        }
        DmlGemm& g = it->second;

        uint32_t base = 0;
        if (!allocSlots(g.descs, base)) return false;
        DML_BINDING_TABLE_DESC td{};
        td.Dispatchable = g.op.Get();
        td.CPUDescriptorHandle = slotCpu(base);
        td.GPUDescriptorHandle = slotGpu(base);
        td.SizeInDescriptors = g.descs;
        ComPtr<IDMLBindingTable> table;
        if (FAILED(dmlDevice_->CreateBindingTable(&td, IID_PPV_ARGS(&table)))) return false;
        list_->SetDescriptorHeaps(1, csuHeap_.GetAddressOf());

        DML_BUFFER_BINDING ab{ x.res.Get(), 0, uint64_t(tokens) * inCh * 2 };
        DML_BUFFER_BINDING bb{ w.res.Get(), 0, uint64_t(outCh) * inCh * 2 };
        DML_BUFFER_BINDING ob{ out.res.Get(), 0, uint64_t(tokens) * outCh * 2 };
        DML_BUFFER_BINDING ins[2] = { ab, bb };
        table->BindInputs(2, ins);
        table->BindOutputs(1, &ob);
        if (g.temporary.valid()) {
            DML_BUFFER_BINDING tb{ g.temporary.res.Get(), 0, g.temporary.bytes };
            table->BindTemporaryResource(&tb);
        }
        if (g.persistent.valid()) {
            DML_BUFFER_BINDING pb{ g.persistent.res.Get(), 0, g.persistent.bytes };
            table->BindPersistentResource(&pb);
        }
        dmlRecorder_->RecordDispatch(list_.Get(), g.op.Get(), table.Get());
        uavBarrier();
        return true;
    }
#else  // !OPENDLSS_D3D12_HAS_DML
    bool initDml() {
        // directml.h not present at build time — the HLSL GEMM path covers
        // everything (documented in the file header)
        return false;
    }
#endif // OPENDLSS_D3D12_HAS_DML

    // =====================================================================
    // frame plumbing (one allocator + fence; mirrors Vulkan's submit+wait)
    // =====================================================================
    bool beginFrame() {
        if (!ok_) return false;
        if (frameOpen_) return true;
        if (FAILED(list_->Reset(allocator_.Get(), nullptr))) return false;
        frameOpen_ = true;
        return true;
    }

    bool flush() {
        if (!frameOpen_) return ok_;
        if (FAILED(list_->Close())) { frameOpen_ = false; return false; }
        frameOpen_ = false;
        ID3D12CommandList* lists[] = { list_.Get() };
        queue_->ExecuteCommandLists(1, lists);
        ++fenceValue_;
        if (FAILED(queue_->Signal(fence_.Get(), fenceValue_))) return false;
        if (fence_->GetCompletedValue() < fenceValue_) {
            if (FAILED(fence_->SetEventOnCompletion(fenceValue_, fenceEvent_))) return false;
            ::WaitForSingleObject(fenceEvent_, INFINITE);
        }
        if (FAILED(allocator_->Reset())) return false;
        heapNext_ = 0;                 // recycle descriptor slots (pool-reset semantics)
        downOff_ = 0;
        for (GpuTex* t : texRegistry_) // simultaneous-access textures decay to COMMON
            t->state = D3D12_RESOURCE_STATE_COMMON;
        return true;
    }

    void abortFrame() {
        // best-effort discard of the open list; every op calls this before
        // reporting failure
        if (frameOpen_ && list_) {
            list_->Reset(allocator_.Get(), nullptr);
            list_->Close();
        }
        frameOpen_ = false;
        heapNext_ = 0;
        downOff_ = 0;
        for (GpuTex* t : texRegistry_)
            t->state = D3D12_RESOURCE_STATE_COMMON;
    }

    // =====================================================================
    // buffers
    // =====================================================================
    bool allocBuffer(size_t bytes, D3D12_HEAP_TYPE heap, GpuBuffer& out, bool writable) {
        out = GpuBuffer{};
        out.bytes = bytes;
        out.upload = (heap == D3D12_HEAP_TYPE_UPLOAD);
        out.writable = writable && heap == D3D12_HEAP_TYPE_DEFAULT;

        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = heap;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Alignment = 0;
        rd.Width = uint64_t(bytes == 0 ? 4 : bytes);
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.SampleDesc.Quality = 0;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = out.writable ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                : D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES initial = out.upload ? D3D12_RESOURCE_STATE_GENERIC_READ
                                             : (heap == D3D12_HEAP_TYPE_READBACK
                                                    ? D3D12_RESOURCE_STATE_COPY_DEST
                                                    : D3D12_RESOURCE_STATE_COMMON);
        if (FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, initial,
                                                    nullptr, IID_PPV_ARGS(&out.res))))
            return false;

        if (heap == D3D12_HEAP_TYPE_UPLOAD || heap == D3D12_HEAP_TYPE_READBACK) {
            D3D12_RANGE none{ 0, 0 };
            if (FAILED(out.res->Map(0, &none, &out.map))) { releaseBuffer(out); return false; }
        }
        return true;
    }

    void releaseBuffer(GpuBuffer& b) {
        if (b.map && b.res) {
            D3D12_RANGE none{ 0, 0 };
            b.res->Unmap(0, &none);
        }
        b.res.Reset();
        b.map = nullptr;
        b.bytes = 0;
        b.upload = false;
        b.writable = false;
    }

    // host-writable, persistent-mapped, GPU-read-only (weights, features)
    bool ensureUpload(GpuBuffer& b, size_t bytes) {
        if (b.valid() && b.bytes >= bytes) return true;
        releaseBuffer(b);
        return allocBuffer(bytes, D3D12_HEAP_TYPE_UPLOAD, b, false);
    }

    // GPU-writable scratch; DEFAULT heap, UAV-capable. Committed resources are
    // zero-initialized by spec; we additionally record one typed-R32_UINT UAV
    // clear (makeBufferUavTyped / ClearUnorderedAccessViewUint) so every fresh
    // region is provably zero — that typed view is what the requirement's
    // "DXGI_FORMAT_R32_UINT typed UAV" refers to.
    bool ensureGpu(GpuBuffer& b, size_t bytes) {
        if (b.valid() && b.bytes >= bytes) return true;
        releaseBuffer(b);
        if (!allocBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, b, true)) return false;
        if (frameOpen_) clearBufferUav(b);
        return true;
    }

    // GPU->host readback target
    bool ensureReadback(GpuBuffer& b, size_t bytes) {
        if (b.valid() && b.bytes >= bytes) return true;
        releaseBuffer(b);
        return allocBuffer(bytes, D3D12_HEAP_TYPE_READBACK, b, false);
    }

    // one dword = one half pair, little-endian (Common.hlsli OdlLoadHalf layout)
    bool ensureTensor(const std::string& name, const GpuBuffer** out) {
        auto it = tensorBufs_.find(name);
        if (it != tensorBufs_.end()) { *out = &it->second; return true; }
        const TensorRecord* t = model_ ? model_->tensor(name) : nullptr;
        if (!t || t->f16.empty()) { *out = nullptr; return false; }
        GpuBuffer b;
        if (!ensureUpload(b, t->f16.size() * 2)) { *out = nullptr; return false; }
        uint32_t* w = static_cast<uint32_t*>(b.map);
        const size_t lanes = t->f16.size();
        for (size_t d = 0; d < lanes / 2; ++d)
            w[d] = uint32_t(t->f16[2 * d]) | (uint32_t(t->f16[2 * d + 1]) << 16);
        if (lanes & 1u)
            w[lanes / 2] = uint32_t(t->f16[lanes - 1]);   // odd tail padded with zero half
        auto res = tensorBufs_.emplace(name, std::move(b));
        *out = &res.first->second;
        return true;
    }

    const GpuBuffer* tensorPtr(const std::string& name) {
        const GpuBuffer* b = nullptr;
        ensureTensor(name, &b);
        return b;
    }

    // scratch ring for transition temporaries (max 3 live at once)
    GpuBuffer& takeScratch(size_t bytes) {
        for (int i = 0; i < 3; ++i) {
            if (scratchUsed_[i]) continue;
            scratchUsed_[i] = true;
            ensureGpu(scratch_[i], bytes);   // failure surfaces at first use
            return scratch_[i];
        }
        static GpuBuffer overflow;           // unreachable: graph uses <= 2 at once
        overflow = GpuBuffer{};
        return overflow;
    }

    // =====================================================================
    // textures (committed RGBA32F/RG32F, UAV-capable, simultaneous access)
    // =====================================================================
    bool allocTexture(uint32_t w, uint32_t h, DXGI_FORMAT fmt, GpuTex& out) {
        out = GpuTex{};
        out.w = w; out.h = h; out.fmt = fmt;
        out.state = D3D12_RESOURCE_STATE_COMMON;
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;          // committed resource
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Alignment = 0;
        rd.Width = w == 0 ? 1 : w;
        rd.Height = h == 0 ? 1 : h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = fmt;
        rd.SampleDesc.Count = 1;
        rd.SampleDesc.Quality = 0;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                   D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        if (FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&out.res))))
            return false;
        texRegistry_.push_back(&out);
        return true;
    }

    void releaseTexture(GpuTex& t) {
        for (size_t i = 0; i < texRegistry_.size(); ++i)
            if (texRegistry_[i] == &t) { texRegistry_[i] = texRegistry_.back(); texRegistry_.pop_back(); break; }
        t.res.Reset();
        t.w = t.h = 0;
        t.state = D3D12_RESOURCE_STATE_COMMON;
    }

    // THE barrier helper: every texture state change goes through here.
    // Simultaneous-access textures decay to COMMON when a command list
    // finishes; flush()/abortFrame() fold that decay back into the tracked
    // state, so tracked is always the real pre-barrier state.
    void transition(GpuTex& t, D3D12_RESOURCE_STATES to) {
        if (!t.valid() || t.state == to) return;
        D3D12_RESOURCE_TRANSITION_BARRIER tb{};
        tb.pResource = t.res.Get();
        tb.StateBefore = t.state;
        tb.StateAfter = to;
        tb.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        b.Transition = tb;
        list_->ResourceBarrier(1, &b);
        t.state = to;
    }

    // full write-after-write / read-after-write ordering between dispatches
    // (the D3D12 equivalent of the Vulkan backend's barrierComputeRW)
    void uavBarrier() {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        b.UAV.pResource = nullptr;                 // global UAV barrier
        list_->ResourceBarrier(1, &b);
    }

    // =====================================================================
    // descriptors (shader-visible CBV_SRV_UAV heap, 512 slots)
    // =====================================================================
    D3D12_CPU_DESCRIPTOR_HANDLE slotCpu(uint32_t slot) const {
        D3D12_CPU_DESCRIPTOR_HANDLE h = csuHeap_->GetCPUDescriptorHandleForHeapStart();
        h.ptr += SIZE_T(slot) * SIZE_T(csuInc_);
        return h;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE slotGpu(uint32_t slot) const {
        D3D12_GPU_DESCRIPTOR_HANDLE h = csuHeap_->GetGPUDescriptorHandleForHeapStart();
        h.ptr += uint64_t(slot) * uint64_t(csuInc_);
        return h;
    }

    bool allocSlots(uint32_t count, uint32_t& baseOut) {
        if (!frameOpen_) return false;
        if (heapNext_ + count > kCsuHeapSlots) {
            // heap exhausted mid-frame: sync once and keep recording on the
            // fresh list (the 512-slot budget is the task-mandated size; big
            // models pay a GPU wait every ~heap-full of dispatches)
            if (!flush()) return false;
            if (!beginFrame()) return false;
        }
        baseOut = heapNext_;
        heapNext_ += count;
        return true;
    }

    void makeBufferSrvRaw(const GpuBuffer& b, D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;                    // ByteAddressBuffer style
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = UINT(b.bytes / 4);
        d.Buffer.StructureByteStride = 0;
        d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        device_->CreateShaderResourceView(b.res.Get(), &d, h);
    }

    void makeBufferUavRaw(const GpuBuffer& b, D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;                    // RWByteAddressBuffer style
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = UINT(b.bytes / 4);
        d.Buffer.StructureByteStride = 0;
        d.Buffer.CounterOffsetInBytes = 0;
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(b.res.Get(), nullptr, &d, h);
    }

    // typed R32_UINT view of the same buffer — used by clearBufferUav()
    void makeBufferUavTyped(const GpuBuffer& b, D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_UINT;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = UINT(b.bytes / 4);
        d.Buffer.StructureByteStride = 0;
        d.Buffer.CounterOffsetInBytes = 0;
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        device_->CreateUnorderedAccessView(b.res.Get(), nullptr, &d, h);
    }

    void makeTexSrv(const GpuTex& t, D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = t.fmt;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Texture2D.MostDetailedMip = 0;
        d.Texture2D.MipLevels = 1;
        d.Texture2D.PlaneSlice = 0;
        device_->CreateShaderResourceView(t.res.Get(), &d, h);
    }

    void makeTexUav(const GpuTex& t, D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = t.fmt;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipSlice = 0;
        d.Texture2D.PlaneSlice = 0;
        device_->CreateUnorderedAccessView(t.res.Get(), nullptr, &d, h);
    }

    // zero a writable buffer through its typed R32_UINT UAV
    void clearBufferUav(const GpuBuffer& b) {
        if (!b.writable || !frameOpen_) return;
        uint32_t base = 0;
        if (!allocSlots(1, base)) return;
        makeBufferUavTyped(b, slotCpu(base));
        list_->SetDescriptorHeaps(1, csuHeap_.GetAddressOf());
        const UINT zeros[4] = { 0, 0, 0, 0 };
        list_->ClearUnorderedAccessViewUint(slotGpu(base), slotCpu(base), b.res.Get(), zeros, 0, nullptr);
        uavBarrier();
    }

    // =====================================================================
    // shaders + PSO cache
    // =====================================================================
    bool getPSO(const std::string& name, ID3D12PipelineState** out) {
        auto it = psoCache_.find(name);
        if (it != psoCache_.end()) { *out = it->second.Get(); return true; }
        const KernelSpec* spec = findKernel(name);
        if (!spec) { log_error("d3d12: unknown kernel %s", name.c_str()); return false; }
        const std::string path = shaderDir_ + "/" + spec->file;

        ComPtr<ID3DBlob> code, err;
        HRESULT hr = D3DCompileFromFile(widen(path).c_str(), nullptr,
                                        D3D_COMPILE_STANDARD_FILE_INCLUDE, spec->entry,
                                        "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
                                        &code, &err);
        if (FAILED(hr)) {
            log_error("d3d12: compile %s (%s) failed (hr=0x%08lX)%s%s", name.c_str(), path.c_str(),
                      unsigned long(hr), err ? ": " : "",
                      err ? static_cast<const char*>(err->GetBufferPointer()) : "");
            return false;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rootSig_.Get();
        pd.CS.pShaderBytecode = code->GetBufferPointer();
        pd.CS.BytecodeLength = code->GetBufferSize();
        pd.NodeMask = 0;
        pd.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        ComPtr<ID3D12PipelineState> pso;
        if (FAILED(device_->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)))) {
            log_error("d3d12: PSO failed for %s", name.c_str());
            return false;
        }
        psoCache_[name] = pso;
        *out = pso.Get();
        return true;
    }

    // =====================================================================
    // dispatch
    // =====================================================================
    // pc: 64 root-constant words (uint4 P[16] or float4 C[8]+pad, zero-filled).
    // args reference registers t<n> / u<n>; every allocated slot inside the
    // used span is prefilled with a dummy descriptor (kernels' hasX flags
    // guarantee unbound slots are never dereferenced — the Vulkan backend's
    // contract).
    bool dispatch(const char* name, const uint32_t pc[kRootConstWords],
                  const std::vector<BindArg>& args,
                  uint32_t gx, uint32_t gy, uint32_t gz) {
        if (!frameOpen_ && !beginFrame()) return false;
        if (gx == 0 || gy == 0 || gz == 0) return true;   // empty dispatch is a no-op

        uint32_t maxT = 0, maxU = 0;
        for (const BindArg& a : args) {
            if (a.buf || a.tex) {
                if (a.reg < kSrvTableRegs) maxT = std::max(maxT, a.reg);
                else maxU = std::max(maxU, a.reg - kSrvTableRegs);
            }
        }
        // slot layout per dispatch: [t0..tMaxT][t8' u0..uMaxU] — the UAV table
        // starts at slot 8 within the dispatch span
        const uint32_t span = std::max(maxT + 1u, kSrvTableRegs) + (maxU + 1u);
        uint32_t base = 0;
        if (!allocSlots(span, base)) return false;

        // prefill the span with dummies, then overwrite the bound slots
        for (uint32_t i = 0; i < std::max(maxT + 1u, kSrvTableRegs); ++i)
            makeBufferSrvRaw(dummyBuf_, slotCpu(base + i));
        for (uint32_t i = 0; i <= maxU; ++i)
            makeBufferUavRaw(dummyBuf_, slotCpu(base + kSrvTableRegs + i));

        for (const BindArg& a : args) {
            if (a.reg < kSrvTableRegs) {
                const uint32_t slot = base + a.reg;
                if (a.buf) makeBufferSrvRaw(*a.buf, slotCpu(slot));
                else if (a.tex) makeTexSrv(*a.tex, slotCpu(slot));
            } else {
                const uint32_t slot = base + kSrvTableRegs + (a.reg - kSrvTableRegs);
                if (a.buf) makeBufferUavRaw(*a.buf, slotCpu(slot));
                else if (a.tex) makeTexUav(*a.tex, slotCpu(slot));
            }
        }

        ID3D12PipelineState* pso = nullptr;
        if (!getPSO(name, &pso)) return false;
        list_->SetDescriptorHeaps(1, csuHeap_.GetAddressOf());
        list_->SetComputeRootSignature(rootSig_.Get());
        list_->SetComputeRootDescriptorTable(0, slotGpu(base));
        list_->SetComputeRootDescriptorTable(1, slotGpu(base + kSrvTableRegs));
        list_->SetComputeRoot32BitConstants(2, kRootConstWords, pc, 0);
        list_->Dispatch(gx, gy, gz);
        uavBarrier();
        return true;
    }

    // =====================================================================
    // image transfer (pitched staging; RGBA32F 16 B/px, RG32F 8 B/px)
    // =====================================================================
    bool uploadImage(const Image& img, GpuTex& dst) {
        if (!dst.valid() || img.pixels.empty()) return false;
        const uint32_t pitch = rowPitch(dst.w * 16);
        const size_t bytes = size_t(pitch) * dst.h;
        if (!ensureUpload(upStage_, bytes)) return false;
        for (uint32_t y = 0; y < dst.h; ++y)
            std::memcpy(static_cast<uint8_t*>(upStage_.map) + size_t(y) * pitch,
                        img.row(y), size_t(dst.w) * 16);

        transition(dst, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_SUBRESOURCE_FOOTPRINT fp{};
        fp.Format = dst.fmt;
        fp.Width = dst.w;
        fp.Height = dst.h;
        fp.Depth = 1;
        fp.RowPitch = pitch;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        placed.Offset = 0;
        placed.Footprint = fp;
        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = upStage_.res.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLoc.PlacedFootprint = placed;
        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = dst.res.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = 0;
        list_->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        transition(dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        return true;
    }

    bool downloadImage(const GpuTex& src, size_t& offsetOut) {
        if (!src.valid()) return false;
        const uint32_t px = (src.fmt == DXGI_FORMAT_R32G32_FLOAT) ? 8u : 16u;
        const uint32_t pitch = rowPitch(src.w * px);
        offsetOut = alignUp(uint32_t(downOff_), kTexOffsetAlign);
        const size_t end = size_t(offsetOut) + size_t(pitch) * src.h;
        if (!ensureReadback(downStage_, end)) return false;
        downOff_ = end;

        transition(const_cast<GpuTex&>(src), D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_SUBRESOURCE_FOOTPRINT fp{};
        fp.Format = src.fmt;
        fp.Width = src.w;
        fp.Height = src.h;
        fp.Depth = 1;
        fp.RowPitch = pitch;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        placed.Offset = offsetOut;
        placed.Footprint = fp;
        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = src.res.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = downStage_.res.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dstLoc.PlacedFootprint = placed;
        list_->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        transition(const_cast<GpuTex&>(src), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        return true;
    }

    bool finishDownload(const GpuTex& tex, Image& out, size_t offset) {
        if (!downStage_.valid() || !downStage_.map) return false;
        if (out.width != tex.w || out.height != tex.h) out = Image(tex.w, tex.h);
        const uint32_t pitch = rowPitch(tex.w * 16);
        const uint8_t* base = static_cast<const uint8_t*>(downStage_.map) + offset;
        for (uint32_t y = 0; y < tex.h; ++y)
            std::memcpy(out.row(y), base + size_t(y) * pitch, size_t(tex.w) * 16);
        return true;
    }

    // RG32F motion readback -> [y*w+x] * 2 floats
    bool finishDownloadRG(const GpuTex& tex, std::vector<float>& xy, size_t offset) {
        if (!downStage_.valid() || !downStage_.map) return false;
        if (xy.size() < size_t(tex.w) * tex.h * 2) xy.resize(size_t(tex.w) * tex.h * 2);
        const uint32_t pitch = rowPitch(tex.w * 8);
        const uint8_t* base = static_cast<const uint8_t*>(downStage_.map) + offset;
        for (uint32_t y = 0; y < tex.h; ++y) {
            const float* row = reinterpret_cast<const float*>(base + size_t(y) * pitch);
            for (uint32_t x = 0; x < tex.w; ++x) {
                xy[(size_t(y) * tex.w + x) * 2 + 0] = row[x * 2 + 0];
                xy[(size_t(y) * tex.w + x) * 2 + 1] = row[x * 2 + 1];
            }
        }
        return true;
    }

    // =====================================================================
    // graph steps (param packing follows the .hlsl P[] comments exactly)
    // =====================================================================
    bool runInputEmbed(GpuBuffer& state, uint32_t fw, uint32_t fh, uint32_t C) {
        const GpuBuffer* proj = tensorPtr("input_proj");
        if (!proj) { log_error("d3d12: missing input_proj"); return false; }
        uint32_t pc[kRootConstWords] = {};
        pcU(pc, 0, fw * fh);                       // P[0].x = tokens
        pcU(pc, 1, C);                             // P[0].y = C
        return dispatch("odl_input_embed", pc,
                        { bindBuf(4, featBuf_), bindBuf(5, *proj), bindBuf(1, state) },
                        divUp(fw * fh, 64), 1, 1);
    }

    bool runPool(const GpuBuffer& src, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh,
                 uint32_t C, GpuBuffer& dst) {
        uint32_t pc[kRootConstWords] = {};
        pcU(pc, 0, sw); pcU(pc, 1, sh); pcU(pc, 2, dw); pcU(pc, 3, dh);  // P[0]
        pcU(pc, 4, C);                                                   // P[1].x
        return dispatch("odl_pool2x2", pc, { bindBuf(0, src), bindBuf(0, dst) },
                        divUp(dw * dh, 64), 1, 1);
    }

    bool runGemm(const GpuBuffer& x, const std::string& wName, uint32_t tokens,
                 uint32_t outCh, uint32_t inCh, uint32_t pubMode, GpuBuffer& out) {
        const GpuBuffer* w = tensorPtr(wName);
        if (!w) { log_error("d3d12: missing tensor %s", wName.c_str()); return false; }
        const GpuBuffer* bias = tensorPtr(wName + ".b");   // transitions are biasless today

        // DirectML accelerates the big E4M3-published GEMMs; the raw-f16 ones
        // stay in the HLSL kernel so their rounding matches the reference.
        if (dmlOk() && pubMode == 0 &&
            uint64_t(outCh) * inCh >= kDmlMinWeightElems) {
            if (runGemmDml(x, *w, out, tokens, outCh, inCh)) {
                // publish pass: odl_channel_gemm mode 2 quantizes in place
                uint32_t pc[kRootConstWords] = {};
                pcU(pc, 0, tokens);
                pcU(pc, 1, outCh);
                pcU(pc, 2, inCh);
                pcU(pc, 3, 0);                             // pubMode (unused in mode 2)
                pcU(pc, 4, 0);                             // hasBias
                pcU(pc, 5, 2);                             // mode 2 = publish
                return dispatch("odl_channel_gemm", pc, { bindBuf(0, out) },
                                divUp(tokens * (outCh / 2), 64), 1, 1);
            }
            log_warn("d3d12: DML GEMM %s failed — falling back to HLSL", wName.c_str());
        }

        uint32_t pc[kRootConstWords] = {};
        pcU(pc, 0, tokens);                                // P[0] = (tokens, outCh, inCh, pubMode)
        pcU(pc, 1, outCh);
        pcU(pc, 2, inCh);
        pcU(pc, 3, pubMode);
        pcU(pc, 4, bias ? 1u : 0u);                        // P[1].x = hasBias
        pcU(pc, 5, 0);                                     // P[1].y = mode 0 (GEMM)
        std::vector<BindArg> args = { bindBuf(0, x), bindBuf(1, *w) };
        if (bias) args.push_back(bindBuf(2, *bias));
        args.push_back(bindBuf(0, out));                   // u0
        return dispatch("odl_channel_gemm", pc, args,
                        divUp(tokens * (outCh / 2), 64), 1, 1);
    }

    bool runUpsample(const GpuBuffer& src, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh,
                     uint32_t C, GpuBuffer& dst) {
        uint32_t pc[kRootConstWords] = {};
        pcU(pc, 0, sw); pcU(pc, 1, sh); pcU(pc, 2, dw); pcU(pc, 3, dh);
        pcU(pc, 4, C);
        return dispatch("odl_nearest_upsample2x", pc, { bindBuf(0, src), bindBuf(0, dst) },
                        divUp(dw * dh * (C / 2), 64), 1, 1);
    }

    bool runDecoderSkip(const GpuBuffer& up, const GpuBuffer& skip, bool hasSkip,
                        const GpuBuffer* scale, const GpuBuffer* inScale,
                        uint32_t tokens, uint32_t C, GpuBuffer& out) {
        uint32_t pc[kRootConstWords] = {};
        pcU(pc, 0, tokens);                                // P[0] = (tokens, C, -, -)
        pcU(pc, 1, C);
        pcU(pc, 4, hasSkip ? 1u : 0u);                     // P[1] = (hasSkip, hasScale, hasInScale)
        pcU(pc, 5, scale ? 1u : 0u);
        pcU(pc, 6, inScale ? 1u : 0u);
        std::vector<BindArg> args = { bindBuf(0, up) };
        if (hasSkip && skip.valid()) args.push_back(bindBuf(1, skip));
        if (scale) args.push_back(bindBuf(2, *scale));
        if (inScale) args.push_back(bindBuf(3, *inScale));
        args.push_back(bindBuf(0, out));
        return dispatch("odl_decoder_skip", pc, args,
                        divUp(tokens * (C / 2), 64), 1, 1);
    }

    bool runBlock(const BlockScheduleEntry& e, GpuBuffer& state) {
        const uint32_t tokens = e.levelWidth * e.levelHeight;
        const uint32_t C = e.channels;
        const std::string b = std::to_string(e.index);

        // 1. FFN -> ffnOut (f16 storage, pre-skip)
        uint32_t hidden = 0, paths = 0, hasW3 = 0;
        if (C == 32 || e.isViT) {                  // dense: C -> 128/4096 -> C
            hidden = e.isViT ? 4096u : 128u;
            paths = 1;
            hasW3 = 0;
        } else {                                   // wide: C/32 paths, C -> 128 -> 32 + C->C
            hidden = 128;
            paths = C / 32;
            hasW3 = 1;
        }
        {
            const GpuBuffer* w1 = tensorPtr("block" + b + ".layer0.w1");
            const GpuBuffer* b1 = tensorPtr("block" + b + ".layer0.b1");
            const GpuBuffer* w2 = tensorPtr("block" + b + ".layer0.w2");
            const GpuBuffer* b2 = tensorPtr("block" + b + ".layer0.b2");
            const GpuBuffer* w3 = hasW3 ? tensorPtr("block" + b + ".layer0.w3") : nullptr;
            const GpuBuffer* b3 = hasW3 ? tensorPtr("block" + b + ".layer0.b3") : nullptr;
            if (!w1 || !w2) {
                log_error("d3d12: block %u missing FFN weights", e.index);
                return false;
            }
            uint32_t pc[kRootConstWords] = {};
            pcU(pc, 0, tokens);                    // P[0] = (tokens, C, hidden, paths)
            pcU(pc, 1, C);
            pcU(pc, 2, hidden);
            pcU(pc, 3, paths);
            pcU(pc, 4, b1 ? 1u : 0u);              // P[1] = (hasB1, hasB2, hasW3)
            pcU(pc, 5, b2 ? 1u : 0u);
            pcU(pc, 6, hasW3);
            std::vector<BindArg> args = {
                bindBuf(0, state), bindBuf(1, *w1), bindBuf(3, *w2),
            };
            if (b1) args.push_back(bindBuf(2, *b1));
            if (b2) args.push_back(bindBuf(4, *b2));
            if (w3) args.push_back(bindBuf(5, *w3));
            if (b3) args.push_back(bindBuf(6, *b3));
            args.push_back(bindBuf(0, ffnOut_));   // u0
            if (!dispatch("odl_ffn", pc, args, tokens, 1, 1)) return false;
        }

        // 2. block residual: y = pub(x * ffnScale + ffn) -> projIn
        {
            const GpuBuffer* s = tensorPtr("block" + b + ".layer0.scale_ffn");
            uint32_t pc[kRootConstWords] = {};
            pcU(pc, 0, tokens);                    // P[0] = (tokens, C, rawF16, -)
            pcU(pc, 1, C);
            pcU(pc, 2, (C == 32 && !e.isViT) ? 1u : 0u);
            pcU(pc, 4, s ? 1u : 0u);               // P[1].x = hasScale
            std::vector<BindArg> args = {
                bindBuf(0, state), bindBuf(1, ffnOut_),
            };
            if (s) args.push_back(bindBuf(2, *s));
            args.push_back(bindBuf(0, projIn_));   // u0
            if (!dispatch("odl_block_skip", pc, args, divUp(tokens * (C / 2), 64), 1, 1))
                return false;
        }

        // 3. attention
        const GpuBuffer* qscale = tensorPtr("block" + b + ".layer1.qscale");
        if (e.isViT) {
            // qkv GEMM over 64-padded tokens (padding rows must read as zero),
            // raw f16 publication; the kernel applies the exp(0) correction.
            const uint32_t padded = (tokens + 63u) & ~63u;
            if (padded != tokens) {
                // zero the padding rows of projIn through a device-side copy
                // from the never-written zeroBuf_ (DEFAULT-heap buffers cannot
                // be memset from the host)
                const size_t padBytes = size_t(padded - tokens) * C * 2;
                if (!ensureGpu(zeroBuf_, padBytes)) return false;
                list_->CopyBufferRegion(projIn_.res.Get(), uint64_t(tokens) * C * 2,
                                        zeroBuf_.res.Get(), 0, uint64_t(padBytes));
                uavBarrier();
            }
            const GpuBuffer* qkvT = tensorPtr("block" + b + ".layer1.qkv");
            if (!qkvT) { log_error("d3d12: block %u missing qkv", e.index); return false; }
            {
                uint32_t pc[kRootConstWords] = {};
                pcU(pc, 0, padded);                // P[0] = (tokens, outCh=3C, inCh=C, pubMode=1)
                pcU(pc, 1, 3 * C);
                pcU(pc, 2, C);
                pcU(pc, 3, 1);                     // raw f16
                pcU(pc, 4, 0);
                pcU(pc, 5, 0);
                if (!dispatch("odl_channel_gemm", pc,
                              { bindBuf(0, projIn_), bindBuf(1, *qkvT), bindBuf(0, qkvBuf_) },
                              divUp(padded * (3 * C / 2), 64), 1, 1))
                    return false;
            }
            uint32_t pc[kRootConstWords] = {};
            pcU(pc, 0, tokens);                    // P[0] = (tokensReal, C, heads, -)
            pcU(pc, 1, C);
            pcU(pc, 2, e.heads);
            pcU(pc, 4, qscale ? 1u : 0u);          // P[1].x = hasQscale
            std::vector<BindArg> args = { bindBuf(0, qkvBuf_) };
            if (qscale) args.push_back(bindBuf(1, *qscale));
            args.push_back(bindBuf(0, attnOut_));  // u0
            if (!dispatch("odl_global_attention", pc, args, divUp(tokens, 64), 1, 1))
                return false;
        } else {
            // QKV GEMM (E4M3-published output), then window attention
            const GpuBuffer* qkvT = tensorPtr("block" + b + ".layer1.qkv");
            if (!qkvT) { log_error("d3d12: block %u missing qkv", e.index); return false; }
            {
                uint32_t pc[kRootConstWords] = {};
                pcU(pc, 0, tokens);
                pcU(pc, 1, 3 * C);
                pcU(pc, 2, C);
                pcU(pc, 3, 0);                     // E4M3 published
                pcU(pc, 4, 0);
                pcU(pc, 5, 0);
                if (!dispatch("odl_channel_gemm", pc,
                              { bindBuf(0, projIn_), bindBuf(1, *qkvT), bindBuf(0, qkvBuf_) },
                              divUp(tokens * (3 * C / 2), 64), 1, 1))
                    return false;
            }
            const GpuBuffer* prior = tensorPtr("block" + b + ".layer1.prior");
            uint32_t pc[kRootConstWords] = {};
            pcU(pc, 0, e.levelWidth);              // P[0] = (levelW, levelH, C, heads)
            pcU(pc, 1, e.levelHeight);
            pcU(pc, 2, C);
            pcU(pc, 3, e.heads);
            pcU(pc, 4, e.windowPhase);             // P[1] = (phase, hasPrior, hasQscale)
            pcU(pc, 5, prior ? 1u : 0u);
            pcU(pc, 6, qscale ? 1u : 0u);
            std::vector<BindArg> args = { bindBuf(0, qkvBuf_) };
            if (prior) args.push_back(bindBuf(1, *prior));
            if (qscale) args.push_back(bindBuf(2, *qscale));
            args.push_back(bindBuf(0, attnOut_));  // u0
            const uint32_t shift = (e.windowPhase == 0) ? 0u : 4u;
            const uint32_t gridX = (e.levelWidth + shift + 7u) / 8u;
            const uint32_t gridY = (e.levelHeight + shift + 7u) / 8u;
            if (!dispatch("odl_window_attention", pc, args, gridX * gridY, 1, 1))
                return false;
        }

        // 3.5 projection GEMM over the attention output (raw f16), then
        // 4. epilogue: out = pub(y * attnScale + projOut) -> state
        {
            const GpuBuffer* projT = tensorPtr("block" + b + ".layer1.proj");
            if (!projT) { log_error("d3d12: block %u missing attention proj", e.index); return false; }
            uint32_t pc[kRootConstWords] = {};
            pcU(pc, 0, tokens);
            pcU(pc, 1, C);
            pcU(pc, 2, C);
            pcU(pc, 3, 1);                         // raw f16 (reference)
            pcU(pc, 4, 0);
            pcU(pc, 5, 0);
            if (!dispatch("odl_channel_gemm", pc,
                          { bindBuf(0, attnOut_), bindBuf(1, *projT), bindBuf(0, projOut_) },
                          divUp(tokens * (C / 2), 64), 1, 1))
                return false;
        }
        {
            const GpuBuffer* s = tensorPtr("block" + b + ".layer0.scale_attn");
            uint32_t pc[kRootConstWords] = {};
            pcU(pc, 0, tokens);                    // P[0] = (tokens, C, hasScale, -)
            pcU(pc, 1, C);
            pcU(pc, 2, s ? 1u : 0u);
            std::vector<BindArg> args = {
                bindBuf(0, projIn_), bindBuf(1, projOut_),
            };
            if (s) args.push_back(bindBuf(2, *s));
            args.push_back(bindBuf(0, state));     // u0
            if (!dispatch("odl_block_epilogue", pc, args, divUp(tokens * (C / 2), 64), 1, 1))
                return false;
        }
        return true;
    }

    bool dmlOk() const {
#ifdef OPENDLSS_D3D12_HAS_DML
        return dmlDevice_ != nullptr && dmlRecorder_ != nullptr;
#else
        return false;
#endif
    }

    // =====================================================================
    // state
    // =====================================================================
    D3D12BackendOptions opt_;
    BackendInfo info_{};
    bool ok_ = false;
    std::string shaderDir_;

    ComPtr<IDXGIFactory4> factory_;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    uint64_t fenceValue_ = 0;
    bool frameOpen_ = false;

    ComPtr<ID3D12DescriptorHeap> csuHeap_;
    uint32_t csuInc_ = 0;
    uint32_t heapNext_ = 0;
    ComPtr<ID3D12RootSignature> rootSig_;
    std::map<std::string, ComPtr<ID3D12PipelineState>> psoCache_;

    GpuBuffer dummyBuf_;           // 4-byte SRV/UAV filler for unbound slots
    GpuBuffer zeroBuf_;            // never-written zeros for device-side region clears
    GpuTex dummyTex_;              // 1x1 RGBA32F (absent history)
    GpuTex dummyTex2_;             // 1x1 RG32F (u4 filler)
    GpuBuffer upStage_;            // host -> texture staging (UPLOAD)
    GpuBuffer downStage_;          // texture -> host staging (READBACK)
    size_t downOff_ = 0;

    std::unique_ptr<Model> model_;
    std::map<std::string, GpuBuffer> tensorBufs_;

    // neural-graph scratch (grown on demand, reused across frames)
    GpuBuffer featBuf_;
    GpuBuffer stateA_, stateB_;
    GpuBuffer ffnOut_;             // pre-skip FFN result (f16 storage)
    GpuBuffer projIn_;             // y (block residual result)
    GpuBuffer qkvBuf_;
    GpuBuffer attnOut_;
    GpuBuffer projOut_;
    GpuBuffer headBuf_;            // [tokens][4] f32 (DEFAULT)
    GpuBuffer headReadback_;
    GpuBuffer scratch_[3];
    bool scratchUsed_[3] = {};

    // live-texture registry: flush()/abortFrame() fold the implicit
    // COMMON-decay of simultaneous-access textures back into the tracking
    std::vector<GpuTex*> texRegistry_;

#ifdef OPENDLSS_D3D12_HAS_DML
    HMODULE dmlLib_ = nullptr;
    ComPtr<IDMLDevice> dmlDevice_;
    ComPtr<IDMLCommandRecorder> dmlRecorder_;
    std::map<std::string, DmlGemm> dmlOps_;
#endif

    // game mode state
    uint32_t gRenderW_ = 0, gRenderH_ = 0, gOutW_ = 0, gOutH_ = 0;
    GpuTex gColor_, gHist_, gMv_, gRp_, gCf_, gBlend_, gTmp_, gOut_;
    Image prevOut_;
    bool gHasOut_ = false;
};

} // namespace opendlss

// ---------------------------------------------------------------------------
// factories (d3d12_backend.h contract: the instance is always returned, inert
// with all-ops-fail when no D3D12 device could be created)
// ---------------------------------------------------------------------------
namespace opendlss {

std::unique_ptr<IBackend> create_d3d12_backend() {
    return std::make_unique<D3D12Backend>(D3D12BackendOptions{});
}

std::unique_ptr<IBackend> create_d3d12_backend(const D3D12BackendOptions& options) {
    return std::make_unique<D3D12Backend>(options);
}

} // namespace opendlss
