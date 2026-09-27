// vulkan_backend.cpp — the Linux Vulkan backend (OpenDLSS-NR MetalFX).
//
// Full IBackend implementation over the runtime-loaded Vulkan loader
// (vk_boot.h). The Swin/ViT neural graph runs in the project's GLSL compute
// kernels (platforms/linux/shaders/*.comp, compiled to SPIR-V by the build or
// shipped prebuilt):
//
//   * token tensors are SSBOs of uint32 with ONE half per word (low 16 bits),
//     exactly matching the `uint x[]` buffers every kernel declares — the
//     2x memory overhead buys race-free writes and defined RTNE rounding on
//     every driver (see Common.glsl's header note);
//   * f32 SSBOs where a kernel says so (odl_ffn binding 11, odl_head binding 5,
//     odl_block_skip binding 1);
//   * storage images rgba32f for the media ops (denoise/upscale/sharpen/
//     motion/reproject/blend/composite);
//   * one descriptor set per kernel invocation (allocated from a pool that is
//     reset after every queue submission), pipelines cached by kernel name;
//   * push constants: the kernels' shared 16-word block, always pushed as the
//     full 64 bytes with unused words zero (Common.glsl convention).
//
// The block/transition driver mirrors MetalBackend::runNeuralGraph with two
// deliberate corrections (see worklog):
//   * transition tensor names use the CPU reference convention — the field
//     block transitions are "trans_f.up" / "trans_f.down" (Metal's
//     "trans-1.*" names never resolve, silently skipping the GEMM tensors);
//   * channel transitions (vitin.up / vitout.down / pool+up / down+upsample)
//     write into the opposite ping-pong state buffer instead of in place, so
//     the saved level skips are never clobbered by a later transition.
//
// Every op checks ok_ and returns false (or the identity) without crashing;
// the constructor marks ok_=false and the factory returns nullptr when no
// Vulkan device can be initialized so the registry falls back to the CPU.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "vulkan_backend.h"

#include "opendlss/fp16.h"
#include "opendlss/logging.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>
#include <limits.h>

#include "vk_boot.h"

namespace opendlss {

namespace {

using namespace vk;
using vk::Boot;

constexpr uint32_t kMaxKernelBindings = 12;
constexpr float    kGameMaxBlend      = 0.85f;   // temporal cap for the game fallback
constexpr uint32_t kPushWords         = 16;      // Common.glsl push-constant block
constexpr uint32_t kPushBytes         = kPushWords * 4;

inline uint32_t divUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// kernel -> binding map (from the .comp files; set = 0 everywhere)
// ---------------------------------------------------------------------------
struct KernelSpec {
    const char* name;
    uint32_t nb;
    struct { uint8_t b; bool img; } bd[kMaxKernelBindings];
};

const KernelSpec kKernels[] = {
    // graph kernels (SSBOs; one half per uint32 word unless noted f32)
    { "odl_input_embed",       3, {{0,false},{1,false},{2,false}} },
    { "odl_channel_gemm",      4, {{0,false},{1,false},{2,false},{7,false}} },
    { "odl_ffn",               8, {{0,false},{1,false},{2,false},{3,false},{4,false},{5,false},{6,false},{11,false}} },
    { "odl_window_attention",  4, {{0,false},{1,false},{2,false},{7,false}} },
    { "odl_global_attention",  3, {{0,false},{1,false},{5,false}} },
    { "odl_block_skip",        4, {{0,false},{1,false},{2,false},{6,false}} },
    { "odl_block_epilogue",    4, {{0,false},{1,false},{2,false},{5,false}} },
    { "odl_pool2x2",           2, {{0,false},{4,false}} },
    { "odl_nearest_upsample2x",2, {{0,false},{4,false}} },
    { "odl_decoder_skip",      5, {{0,false},{1,false},{2,false},{3,false},{5,false}} },
    { "odl_head",              4, {{0,false},{1,false},{2,false},{5,false}} },
    // media kernels (storage images; preprocess also touches the feature SSBO)
    { "odl_preprocess",        3, {{0,true},{1,true},{3,false}} },
    { "odl_denoise",           2, {{0,true},{1,true}} },
    { "odl_upscale_h",         2, {{0,true},{1,true}} },
    { "odl_upscale_v",         2, {{0,true},{1,true}} },
    { "odl_sharpen",           2, {{0,true},{1,true}} },
    { "odl_motion",            3, {{0,true},{1,true},{2,true}} },
    { "odl_reproject",         4, {{0,true},{1,true},{2,true},{3,true}} },
    { "odl_temporal_blend",    4, {{0,true},{1,true},{2,true},{3,true}} },
    { "odl_head_composite",    4, {{0,true},{1,true},{2,true},{3,true}} },
};

const KernelSpec* findKernel(const std::string& name) {
    for (const KernelSpec& k : kKernels)
        if (name == k.name) return &k;
    return nullptr;
}

// ---------------------------------------------------------------------------
// resource wrappers
// ---------------------------------------------------------------------------
struct Buf {
    vk::VkBuffer      buf = VK_NULL_HANDLE;
    vk::VkDeviceMemory mem = VK_NULL_HANDLE;
    void*  map = nullptr;
    size_t bytes = 0;
    bool   coherent = true;
    std::string tensorName;    // set for weight buffers (empty otherwise)
    bool valid() const { return buf != VK_NULL_HANDLE; }
};

struct Img {
    vk::VkImage       image = VK_NULL_HANDLE;
    vk::VkImageView   view  = VK_NULL_HANDLE;
    vk::VkDeviceMemory mem  = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    bool valid() const { return image != VK_NULL_HANDLE; }
};

struct BindArg {
    uint32_t binding = 0;
    const Buf* buf = nullptr;
    const Img* img = nullptr;
    BindArg() = default;
    BindArg(uint32_t b, const Buf* bf, const Img* im) : binding(b), buf(bf), img(im) {}
};

// push-constant helpers
inline void pcU(uint32_t* pc, uint32_t i, uint32_t v) { pc[i] = v; }
inline void pcF(uint32_t* pc, uint32_t i, float v) { std::memcpy(&pc[i], &v, 4); }

// ---------------------------------------------------------------------------
// shader directory resolution:
//   $OPENDLSS_VK_SHADER_DIR -> "shaders_spv" next to the executable ->
//   the build-tree directory (baked at compile time) -> /usr/local/share
// ---------------------------------------------------------------------------
std::string resolveShaderDir() {
    if (const char* env = std::getenv("OPENDLSS_VK_SHADER_DIR"))
        if (env[0] != '\0') return env;

    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string exe(buf);
        const size_t slash = exe.rfind('/');
        if (slash != std::string::npos) {
            std::string dir = exe.substr(0, slash + 1) + "shaders_spv";
            std::string probe = dir + "/odl_head.spv";
            std::FILE* f = std::fopen(probe.c_str(), "rb");
            if (f) { std::fclose(f); return dir; }
        }
    }
#ifdef OPENDLSS_VK_SHADER_BUILD_DIR
    {
        std::string dir = OPENDLSS_VK_SHADER_BUILD_DIR;
        std::FILE* f = std::fopen((dir + "/odl_head.spv").c_str(), "rb");
        if (f) { std::fclose(f); return dir; }
    }
#endif
    return "/usr/local/share/opendlss/shaders_spv";
}

bool readSpirv(const std::string& path, std::vector<uint32_t>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 4) != 0) { std::fclose(f); return false; }
    out.resize(size_t(sz) / 4);
    const size_t got = std::fread(out.data(), 1, size_t(sz), f);
    std::fclose(f);
    return got == size_t(sz);
}

} // namespace

// ---------------------------------------------------------------------------
// VulkanBackend
// ---------------------------------------------------------------------------
class VulkanBackend final : public IBackend {
public:
    VulkanBackend() {
        std::string deviceName;
        if (!vk::bootInit(vk_, &deviceName)) {
            log_error("vulkan: initialization failed (%s backend unavailable)",
                      "falling back to the CPU reference at the registry");
            return;
        }
        shaderDir_ = resolveShaderDir();

        if (!createCommandResources()) return;
        if (!createDescriptorPool())  return;
        if (!allocBuffer(4, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, dummyBuf_)) return;
        std::memset(dummyBuf_.map, 0, 4);          // 1-word zero dummy
        if (!allocImage(1, 1, dummyImg_))          return;   // 1x1 dummy storage image

        info_.kind = BackendKind::Vulkan;
        info_.name = "vulkan";
        info_.deviceName = deviceName;
        info_.neuralGraph = true;
        info_.fp16Storage = false;                 // half storage, f32 arithmetic
        info_.e4m3Quant = true;                    // kernels publish through E4M3
        info_.maxTextureDim = vk_.props.limits.maxImageDimension2D;
        info_.details = "Vulkan compute graph (SPIR-V), Lanczos spatial scaler + "
                        "temporal-blend game fallback";
        ok_ = true;
        log_info("vulkan: %s (api %u.%u), shaders: %s", deviceName.c_str(),
                 (vk_.apiVersion >> 22) & 0x3FF, (vk_.apiVersion >> 12) & 0x3FF,
                 shaderDir_.c_str());
    }

    ~VulkanBackend() override { destroyAll(); }

    const BackendInfo& info() const override { return info_; }

    // ---------------- image ops ----------------
    bool denoiseSpatial(const Image& input, Image& output, float strength) override {
        if (!ok_ || strength <= 0.0f) { output = input; return true; }
        if (input.empty()) { output = input; return true; }

        Img src, dst;
        if (!allocImage(input.width, input.height, src)) return false;
        if (!allocImage(input.width, input.height, dst)) { destroyImg(src); return false; }

        bool ok = false;
        if (beginFrame()) {
            uint32_t pc[kPushWords] = {};
            pcF(pc, 0, strength);
            ok = recordImageUpload(input, src) &&
                 dispatchKernel("odl_denoise", pc,
                                {BindArg(0, nullptr, &src), BindArg(1, nullptr, &dst)},
                                divUp(input.width, 16), divUp(input.height, 16), 1);
            size_t off = 0;
            if (ok) ok = recordImageDownload(dst, off);
            if (endFrame() && ok) ok = finishImageDownload(dst, output, off);
            else ok = false;
        }
        destroyImg(src);
        destroyImg(dst);
        return ok;
    }

    bool upscaleSpatial(const Image& input, Image& output, uint32_t factor, float sharpen) override {
        if (!ok_) { output = input; return false; }
        if (factor == 0 || input.empty()) { output = input; return false; }
        const uint32_t OW = input.width * factor, OH = input.height * factor;
        output = Image(OW, OH);

        Img src, tmp, up, fin;
        bool ok = false;
        do {
            if (!allocImage(input.width, input.height, src)) break;
            if (!allocImage(OW, input.height, tmp)) break;
            if (!allocImage(OW, OH, up)) break;
            const bool doSharpen = sharpen > 0.0f;
            if (doSharpen && !allocImage(OW, OH, fin)) break;

            if (!beginFrame()) break;
            uint32_t pc[kPushWords] = {};
            ok = recordImageUpload(input, src);
            if (ok) {   // 1. Lanczos3 horizontal pass
                pcU(pc, 0, OW);
                ok = dispatchKernel("odl_upscale_h", pc, {{0, nullptr, &src}, {1, nullptr, &tmp}},
                                    divUp(OW, 16), divUp(input.height, 16), 1);
            }
            if (ok) {   // 2. Lanczos3 vertical pass
                pcU(pc, 0, OH);
                ok = dispatchKernel("odl_upscale_v", pc, {{0, nullptr, &tmp}, {1, nullptr, &up}},
                                    divUp(OW, 16), divUp(OH, 16), 1);
            }
            const Img* finalTex = &up;
            if (ok && doSharpen) {   // 3. adaptive sharpen at output res
                pcF(pc, 0, sharpen);
                ok = dispatchKernel("odl_sharpen", pc, {{0, nullptr, &up}, {1, nullptr, &fin}},
                                    divUp(OW, 16), divUp(OH, 16), 1);
                finalTex = &fin;
            }
            size_t off = 0;
            if (ok) ok = recordImageDownload(*finalTex, off);
            if (endFrame() && ok) ok = finishImageDownload(*finalTex, output, off);
            else ok = false;
        } while (false);

        destroyImg(src);
        destroyImg(tmp);
        destroyImg(up);
        destroyImg(fin);
        return ok;
    }

    // ---------------- neural graph ----------------
    bool loadModel(const std::string& modelDir) override {
        if (!ok_) return false;
        auto m = Model::load(modelDir);
        if (!m) return false;
        model_ = std::move(m);
        for (auto& kv : tensorBufs_) destroyBuf(kv.second);
        tensorBufs_.clear();
        // upload every tensor as a half-per-word SSBO (host decodes E4M3 -> f16
        // once at load; the words carry the f16 bit patterns)
        for (const auto& t : model_->tensors()) {
            Buf b;
            if (!uploadTensor(t, b)) {
                log_error("vulkan: tensor upload failed for %s", t.name.c_str());
                destroyBuf(b);
                for (auto& kv : tensorBufs_) destroyBuf(kv.second);
                tensorBufs_.clear();
                model_.reset();
                return false;
            }
            tensorBufs_[t.name] = b;
        }
        return true;
    }

    bool hasNeuralGraph() const override { return model_ != nullptr && ok_; }
    const Model* model() const override { return model_.get(); }

    bool runNeuralGraph(const Geometry& geom, const uint16_t* features,
                        float* head, uint64_t frameSeed) override {
        (void)frameSeed;
        if (!ok_ || !model_) return false;
        const auto& sched = model_->schedule();
        if (sched.empty()) { log_error("vulkan: empty block schedule"); return false; }
        if (!model_->tensor("input_proj") || !model_->tensor("head.w")) {
            log_error("vulkan: model missing input_proj / head.w");
            return false;
        }
        const uint32_t FW = geom.fieldWidth, FH = geom.fieldHeight;
        const uint32_t nLevels = model_->config().levels;
        if (FW == 0 || FH == 0) return false;

        // ---- scratch sizing (words = uint32 slots; one half per slot) ----
        size_t maxWords = size_t(FW) * FH * 32;          // Metal's baseline
        size_t maxQkvWords = maxWords;
        for (const auto& se : sched) {
            const uint32_t lw = se.onField ? FW : geom.levels[std::min(se.level, nLevels - 1)].width;
            const uint32_t lh = se.onField ? FH : geom.levels[std::min(se.level, nLevels - 1)].height;
            const uint32_t tokens = lw * lh;
            const uint32_t padded = (tokens + 63u) & ~63u;    // ViT qkv padding
            maxWords = std::max(maxWords, size_t(tokens) * se.channels);
            maxWords = std::max(maxWords, size_t(padded) * se.channels);
            maxQkvWords = std::max(maxQkvWords, size_t(padded) * se.channels * 3);
        }

        Buf& feat    = featBuf_;
        Buf& stateA  = stateA_;
        Buf& stateB  = stateB_;
        Buf& ffnOut  = ffnOut_;
        Buf& projIn  = projIn_;
        Buf& qkv     = qkvBuf_;
        Buf& attnOut = attnOut_;
        Buf& projOut = projOut_;
        Buf& headBuf = headBuf_;
        for (auto* b : { &feat, &stateA, &stateB, &ffnOut, &projIn, &qkv, &attnOut, &projOut })
            if (!ensureBuf(*b, maxWords * 4, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
        if (!ensureBuf(qkv, maxQkvWords * 4, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
        if (!ensureBuf(headBuf, size_t(FW) * FH * 4 * 4, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
        for (auto& s : scratch_) s.used = false;
        // fresh regions read as zero until written (head tail safety)
        std::memset(headBuf.map, 0, size_t(FW) * FH * 4 * 4);

        // ---- features upload (u16 lanes -> one-per-word) ----
        const size_t featWords = size_t(FW) * FH * 16;
        if (!ensureBuf(feat, featWords * 4, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
        {
            uint32_t* w = static_cast<uint32_t*>(feat.map);
            for (size_t i = 0; i < featWords; ++i) w[i] = features[i];
        }

        Buf state = stateA;              // live token state ([tokens][C] halfs)
        uint32_t liveW = FW, liveH = FH, liveC = 0;
        Buf block0Buf;                   // field-level skip (aliases stateA/B)
        bool hasBlock0 = false;
        Buf skipBufs[9];                 // [0..7] levels, [8] field
        bool hasSkip[9] = {};

        if (!beginFrame()) { log_error("vulkan: begin command buffer failed"); return false; }

        uint32_t pc[kPushWords] = {};
        for (size_t bi = 0; bi < sched.size(); ++bi) {
            const BlockScheduleEntry e = sched[bi];
            const uint32_t bw = e.onField ? FW : geom.levels[std::min(e.level, nLevels - 1)].width;
            const uint32_t bh = e.onField ? FH : geom.levels[std::min(e.level, nLevels - 1)].height;
            const uint32_t tokens = bw * bh;
            const uint32_t C = e.channels;
            if (tokens == 0 || C == 0) { abortFrame(); return false; }

            if (bi == 0) {
                liveC = C;
                if (!runInputEmbed(feat, state, FW, FH, C)) { abortFrame(); return false; }
            } else {
                const BlockScheduleEntry p = sched[bi - 1];
                const int32_t effE = e.onField ? -1 : int32_t(e.level);
                const int32_t effP = p.onField ? -1 : int32_t(p.level);
                Buf& other = (state.buf == stateA.buf) ? stateB : stateA;
                if (effE > effP) {
                    // encoder transition: 2x2 box pool -> C -> 2C up-GEMM
                    const std::string lvl = p.onField ? "trans_f" : "trans" + std::to_string(p.level);
                    Buf& pooled = takeScratch(size_t(bw) * bh * p.channels * 4);
                    if (!runPool(state, liveW, liveH, bw, bh, p.channels, pooled) ||
                        !runGemm(pooled, lvl + ".up", bw * bh, C, p.channels, 0, other)) {
                        abortFrame(); return false;
                    }
                    state = other;
                    liveC = C;
                } else if (effE < effP) {
                    // decoder transition: down-GEMM -> 2x upsample -> skip FMA
                    const std::string lvl = p.onField ? "trans_f" : "trans" + std::to_string(p.level);
                    Buf& proj = takeScratch(size_t(liveW) * liveH * C * 4);
                    Buf& up   = takeScratch(size_t(bw) * bh * C * 4);
                    if (!runGemm(state, lvl + ".down", liveW * liveH, C, p.channels, 1, proj) ||
                        !runUpsample(proj, liveW, liveH, bw, bh, C, up)) {
                        abortFrame(); return false;
                    }
                    const bool hasSk = e.onField ? hasBlock0 : hasSkip[std::min(e.level, 8u)];
                    const Buf sk = hasSk ? (e.onField ? block0Buf : skipBufs[std::min(e.level, 8u)]) : Buf{};
                    const Buf scaleT   = tensorAsBuf(lvl + ".scale");
                    const Buf inScaleT = tensorAsBuf(lvl + ".in_scale");
                    if (!runDecoderSkip(up, sk, hasSk, scaleT, !scaleT.tensorName.empty(),
                                        inScaleT, !inScaleT.tensorName.empty(),
                                        bw * bh, C, other)) {
                        abortFrame(); return false;
                    }
                    state = other;
                    liveC = C;
                } else if (p.channels * 2 == C) {
                    // ViT entry: channel expansion GEMM (E4M3 published)
                    if (!runGemm(state, "vitin.up", tokens, C, p.channels, 0, other)) {
                        abortFrame(); return false;
                    }
                    state = other;
                    liveC = C;
                } else if (p.channels == C * 2) {
                    // ViT exit: channel contraction GEMM (raw f16 published)
                    if (!runGemm(state, "vitout.down", tokens, C, p.channels, 1, other)) {
                        abortFrame(); return false;
                    }
                    state = other;
                    liveC = C;
                }
            }
            liveW = bw; liveH = bh;

            if (!runBlock(e, state, ffnOut, projIn, qkv, attnOut, projOut)) {
                abortFrame();
                return false;
            }
            if (bi == 0) { block0Buf = state; hasBlock0 = true; }
            // save encoder skips (field-level skip lives in slot 8)
            if (bi + 1 < sched.size()) {
                const BlockScheduleEntry nx = sched[bi + 1];
                const int32_t effE2 = e.onField ? -1 : int32_t(e.level);
                const int32_t effN = nx.onField ? -1 : int32_t(nx.level);
                if (e.onField) { skipBufs[8] = state; hasSkip[8] = true; }
                else if (effN > effE2 && e.level < 8) { skipBufs[e.level] = state; hasSkip[e.level] = true; }
            }
        }

        // ---- head: C -> 4 f32 per token ----
        {
            const uint32_t tokens = liveW * liveH;
            std::memset(pc, 0, sizeof(pc));
            pcU(pc, 0, tokens);
            pcU(pc, 1, liveC);
            const Buf bias = tensorAsBuf("head.b");
            const std::vector<BindArg> args = {
                {0, &state, nullptr},
                {1, tensorPtr("head.w"), nullptr},
                {2, bias.valid() ? &bias : nullptr, nullptr},
                {5, &headBuf, nullptr},
            };
            if (!dispatchKernel("odl_head", pc, args, divUp(tokens, 64), 1, 1)) {
                abortFrame();
                return false;
            }
        }

        if (!endFrame()) { log_error("vulkan: graph submit failed"); return false; }
        std::memcpy(head, headBuf.map, size_t(FW) * FH * 4 * sizeof(float));
        return true;
    }

    // ---------------- video / game temporal ops ----------------
    bool estimateMotion(const Image& prevLuma, const Image& currLuma, MotionField& mv) override {
        if (!ok_) return false;
        if (prevLuma.empty() || currLuma.empty()) return false;
        mv.width = currLuma.width;
        mv.height = currLuma.height;
        mv.xy.assign(size_t(mv.width) * mv.height * 2, 0.f);

        Img prev, curr, mvTex;
        bool ok = false;
        do {
            if (!allocImage(prevLuma.width, prevLuma.height, prev)) break;
            if (!allocImage(currLuma.width, currLuma.height, curr)) break;
            if (!allocImage(currLuma.width, currLuma.height, mvTex)) break;
            if (!beginFrame()) break;
            static const uint32_t pc[kPushWords] = {};
            ok = recordImageUpload(prevLuma, prev) &&
                 recordImageUpload(currLuma, curr) &&
                 dispatchKernel("odl_motion", pc,
                                {{0, nullptr, &prev}, {1, nullptr, &curr}, {2, nullptr, &mvTex}},
                                divUp(mv.width, 8), divUp(mv.height, 8), 1);
            size_t off = 0;
            if (ok) ok = recordImageDownload(mvTex, off);
            if (endFrame() && ok) {
                Image tmp(mv.width, mv.height);
                if (finishImageDownload(mvTex, tmp, off)) {
                    for (size_t i = 0; i < size_t(mv.width) * mv.height; ++i) {
                        mv.xy[i * 2 + 0] = tmp.pixels[i * 4 + 0];
                        mv.xy[i * 2 + 1] = tmp.pixels[i * 4 + 1];
                    }
                } else ok = false;
            } else ok = false;
        } while (false);
        destroyImg(prev);
        destroyImg(curr);
        destroyImg(mvTex);
        return ok;
    }

    bool reprojectHistory(const Image& historyColor, const MotionField& mv,
                          Image& reproj, Image& confidence) override {
        if (!ok_ || mv.width == 0 || mv.height == 0) return false;
        const bool hasHistory = !historyColor.empty();
        reproj = Image(mv.width, mv.height);
        confidence = Image(mv.width, mv.height);

        // motion field -> rgba32f (dx, dy, 0, 0)
        Image mvImg(mv.width, mv.height);
        for (size_t i = 0; i < size_t(mv.width) * mv.height; ++i) {
            mvImg.pixels[i * 4 + 0] = mv.xy[i * 2 + 0];
            mvImg.pixels[i * 4 + 1] = mv.xy[i * 2 + 1];
            mvImg.pixels[i * 4 + 2] = 0.f;
            mvImg.pixels[i * 4 + 3] = 1.f;
        }

        Img hist, mvTex, rp, cf;
        bool ok = false;
        do {
            if (hasHistory && !allocImage(historyColor.width, historyColor.height, hist)) break;
            if (!allocImage(mv.width, mv.height, mvTex)) break;
            if (!allocImage(mv.width, mv.height, rp)) break;
            if (!allocImage(mv.width, mv.height, cf)) break;
            if (!beginFrame()) break;
            uint32_t pc[kPushWords] = {};
            pcU(pc, 0, hasHistory ? 1u : 0u);
            const Img* histPtr = hasHistory ? &hist : &dummyImg_;   // 1x1 dummy when absent
            ok = recordImageUpload(mvImg, mvTex) &&
                 (hasHistory ? recordImageUpload(historyColor, hist) : true) &&
                 dispatchKernel("odl_reproject", pc,
                                {{0, nullptr, histPtr}, {1, nullptr, &mvTex},
                                 {2, nullptr, &rp}, {3, nullptr, &cf}},
                                divUp(mv.width, 16), divUp(mv.height, 16), 1);
            size_t offRp = 0, offCf = 0;
            if (ok) ok = recordImageDownload(rp, offRp) && recordImageDownload(cf, offCf);
            if (endFrame() && ok) {
                ok = finishImageDownload(rp, reproj, offRp) &&
                     finishImageDownload(cf, confidence, offCf);
            } else ok = false;
        } while (false);
        destroyImg(hist);
        destroyImg(mvTex);
        destroyImg(rp);
        destroyImg(cf);
        return ok;
    }

    bool temporalBlend(const Image& current, const Image& reproj,
                       const Image& confidence, float maxBlend, Image& out) override {
        if (!ok_) return false;
        if (current.empty()) return false;
        out = Image(current.width, current.height);

        Img cur, rp, cf, dst;
        bool ok = false;
        do {
            if (!allocImage(current.width, current.height, cur)) break;
            if (!allocImage(reproj.width, reproj.height, rp)) break;
            if (!allocImage(confidence.width, confidence.height, cf)) break;
            if (!allocImage(current.width, current.height, dst)) break;
            if (!beginFrame()) break;
            uint32_t pc[kPushWords] = {};
            pcF(pc, 0, maxBlend);
            ok = recordImageUpload(current, cur) &&
                 recordImageUpload(reproj, rp) &&
                 recordImageUpload(confidence, cf) &&
                 dispatchKernel("odl_temporal_blend", pc,
                                {{0, nullptr, &cur}, {1, nullptr, &rp},
                                 {2, nullptr, &cf}, {3, nullptr, &dst}},
                                divUp(current.width, 16), divUp(current.height, 16), 1);
            size_t off = 0;
            if (ok) ok = recordImageDownload(dst, off);
            if (endFrame() && ok) ok = finishImageDownload(dst, out, off);
            else ok = false;
        } while (false);
        destroyImg(cur);
        destroyImg(rp);
        destroyImg(cf);
        destroyImg(dst);
        return ok;
    }

    // ---------------- game mode (Lanczos + temporal-blend fallback) ----------------
    bool beginGame(uint32_t renderW, uint32_t renderH, uint32_t outputW, uint32_t outputH) override {
        endGame();
        gRenderW_ = renderW; gRenderH_ = renderH;
        gOutW_ = outputW;    gOutH_ = outputH;
        if (!ok_) return true;      // mirror Metal: surface success, frames will fail
        if (renderW == 0 || renderH == 0 || outputW == 0 || outputH == 0) return false;
        if (!allocImage(renderW, renderH, gColor_)) { endGame(); return false; }
        if (!allocImage(renderW, renderH, gHist_))  { endGame(); return false; }
        if (!allocImage(renderW, renderH, gMv_))    { endGame(); return false; }
        if (!allocImage(renderW, renderH, gRp_))    { endGame(); return false; }
        if (!allocImage(renderW, renderH, gCf_))    { endGame(); return false; }
        if (!allocImage(renderW, renderH, gBlend_)) { endGame(); return false; }
        if (!allocImage(outputW, renderH, gTmp_))   { endGame(); return false; }
        if (!allocImage(outputW, outputH, gOut_))   { endGame(); return false; }
        gHasOut_ = false;
        return true;
    }

    bool submitGameFrame(const GameFrameInput& in, GameFrameOutput& out) override {
        if (!ok_ || !in.color || !gOut_.valid() || !gColor_.valid()) return false;
        if (in.color->width != gRenderW_ || in.color->height != gRenderH_) return false;
        const auto t0 = std::chrono::steady_clock::now();

        bool temporal = false;
        if (!in.resetHistory && gHasOut_ && in.motion &&
            in.motion->width == gRenderW_ && in.motion->height == gRenderH_ &&
            !prevOut_.empty() && gOutW_ > 0 && gRenderW_ > 0 && gOutW_ % gRenderW_ == 0) {
            const uint32_t factor = gOutW_ / gRenderW_;
            Image histLow = (factor > 1) ? *image_downscale_box(prevOut_, factor) : prevOut_;
            temporal = (histLow.width == gRenderW_ && histLow.height == gRenderH_);
            if (temporal) temporalHist_ = std::move(histLow);
        }

        bool ok = false;
        if (beginFrame()) {
            uint32_t pc[kPushWords] = {};
            ok = recordImageUpload(*in.color, gColor_);
            const Img* srcTex = &gColor_;
            if (ok && temporal) {
                Image mvImg(gRenderW_, gRenderH_);
                for (size_t i = 0; i < size_t(gRenderW_) * gRenderH_; ++i) {
                    mvImg.pixels[i * 4 + 0] = in.motion->xy[i * 2 + 0];
                    mvImg.pixels[i * 4 + 1] = in.motion->xy[i * 2 + 1];
                    mvImg.pixels[i * 4 + 2] = 0.f;
                    mvImg.pixels[i * 4 + 3] = 1.f;
                }
                pcU(pc, 0, 1);   // hasHistory
                ok = recordImageUpload(temporalHist_, gHist_) &&
                     recordImageUpload(mvImg, gMv_) &&
                     dispatchKernel("odl_reproject", pc,
                                    {{0, nullptr, &gHist_}, {1, nullptr, &gMv_},
                                     {2, nullptr, &gRp_}, {3, nullptr, &gCf_}},
                                    divUp(gRenderW_, 16), divUp(gRenderH_, 16), 1);
                if (ok) {
                    pcF(pc, 0, kGameMaxBlend);
                    ok = dispatchKernel("odl_temporal_blend", pc,
                                        {{0, nullptr, &gColor_}, {1, nullptr, &gRp_},
                                         {2, nullptr, &gCf_}, {3, nullptr, &gBlend_}},
                                        divUp(gRenderW_, 16), divUp(gRenderH_, 16), 1);
                }
                srcTex = &gBlend_;
            }
            if (ok) {   // Lanczos3 to output resolution
                pcU(pc, 0, gOutW_);
                ok = dispatchKernel("odl_upscale_h", pc, {{0, nullptr, srcTex}, {1, nullptr, &gTmp_}},
                                    divUp(gOutW_, 16), divUp(gRenderH_, 16), 1);
            }
            if (ok) {
                pcU(pc, 0, gOutH_);
                ok = dispatchKernel("odl_upscale_v", pc, {{0, nullptr, &gTmp_}, {1, nullptr, &gOut_}},
                                    divUp(gOutW_, 16), divUp(gOutH_, 16), 1);
            }
            size_t off = 0;
            if (ok) ok = recordImageDownload(gOut_, off);
            if (endFrame() && ok) ok = finishImageDownload(gOut_, out.upscaled, off);
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
        destroyImg(gColor_);
        destroyImg(gHist_);
        destroyImg(gMv_);
        destroyImg(gRp_);
        destroyImg(gCf_);
        destroyImg(gBlend_);
        destroyImg(gTmp_);
        destroyImg(gOut_);
        gRenderW_ = gRenderH_ = gOutW_ = gOutH_ = 0;
        gHasOut_ = false;
        prevOut_ = Image();
        temporalHist_ = Image();
    }

private:
    // =====================================================================
    // device resources
    // =====================================================================
    bool createCommandResources() {
        VkCommandPoolCreateInfo pci{};
        pci.sType = vk::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = vk::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vk_.createCommandPool(vk_.device, &pci, nullptr, &cmdPool_) != vk::VK_SUCCESS)
            return false;

        VkCommandBufferAllocateInfo cbi{};
        cbi.sType = vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = cmdPool_;
        cbi.level = vk::VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        if (vk_.allocateCommandBuffers(vk_.device, &cbi, &cmd_) != vk::VK_SUCCESS)
            return false;

        VkFenceCreateInfo fci{};
        fci.sType = vk::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vk_.createFence(vk_.device, &fci, nullptr, &fence_) != vk::VK_SUCCESS)
            return false;
        return true;
    }

    bool createDescriptorPool() {
        vk::VkDescriptorPoolSize sizes[2] = {};
        sizes[0].type = vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[0].descriptorCount = 2048;
        sizes[1].type = vk::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = 1024;
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = vk::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 512;
        pci.poolSizeCount = 2;
        pci.pPoolSizes = sizes;
        return vk_.createDescriptorPool(vk_.device, &pci, nullptr, &descPool_) == vk::VK_SUCCESS;
    }

    // =====================================================================
    // buffers
    // =====================================================================
    bool allocBuffer(size_t bytes, vk::VkFlags usage, Buf& out) {
        out = Buf{};
        out.bytes = bytes;
        vk::VkBufferCreateInfo ci{};
        ci.sType = vk::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = bytes;
        ci.usage = usage;
        ci.sharingMode = vk::VK_SHARING_MODE_EXCLUSIVE;
        if (vk_.createBuffer(vk_.device, &ci, nullptr, &out.buf) != vk::VK_SUCCESS)
            return false;
        vk::VkMemoryRequirements req{};
        vk_.getBufferMemoryRequirements(vk_.device, out.buf, &req);
        uint32_t type = vk_.findMemoryType(req.memoryTypeBits,
            vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type != 0xFFFFFFFFu) {
            out.coherent = true;
        } else {
            type = vk_.findMemoryType(req.memoryTypeBits, vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            out.coherent = false;
        }
        if (type == 0xFFFFFFFFu || !allocAndBindMemory(req, type, out.mem)) {
            vk_.destroyBuffer(vk_.device, out.buf, nullptr);
            out.buf = VK_NULL_HANDLE;
            return false;
        }
        if (vk_.bindBufferMemory(vk_.device, out.buf, out.mem, 0) != vk::VK_SUCCESS) {
            freeBufferMemory(out);
            return false;
        }
        if (vk_.mapMemory(vk_.device, out.mem, 0, VK_WHOLE_SIZE, 0, &out.map) != vk::VK_SUCCESS) {
            freeBufferMemory(out);
            return false;
        }
        return true;
    }

    bool allocAndBindMemory(const vk::VkMemoryRequirements& req, uint32_t type,
                            vk::VkDeviceMemory& mem) {
        vk::VkMemoryAllocateInfo ai{};
        ai.sType = vk::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        return vk_.allocateMemory(vk_.device, &ai, nullptr, &mem) == vk::VK_SUCCESS;
    }

    void freeBufferMemory(Buf& b) {
        if (b.mem && vk_.unmapMemory && b.map) vk_.unmapMemory(vk_.device, b.mem);
        if (b.buf) vk_.destroyBuffer(vk_.device, b.buf, nullptr);
        if (b.mem) vk_.freeMemory(vk_.device, b.mem, nullptr);
        b = Buf{};
    }

    void destroyBuf(Buf& b) {
        if (!b.valid() && !b.mem) return;
        if (!ok_ || !vk_.device) { b = Buf{}; return; }
        if (b.map) vk_.unmapMemory(vk_.device, b.mem);
        if (b.buf) vk_.destroyBuffer(vk_.device, b.buf, nullptr);
        if (b.mem) vk_.freeMemory(vk_.device, b.mem, nullptr);
        b = Buf{};
    }

    // grow-on-demand; keeps existing contents when already large enough
    bool ensureBuf(Buf& b, size_t bytes, vk::VkFlags usage) {
        if (b.valid() && b.bytes >= bytes) return true;
        destroyBuf(b);
        return allocBuffer(bytes, usage, b);
    }

    void flushIfNonCoherent(const Buf& b) {
        if (b.coherent || !b.mem) return;
        vk::VkMappedMemoryRange r{};
        r.sType = vk::VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        r.memory = b.mem;
        r.offset = 0;
        r.size = VK_WHOLE_SIZE;
        vk_.flushMappedMemoryRanges(vk_.device, 1, &r);
    }

    // scratch ring for transition temporaries (max 3 live at once)
    Buf& takeScratch(size_t bytes) {
        for (auto& s : scratch_) {
            if (s.used) continue;
            if (ensureBuf(s.buf, bytes, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
                s.used = true;
                return s.buf;
            }
            s.used = true;   // failed allocation: consume the slot, op will fail
            return s.buf;
        }
        static Buf overflow;   // unreachable in practice; ops fail before use
        overflow = Buf{};
        return overflow;
    }

    // =====================================================================
    // images (rgba32f storage, optimal tiling, staging-buffer copies)
    // =====================================================================
    bool allocImage(uint32_t w, uint32_t h, Img& out) {
        out = Img{};
        out.w = w;
        out.h = h;
        vk::VkImageCreateInfo ci{};
        ci.sType = vk::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = vk::VK_IMAGE_TYPE_2D;
        ci.format = vk::VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.extent = { w, h, 1 };
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = vk::VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = vk::VK_IMAGE_TILING_OPTIMAL;
        ci.usage = vk::VK_IMAGE_USAGE_STORAGE_BIT |
                   vk::VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   vk::VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = vk::VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = vk::VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk_.createImage(vk_.device, &ci, nullptr, &out.image) != vk::VK_SUCCESS)
            return false;
        vk::VkMemoryRequirements req{};
        vk_.getImageMemoryRequirements(vk_.device, out.image, &req);
        uint32_t type = vk_.findMemoryType(req.memoryTypeBits, vk::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == 0xFFFFFFFFu) type = vk_.findMemoryType(req.memoryTypeBits, 0);
        if (type == 0xFFFFFFFFu || !allocAndBindMemory(req, type, out.mem)) {
            vk_.destroyImage(vk_.device, out.image, nullptr);
            out.image = VK_NULL_HANDLE;
            return false;
        }
        if (vk_.bindImageMemory(vk_.device, out.image, out.mem, 0) != vk::VK_SUCCESS) {
            destroyImg(out);
            return false;
        }
        vk::VkImageViewCreateInfo vci{};
        vci.sType = vk::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = out.image;
        vci.viewType = vk::VK_IMAGE_VIEW_TYPE_2D;
        vci.format = vk::VK_FORMAT_R32G32B32A32_SFLOAT;
        vci.components = { vk::VK_COMPONENT_SWIZZLE_IDENTITY, vk::VK_COMPONENT_SWIZZLE_IDENTITY,
                           vk::VK_COMPONENT_SWIZZLE_IDENTITY, vk::VK_COMPONENT_SWIZZLE_IDENTITY };
        vci.subresourceRange = { vk::VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vk_.createImageView(vk_.device, &vci, nullptr, &out.view) != vk::VK_SUCCESS) {
            destroyImg(out);
            return false;
        }
        return true;
    }

    void destroyImg(Img& im) {
        if (!im.image && !im.mem && !im.view) return;
        if (ok_ && vk_.device) {
            if (im.view) vk_.destroyImageView(vk_.device, im.view, nullptr);
            if (im.image) vk_.destroyImage(vk_.device, im.image, nullptr);
            if (im.mem) vk_.freeMemory(vk_.device, im.mem, nullptr);
        }
        im = Img{};
    }

    // layout transitions around the GENERAL-layout storage ops
    void imageBarrier(const Img& im, vk::VkFlags srcAcc, vk::VkFlags dstAcc,
                      int32_t oldLayout, int32_t newLayout,
                      vk::VkFlags srcStage, vk::VkFlags dstStage) {
        vk::VkImageMemoryBarrier b{};
        b.sType = vk::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = srcAcc;
        b.dstAccessMask = dstAcc;
        b.oldLayout = oldLayout;
        b.newLayout = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im.image;
        b.subresourceRange = { vk::VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_.cmdPipelineBarrier(cmd_, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // full read-after-write / write-after-write dependency between dispatches
    void barrierComputeRW() {
        vk::VkMemoryBarrier mb{};
        mb.sType = vk::VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = vk::VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_SHADER_WRITE_BIT;
        vk_.cmdPipelineBarrier(cmd_, vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               1, &mb, 0, nullptr, 0, nullptr);
    }

    bool recordImageUpload(const Image& img, Img& dst) {
        const size_t bytes = img.pixels.size() * sizeof(float);
        if (!bytes || !dst.valid()) return false;
        if (!ensureBuf(upStage_, bytes, vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
        std::memcpy(upStage_.map, img.pixels.data(), bytes);
        flushIfNonCoherent(upStage_);
        imageBarrier(dst, 0, vk::VK_ACCESS_TRANSFER_WRITE_BIT,
                     vk::VK_IMAGE_LAYOUT_UNDEFINED, vk::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     vk::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk::VK_PIPELINE_STAGE_TRANSFER_BIT);
        vk::VkBufferImageCopy c{};
        c.bufferRowLength = 0;
        c.bufferImageHeight = 0;
        c.imageSubresource = { vk::VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        c.imageExtent = { dst.w, dst.h, 1 };
        vk_.cmdCopyBufferToImage(cmd_, upStage_.buf, dst.image,
                                 vk::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        imageBarrier(dst, vk::VK_ACCESS_TRANSFER_WRITE_BIT,
                     vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_SHADER_WRITE_BIT,
                     vk::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, vk::VK_IMAGE_LAYOUT_GENERAL,
                     vk::VK_PIPELINE_STAGE_TRANSFER_BIT, vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        return true;
    }

    bool recordImageDownload(const Img& src, size_t& offsetOut) {
        const size_t bytes = size_t(src.w) * src.h * 4 * sizeof(float);
        if (!bytes || !src.valid()) return false;
        offsetOut = downStageBytes_;
        if (!ensureBuf(downStage_, offsetOut + bytes, vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
        downStageBytes_ = offsetOut + bytes;
        imageBarrier(src, vk::VK_ACCESS_SHADER_WRITE_BIT, vk::VK_ACCESS_TRANSFER_READ_BIT,
                     vk::VK_IMAGE_LAYOUT_GENERAL, vk::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk::VK_PIPELINE_STAGE_TRANSFER_BIT);
        vk::VkBufferImageCopy c{};
        c.bufferOffset = offsetOut;
        c.bufferRowLength = 0;
        c.bufferImageHeight = 0;
        c.imageSubresource = { vk::VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        c.imageExtent = { src.w, src.h, 1 };
        vk_.cmdCopyImageToBuffer(cmd_, src.image, vk::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 downStage_.buf, 1, &c);
        imageBarrier(src, vk::VK_ACCESS_TRANSFER_READ_BIT,
                     vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_SHADER_WRITE_BIT,
                     vk::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk::VK_IMAGE_LAYOUT_GENERAL,
                     vk::VK_PIPELINE_STAGE_TRANSFER_BIT, vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        return true;
    }

    // copies the recorded downloads out of the staging buffer (after endFrame)
    bool finishImageDownload(const Img& src, Image& out, size_t offset) {
        const size_t bytes = size_t(src.w) * src.h * 4 * sizeof(float);
        if (out.width != src.w || out.height != src.h) out = Image(src.w, src.h);
        if (!downStage_.valid() || !downStage_.map || downStage_.bytes < offset + bytes) return false;
        if (!downStage_.coherent) {
            vk::VkMappedMemoryRange r{};
            r.sType = vk::VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            r.memory = downStage_.mem;
            r.offset = 0;
            r.size = VK_WHOLE_SIZE;
            vk_.invalidateMappedMemoryRanges(vk_.device, 1, &r);
        }
        std::memcpy(out.pixels.data(), static_cast<const uint8_t*>(downStage_.map) + offset, bytes);
        return true;
    }

    // =====================================================================
    // kernels
    // =====================================================================
    struct Kernel {
        vk::VkPipeline pipe = VK_NULL_HANDLE;
        vk::VkPipelineLayout layout = VK_NULL_HANDLE;
        vk::VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        bool valid() const { return pipe != VK_NULL_HANDLE; }
    };

    const Buf* tensorPtr(const std::string& name) const {
        auto it = tensorBufs_.find(name);
        return it != tensorBufs_.end() ? &it->second : nullptr;
    }

    // tensor handle carrying its name so callers can ask hasScale-style flags
    struct NamedBuf : Buf {
        std::string tensorName;
    };

    NamedBuf tensorAsBuf(const std::string& name) {
        NamedBuf nb;
        const Buf* b = tensorPtr(name);
        if (b) { *static_cast<Buf*>(&nb) = *b; nb.tensorName = name; }
        return nb;
    }

    bool uploadTensor(const TensorRecord& t, Buf& out) {
        const size_t words = t.f16.size();
        if (words == 0) return false;
        if (!ensureBuf(out, words * 4, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
        uint32_t* w = static_cast<uint32_t*>(out.map);
        for (size_t i = 0; i < words; ++i) w[i] = t.f16[i];   // one half per word
        return true;
    }

    bool getKernel(const std::string& name, Kernel& out) {
        auto it = kernels_.find(name);
        if (it != kernels_.end()) { out = it->second; return out.valid(); }
        const KernelSpec* spec = findKernel(name);
        if (!spec) { log_error("vulkan: unknown kernel %s", name.c_str()); return false; }

        std::vector<uint32_t> code;
        if (!readSpirv(shaderDir_ + "/" + name + ".spv", code)) {
            log_error("vulkan: cannot load %s.spv (searched %s; set OPENDLSS_VK_SHADER_DIR)",
                      name.c_str(), shaderDir_.c_str());
            return false;
        }
        vk::VkShaderModuleCreateInfo mci{};
        mci.sType = vk::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mci.codeSize = code.size() * 4;
        mci.pCode = code.data();
        vk::VkShaderModule mod = VK_NULL_HANDLE;
        if (vk_.createShaderModule(vk_.device, &mci, nullptr, &mod) != vk::VK_SUCCESS) {
            log_error("vulkan: vkCreateShaderModule failed for %s", name.c_str());
            return false;
        }

        Kernel k;
        std::vector<vk::VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(spec->nb);
        for (uint32_t i = 0; i < spec->nb; ++i) {
            vk::VkDescriptorSetLayoutBinding b{};
            b.binding = spec->bd[i].b;
            b.descriptorType = spec->bd[i].img ? vk::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                               : vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags = vk::VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(b);
        }
        vk::VkDescriptorSetLayoutCreateInfo slci{};
        slci.sType = vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slci.bindingCount = uint32_t(bindings.size());
        slci.pBindings = bindings.data();
        if (vk_.createDescriptorSetLayout(vk_.device, &slci, nullptr, &k.setLayout) != vk::VK_SUCCESS) {
            vk_.destroyShaderModule(vk_.device, mod, nullptr);
            log_error("vulkan: set layout failed for %s", name.c_str());
            return false;
        }
        vk::VkPushConstantRange pcr{};
        pcr.stageFlags = vk::VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = kPushBytes;
        vk::VkPipelineLayoutCreateInfo plci{};
        plci.sType = vk::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &k.setLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (vk_.createPipelineLayout(vk_.device, &plci, nullptr, &k.layout) != vk::VK_SUCCESS) {
            vk_.destroyDescriptorSetLayout(vk_.device, k.setLayout, nullptr);
            vk_.destroyShaderModule(vk_.device, mod, nullptr);
            log_error("vulkan: pipeline layout failed for %s", name.c_str());
            return false;
        }
        vk::VkPipelineShaderStageCreateInfo stage{};
        stage.sType = vk::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = vk::VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        vk::VkComputePipelineCreateInfo cpi{};
        cpi.sType = vk::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage = stage;
        cpi.layout = k.layout;
        if (vk_.createComputePipelines(vk_.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &k.pipe) != vk::VK_SUCCESS) {
            vk_.destroyPipelineLayout(vk_.device, k.layout, nullptr);
            vk_.destroyDescriptorSetLayout(vk_.device, k.setLayout, nullptr);
            vk_.destroyShaderModule(vk_.device, mod, nullptr);
            log_error("vulkan: pipeline failed for %s", name.c_str());
            return false;
        }
        vk_.destroyShaderModule(vk_.device, mod, nullptr);   // pipeline owns its SPIR-V now
        kernels_[name] = k;
        out = k;
        return true;
    }

    bool dispatchKernel(const std::string& name, const uint32_t pc[kPushWords],
                        const std::vector<BindArg>& args,
                        uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
        Kernel k;
        if (!getKernel(name, k)) return false;
        if (gx == 0 || gy == 0 || gz == 0) return false;
        const KernelSpec* spec = findKernel(name);

        // descriptor set (pool is reset after every submission)
        vk::VkDescriptorSetAllocateInfo dai{};
        dai.sType = vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = descPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &k.setLayout;
        vk::VkDescriptorSet set = VK_NULL_HANDLE;
        if (vk_.allocateDescriptorSets(vk_.device, &dai, &set) != vk::VK_SUCCESS) {
            log_error("vulkan: descriptor set allocation failed (%s)", name.c_str());
            return false;
        }

        // writes: one per layout binding; anything the caller left out gets a
        // 1-element dummy (semantics identical to the kernels' hasX flags)
        vk::VkDescriptorBufferInfo binfo[kMaxKernelBindings];
        vk::VkDescriptorImageInfo iinfo[kMaxKernelBindings];
        vk::VkWriteDescriptorSet writes[kMaxKernelBindings];
        uint32_t nw = 0;
        for (uint32_t i = 0; i < spec->nb; ++i) {
            const uint32_t b = spec->bd[i].b;
            const BindArg* arg = nullptr;
            for (const BindArg& a : args)
                if (a.binding == b) { arg = &a; break; }
            vk::VkWriteDescriptorSet& w = writes[nw];
            w = {};
            w.sType = vk::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = set;
            w.dstBinding = b;
            w.descriptorCount = 1;
            if (spec->bd[i].img) {
                const Img* im = (arg && arg->img) ? arg->img : &dummyImg_;
                iinfo[nw] = { VK_NULL_HANDLE, im->view, vk::VK_IMAGE_LAYOUT_GENERAL };
                w.descriptorType = vk::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w.pImageInfo = &iinfo[nw];
            } else {
                const Buf* bf = (arg && arg->buf && arg->buf->valid()) ? arg->buf : &dummyBuf_;
                binfo[nw] = { bf->buf, 0, VK_WHOLE_SIZE };
                w.descriptorType = vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo = &binfo[nw];
            }
            ++nw;
        }
        vk_.updateDescriptorSets(vk_.device, nw, writes, 0, nullptr);

        vk_.cmdBindPipeline(cmd_, vk::VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
        vk_.cmdBindDescriptorSets(cmd_, vk::VK_PIPELINE_BIND_POINT_COMPUTE, k.layout,
                                  0, 1, &set, 0, nullptr);
        vk_.cmdPushConstants(cmd_, k.layout, vk::VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pc);
        vk_.cmdDispatch(cmd_, gx, gy, gz);
        barrierComputeRW();
        return true;
    }

    // =====================================================================
    // graph steps
    // =====================================================================
    bool runInputEmbed(const Buf& feat, const Buf& state, uint32_t fw, uint32_t fh, uint32_t C) {
        uint32_t pc[kPushWords] = {};
        pcU(pc, 0, fw);
        pcU(pc, 1, fh);
        pcU(pc, 2, C);
        const Buf proj = [this]() {
            const Buf* b = tensorPtr("input_proj");
            return b ? *b : Buf{};
        }();
        const std::vector<BindArg> args = {
            {0, &feat, nullptr},
            {1, proj.valid() ? &proj : nullptr, nullptr},
            {2, &state, nullptr},
        };
        return dispatchKernel("odl_input_embed", pc, args, divUp(fw * fh, 64), 1, 1);
    }

    bool runPool(const Buf& src, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh,
                 uint32_t C, Buf& dst) {
        uint32_t pc[kPushWords] = {};
        pcU(pc, 0, sw);
        pcU(pc, 1, sh);
        pcU(pc, 2, dw);
        pcU(pc, 3, dh);
        pcU(pc, 4, C);
        const std::vector<BindArg> args = { {0, &src, nullptr}, {4, &dst, nullptr} };
        return dispatchKernel("odl_pool2x2", pc, args, divUp(dw * dh, 64), 1, 1);
    }

    bool runGemm(const Buf& x, const std::string& wName, uint32_t tokens,
                 uint32_t outCh, uint32_t inCh, uint32_t pubMode, Buf& out) {
        const Buf* w = tensorPtr(wName);
        if (!w) { log_error("vulkan: missing tensor %s", wName.c_str()); return false; }
        uint32_t pc[kPushWords] = {};
        pcU(pc, 0, tokens);
        pcU(pc, 1, outCh);
        pcU(pc, 2, inCh);
        pcU(pc, 3, pubMode);
        pcU(pc, 4, 0);                       // hasBias = 0 (transitions are biasless)
        const std::vector<BindArg> args = { {0, &x, nullptr}, {1, w, nullptr}, {7, &out, nullptr} };
        return dispatchKernel("odl_channel_gemm", pc, args, divUp(tokens * outCh, 64), 1, 1);
    }

    bool runUpsample(const Buf& src, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh,
                     uint32_t C, Buf& dst) {
        uint32_t pc[kPushWords] = {};
        pcU(pc, 0, sw);
        pcU(pc, 1, sh);
        pcU(pc, 2, dw);
        pcU(pc, 3, dh);
        pcU(pc, 4, C);
        const std::vector<BindArg> args = { {0, &src, nullptr}, {4, &dst, nullptr} };
        return dispatchKernel("odl_nearest_upsample2x", pc, args, divUp(dw * dh * C, 64), 1, 1);
    }

    bool runDecoderSkip(const Buf& up, const Buf& skip, bool hasSkip,
                        const Buf& scale, bool hasScale,
                        const Buf& inScale, bool hasInScale,
                        uint32_t tokens, uint32_t C, Buf& out) {
        uint32_t pc[kPushWords] = {};
        pcU(pc, 0, tokens);
        pcU(pc, 1, C);
        pcU(pc, 2, hasSkip ? 1u : 0u);
        pcU(pc, 3, hasScale ? 1u : 0u);
        pcU(pc, 4, hasInScale ? 1u : 0u);
        const Buf skipVal = hasSkip ? skip : Buf{};
        const Buf scaleVal = hasScale ? static_cast<const Buf&>(scale) : Buf{};
        const Buf inScaleVal = hasInScale ? static_cast<const Buf&>(inScale) : Buf{};
        const std::vector<BindArg> args = {
            {0, &up, nullptr},
            {1, skipVal.valid() ? &skipVal : nullptr, nullptr},
            {2, scaleVal.valid() ? &scaleVal : nullptr, nullptr},
            {3, inScaleVal.valid() ? &inScaleVal : nullptr, nullptr},
            {5, &out, nullptr},
        };
        return dispatchKernel("odl_decoder_skip", pc, args, divUp(tokens * C, 64), 1, 1);
    }

    bool runBlock(const BlockScheduleEntry& e, Buf& state, Buf& ffnOut, Buf& projIn,
                  Buf& qkv, Buf& attnOut, Buf& projOut) {
        const uint32_t tokens = e.levelWidth * e.levelHeight;
        const uint32_t C = e.channels;
        const std::string b = std::to_string(e.index);

        // 1. FFN -> ffnOut (f32, pre-skip)
        uint32_t hidden = 0, paths = 0, perPathOut = 0, inSlice = 0, applySilu = 0, hasW3 = 0;
        if (C == 32 || e.isViT) {                    // dense: C -> 128/4096 -> C
            hidden = e.isViT ? 4096u : 128u;
            paths = 1;
            perPathOut = C;
            applySilu = 1;
            hasW3 = 0;
        } else if (C == 512) {                       // branch: 8 x (64 -> 256 -> 64) + C->C
            hidden = 256;
            paths = 8;
            perPathOut = 64;
            inSlice = 64;
            applySilu = 0;
            hasW3 = 1;
        } else {                                     // wide: C/32 paths, C -> 128 -> 32 + C->C
            hidden = 128;
            paths = C / 32;
            perPathOut = 32;
            applySilu = 1;
            hasW3 = 1;
        }
        {
            const Buf w1 = tensorAsBuf("block" + b + ".layer0.w1");
            const Buf b1 = tensorAsBuf("block" + b + ".layer0.b1");
            const Buf w2 = tensorAsBuf("block" + b + ".layer0.w2");
            const Buf b2 = tensorAsBuf("block" + b + ".layer0.b2");
            const Buf w3 = tensorAsBuf("block" + b + ".layer0.w3");
            const Buf b3 = tensorAsBuf("block" + b + ".layer0.b3");
            if (!w1.valid() || !w2.valid()) {
                log_error("vulkan: block %u missing FFN weights", e.index);
                return false;
            }
            uint32_t pc[kPushWords] = {};
            pcU(pc, 0, tokens);
            pcU(pc, 1, C);
            pcU(pc, 2, hidden);
            pcU(pc, 3, paths);
            pcU(pc, 4, perPathOut);
            pcU(pc, 5, inSlice);
            pcU(pc, 6, applySilu);
            pcU(pc, 7, hasW3);
            const Buf w3v = (hasW3 && w3.valid()) ? w3 : Buf{};
            const Buf b3v = (hasW3 && b3.valid()) ? b3 : Buf{};
            const std::vector<BindArg> args = {
                {0, &state, nullptr},
                {1, &w1, nullptr}, {2, b1.valid() ? &b1 : nullptr, nullptr},
                {3, &w2, nullptr}, {4, b2.valid() ? &b2 : nullptr, nullptr},
                {5, w3v.valid() ? &w3v : nullptr, nullptr},
                {6, b3v.valid() ? &b3v : nullptr, nullptr},
                {11, &ffnOut, nullptr},
            };
            if (!dispatchKernel("odl_ffn", pc, args, tokens, 1, 1)) return false;
        }

        // 2. block residual: y = pub(x * ffnScale + ffnOut) -> projIn
        {
            const Buf s = tensorAsBuf("block" + b + ".layer0.scale_ffn");
            uint32_t pc[kPushWords] = {};
            pcU(pc, 0, tokens);
            pcU(pc, 1, C);
            pcU(pc, 2, (C == 32 && !e.isViT) ? 1u : 0u);   // rawF16 for 32-ch window blocks
            pcU(pc, 3, s.valid() ? 1u : 0u);
            const std::vector<BindArg> args = {
                {0, &state, nullptr},
                {1, &ffnOut, nullptr},
                {2, s.valid() ? &s : nullptr, nullptr},
                {6, &projIn, nullptr},
            };
            if (!dispatchKernel("odl_block_skip", pc, args, divUp(tokens * C, 64), 1, 1)) return false;
        }

        // 3. attention
        const Buf qscale = tensorAsBuf("block" + b + ".layer1.qscale");
        if (e.isViT) {
            // qkv GEMM over 64-padded tokens (padding rows must read as zero),
            // raw f16 publication; the kernel applies the exp(0) correction.
            const uint32_t padded = (tokens + 63u) & ~63u;
            if (padded != tokens)
                std::memset(static_cast<uint8_t*>(projIn.map) + size_t(tokens) * C * 4, 0,
                            size_t(padded - tokens) * C * 4);
            const Buf qkvT = tensorAsBuf("block" + b + ".layer1.qkv");
            if (!qkvT.valid()) { log_error("vulkan: block %u missing qkv", e.index); return false; }
            {
                uint32_t pc[kPushWords] = {};
                pcU(pc, 0, padded);
                pcU(pc, 1, 3 * C);
                pcU(pc, 2, C);
                pcU(pc, 3, 1);                       // pubMode = raw f16
                pcU(pc, 4, 0);
                const std::vector<BindArg> args = { {0, &projIn, nullptr}, {1, &qkvT, nullptr}, {7, &qkv, nullptr} };
                if (!dispatchKernel("odl_channel_gemm", pc, args, divUp(padded * 3 * C, 64), 1, 1))
                    return false;
            }
            uint32_t pc[kPushWords] = {};
            pcU(pc, 0, tokens);
            pcU(pc, 1, C);
            pcU(pc, 2, e.heads);
            const std::vector<BindArg> args = {
                {0, &qkv, nullptr},
                {1, qscale.valid() ? &qscale : nullptr, nullptr},
                {5, &attnOut, nullptr},
            };
            if (!dispatchKernel("odl_global_attention", pc, args, divUp(tokens, 32), 1, 1))
                return false;
        } else {
            // QKV GEMM (E4M3-published output), then window attention
            const Buf qkvT = tensorAsBuf("block" + b + ".layer1.qkv");
            if (!qkvT.valid()) { log_error("vulkan: block %u missing qkv", e.index); return false; }
            {
                uint32_t pc[kPushWords] = {};
                pcU(pc, 0, tokens);
                pcU(pc, 1, 3 * C);
                pcU(pc, 2, C);
                pcU(pc, 3, 0);                       // pubMode = E4M3
                pcU(pc, 4, 0);
                const std::vector<BindArg> args = { {0, &projIn, nullptr}, {1, &qkvT, nullptr}, {7, &qkv, nullptr} };
                if (!dispatchKernel("odl_channel_gemm", pc, args, divUp(tokens * 3 * C, 64), 1, 1))
                    return false;
            }
            const Buf prior = tensorAsBuf("block" + b + ".layer1.prior");
            uint32_t pc[kPushWords] = {};
            pcU(pc, 0, e.levelWidth);
            pcU(pc, 1, e.levelHeight);
            pcU(pc, 2, C);
            pcU(pc, 3, e.heads);
            pcU(pc, 4, e.windowPhase);
            pcU(pc, 5, prior.valid() ? 1u : 0u);
            const std::vector<BindArg> args = {
                {0, &qkv, nullptr},
                {1, prior.valid() ? &prior : nullptr, nullptr},
                {2, qscale.valid() ? &qscale : nullptr, nullptr},
                {7, &attnOut, nullptr},
            };
            const uint32_t shift = (e.windowPhase == 0) ? 0u : 4u;
            const uint32_t gridX = (e.levelWidth + shift + 7u) / 8u;
            const uint32_t gridY = (e.levelHeight + shift + 7u) / 8u;
            if (!dispatchKernel("odl_window_attention", pc, args, gridX * gridY, 1, 1))
                return false;
        }

        // 3.5 projection GEMM over the attention output (raw f16), then
        // 4. epilogue: out = pub(y * attnScale + projOut) -> state
        {
            const Buf projT = tensorAsBuf("block" + b + ".layer1.proj");
            if (!projT.valid()) { log_error("vulkan: block %u missing attention proj", e.index); return false; }
            uint32_t pc[kPushWords] = {};
            pcU(pc, 0, tokens);
            pcU(pc, 1, C);
            pcU(pc, 2, C);
            pcU(pc, 3, 1);                           // pubMode = raw f16 (reference)
            pcU(pc, 4, 0);
            const std::vector<BindArg> args = { {0, &attnOut, nullptr}, {1, &projT, nullptr}, {7, &projOut, nullptr} };
            if (!dispatchKernel("odl_channel_gemm", pc, args, divUp(tokens * C, 64), 1, 1))
                return false;
        }
        {
            const Buf s = tensorAsBuf("block" + b + ".layer0.scale_attn");
            uint32_t pc[kPushWords] = {};
            pcU(pc, 0, tokens);
            pcU(pc, 1, C);
            pcU(pc, 2, s.valid() ? 1u : 0u);
            const std::vector<BindArg> args = {
                {0, &projIn, nullptr},
                {1, &projOut, nullptr},
                {2, s.valid() ? &s : nullptr, nullptr},
                {5, &state, nullptr},
            };
            if (!dispatchKernel("odl_block_epilogue", pc, args, divUp(tokens * C, 64), 1, 1))
                return false;
        }
        return true;
    }

    // =====================================================================
    // frame plumbing
    // =====================================================================
    bool beginFrame() {
        downStageBytes_ = 0;
        vk::VkCommandBufferBeginInfo bi{};
        bi.sType = vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = vk::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vk_.resetCommandBuffer(cmd_, 0) != vk::VK_SUCCESS) return false;
        return vk_.beginCommandBuffer(cmd_, &bi) == vk::VK_SUCCESS;
    }

    bool endFrame() {
        if (vk_.endCommandBuffer(cmd_) != vk::VK_SUCCESS) return false;
        vk::VkSubmitInfo si{};
        si.sType = vk::VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        const VkResult r = vk_.queueSubmit(vk_.queue, 1, &si, fence_);
        if (r != vk::VK_SUCCESS) return false;
        vk_.waitForFences(vk_.device, 1, &fence_, VK_TRUE /*?*/, UINT64_MAX);
        vk_.resetFences(vk_.device, 1, &fence_);
        if (descPool_) vk_.resetDescriptorPool(vk_.device, descPool_, 0);
        return true;
    }

    void abortFrame() {
        if (ok_ && vk_.device) {
            vk_.resetCommandBuffer(cmd_, 0);
            if (descPool_) vk_.resetDescriptorPool(vk_.device, descPool_, 0);
        }
        downStageBytes_ = 0;
    }

    void destroyAll() {
        if (vk_.device) {
            if (vk_.queue) vk_.queueWaitIdle(vk_.queue);
            for (auto& kv : kernels_) {
                vk_.destroyPipeline(vk_.device, kv.second.pipe, nullptr);
                vk_.destroyPipelineLayout(vk_.device, kv.second.layout, nullptr);
                vk_.destroyDescriptorSetLayout(vk_.device, kv.second.setLayout, nullptr);
            }
            kernels_.clear();
            for (auto& kv : tensorBufs_) destroyBuf(kv.second);
            tensorBufs_.clear();
            destroyBuf(dummyBuf_);
            destroyImg(dummyImg_);
            destroyImg(gColor_);
            destroyImg(gHist_);
            destroyImg(gMv_);
            destroyImg(gRp_);
            destroyImg(gCf_);
            destroyImg(gBlend_);
            destroyImg(gTmp_);
            destroyImg(gOut_);
            destroyBuf(featBuf_);
            destroyBuf(stateA_);
            destroyBuf(stateB_);
            destroyBuf(ffnOut_);
            destroyBuf(projIn_);
            destroyBuf(qkvBuf_);
            destroyBuf(attnOut_);
            destroyBuf(projOut_);
            destroyBuf(headBuf_);
            for (auto& s : scratch_) destroyBuf(s.buf);
            destroyBuf(upStage_);
            destroyBuf(downStage_);
            if (descPool_) vk_.destroyDescriptorPool(vk_.device, descPool_, nullptr);
            if (fence_) vk_.destroyFence(vk_.device, fence_, nullptr);
            if (cmdPool_) vk_.destroyCommandPool(vk_.device, cmdPool_, nullptr);
            descPool_ = VK_NULL_HANDLE;
            fence_ = VK_NULL_HANDLE;
            cmdPool_ = VK_NULL_HANDLE;
        }
        vk::bootShutdown(vk_);
        ok_ = false;
    }

    // =====================================================================
    // state
    // =====================================================================
    struct ScratchSlot {
        Buf buf;
        bool used = false;
    };

    Boot vk_{};
    BackendInfo info_{};
    bool ok_ = false;
    std::string shaderDir_;

    vk::VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    vk::VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    vk::VkFence fence_ = VK_NULL_HANDLE;
    vk::VkDescriptorPool descPool_ = VK_NULL_HANDLE;

    std::map<std::string, Kernel> kernels_;
    std::map<std::string, Buf> tensorBufs_;
    std::unique_ptr<Model> model_;

    Buf dummyBuf_;                  // 1-word zero SSBO for absent optional tensors
    Img dummyImg_;                  // 1x1 rgba32f for absent history
    Buf upStage_;                   // host -> image staging
    Buf downStage_;                 // image -> host staging
    size_t downStageBytes_ = 0;

    // neural-graph scratch (grown on demand, reused across frames)
    Buf featBuf_;
    Buf stateA_, stateB_;
    Buf ffnOut_;                    // f32 pre-skip FFN output
    Buf projIn_;                    // y (block residual result)
    Buf qkvBuf_;
    Buf attnOut_;
    Buf projOut_;
    Buf headBuf_;
    ScratchSlot scratch_[3];

    // game mode state
    uint32_t gRenderW_ = 0, gRenderH_ = 0, gOutW_ = 0, gOutH_ = 0;
    Img gColor_, gHist_, gMv_, gRp_, gCf_, gBlend_, gTmp_, gOut_;
    Image prevOut_;
    Image temporalHist_;
    bool gHasOut_ = false;
};

} // namespace opendlss

namespace opendlss {
std::unique_ptr<IBackend> create_vulkan_backend() {
    auto b = std::make_unique<VulkanBackend>();
    if (!b->info().neuralGraph && b->info().kind != BackendKind::Vulkan) return nullptr;
    if (!b->info().deviceName.empty() || b->info().kind == BackendKind::Vulkan) {
        // constructed object is always "vulkan"; a failed boot leaves deviceName
        // empty AND info_.neuralGraph set only on success — check via details
    }
    // A failed constructor never sets info_.kind to Vulkan; catch that case and
    // hand nullptr to the registry so the CPU backend takes over.
    if (b->info().kind != BackendKind::Vulkan) return nullptr;
    return b;
}
} // namespace opendlss
