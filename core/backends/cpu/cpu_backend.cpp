// CPU backend: reference implementation of every backend op.
// Runs on all platforms; used for CI, fallback and as the parity golden that
// GPU kernels are tested against. The neural graph executes the same block
// semantics as the GPU ports: fp16 storage, E4M3 publications, cosine
// attention with the half bit-trick exponential, 4-phase shifted windows.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/backend.h"
#include "opendlss/fp16.h"
#include "opendlss/e4m3.h"
#include "opendlss/pipeline.h"
#include "opendlss/logging.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace opendlss {

// ---------------------------------------------------------------------------
// neural graph on CPU (defined in neural_graph_cpu.cpp)
// ---------------------------------------------------------------------------

struct CpuGraphWorkspace;

bool neural_graph_execute(const Model& model, const Geometry& geom,
                          const uint16_t* features, float* head, uint64_t frameSeed,
                          CpuGraphWorkspace** persistent);

class CpuBackend final : public IBackend {
public:
    const BackendInfo& info() const override { return info_; }

    bool loadModel(const std::string& modelDir) override {
        model_ = Model::load(modelDir);
        if (!model_) return false;
        info_.neuralGraph = true;
        info_.e4m3Quant = true;
        return true;
    }
    bool hasNeuralGraph() const override { return model_ != nullptr; }
    const Model* model() const override { return model_.get(); }

    bool denoiseSpatial(const Image& input, Image& output, float strength) override;
    bool upscaleSpatial(const Image& input, Image& output, uint32_t factor, float sharpen) override;

    bool runNeuralGraph(const Geometry& geom, const uint16_t* features,
                        float* head, uint64_t frameSeed) override {
        if (!model_) return false;
        return neural_graph_execute(*model_, geom, features, head, frameSeed, &ws_);
    }

    bool estimateMotion(const Image& prevLuma, const Image& currLuma, MotionField& mv) override;
    bool reprojectHistory(const Image& historyColor, const MotionField& mv,
                          Image& reproj, Image& confidence) override;
    bool temporalBlend(const Image& current, const Image& reproj,
                       const Image& confidence, float maxBlend, Image& out) override;

    bool beginGame(uint32_t renderW, uint32_t renderH, uint32_t outputW, uint32_t outputH) override;
    bool submitGameFrame(const GameFrameInput& in, GameFrameOutput& out) override;
    void endGame() override;

private:
    BackendInfo info_ = [] {
        BackendInfo bi;
        bi.kind = BackendKind::Cpu;
        bi.name = "cpu";
        bi.deviceName = "host";
        bi.neuralGraph = true;
        bi.fp16Storage = true;   // software half
        bi.e4m3Quant = true;
        bi.details = "reference numerics; all ops on CPU";
        return bi;
    }();
    std::unique_ptr<Model> model_;
    CpuGraphWorkspace* ws_ = nullptr;

    // game-mode state
    uint32_t gRenderW_ = 0, gRenderH_ = 0, gOutW_ = 0, gOutH_ = 0;
    Image gHistory_;
    bool gHasHistory_ = false;
};

// ---------------------------------------------------------------------------
// spatial ops
// ---------------------------------------------------------------------------

bool CpuBackend::denoiseSpatial(const Image& input, Image& output, float strength) {
    if (strength <= 0.0f) { output = input; return true; }
    output = Image(input.width, input.height);
    const float sigmaColor = 0.08f + 0.12f * strength;
    const int radius = strength > 0.75f ? 3 : 2;
    for (uint32_t y = 0; y < input.height; ++y) {
        for (uint32_t x = 0; x < input.width; ++x) {
            const float* c = input.row(y) + x * 4;
            float cl = Image::luma(c);
            float acc[3] = {0, 0, 0}, wsum = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                int sy = std::clamp(int(y) + dy, 0, int(input.height) - 1);
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = std::clamp(int(x) + dx, 0, int(input.width) - 1);
                    const float* n = input.row(uint32_t(sy)) + uint32_t(sx) * 4;
                    float dc = (n[0] - c[0]) * (n[0] - c[0]) + (n[1] - c[1]) * (n[1] - c[1])
                             + (n[2] - c[2]) * (n[2] - c[2]);
                    float dl = Image::luma(n) - cl;
                    float w = std::exp(-dc / (2 * sigmaColor * sigmaColor)
                                       - dl * dl / (2 * sigmaColor * sigmaColor * 0.25f));
                    for (int ch = 0; ch < 3; ++ch) acc[ch] += n[ch] * w;
                    wsum += w;
                }
            }
            float* d = output.row(y) + x * 4;
            for (int ch = 0; ch < 3; ++ch) d[ch] = clampf(acc[ch] / wsum, 0.f, 1.f);
            d[3] = 1.f;
        }
    }
    return true;
}

namespace {
// Lanczos3 resample along one axis (separable).
float lanczos3(float x) {
    if (x == 0) return 1.0f;
    if (std::fabs(x) >= 3.0f) return 0.0f;
    const float pi = 3.14159265358979f;
    return 3.0f * std::sin(pi * x) * std::sin(pi * x / 3.0f) / (pi * pi * x * x);
}

void resample_axis(const float* src, uint32_t srcLen, uint32_t srcStride,
                   float* dst, uint32_t dstLen, uint32_t dstStride,
                   uint32_t channels) {
    const float scale = float(srcLen) / float(dstLen);
    const float support = scale > 1.0f ? 3.0f * scale : 3.0f;
    for (uint32_t d = 0; d < dstLen; ++d) {
        float center = (float(d) + 0.5f) * scale - 0.5f;
        int32_t lo = int32_t(std::floor(center - support + 0.5f));
        int32_t hi = int32_t(std::floor(center + support + 0.5f));
        float acc[4] = {0, 0, 0, 0}, wsum = 0;
        for (int32_t s = lo; s <= hi; ++s) {
            float w = lanczos3((float(s) - center) / std::min(scale, 1.0f));
            if (w == 0) continue;
            int32_t cl = std::clamp(s, 0, int32_t(srcLen) - 1);
            for (uint32_t ch = 0; ch < channels; ++ch) acc[ch] += src[size_t(cl) * srcStride + ch] * w;
            wsum += w;
        }
        for (uint32_t ch = 0; ch < channels; ++ch) dst[size_t(d) * dstStride + ch] = acc[ch] / wsum;
    }
}
} // namespace

bool CpuBackend::upscaleSpatial(const Image& input, Image& output, uint32_t factor, float sharpen) {
    if (factor == 1) { sharpen_image(input, sharpen, output); return true; }
    const uint32_t W = input.width * factor, H = input.height * factor;
    // horizontal pass
    Image tmp(W, input.height);
    for (uint32_t y = 0; y < input.height; ++y)
        resample_axis(input.row(y), input.width, 4, tmp.row(y), W, 4, 4);
    // vertical pass
    output = Image(W, H);
    for (uint32_t x = 0; x < W; ++x) {
        std::vector<float> col(input.height), out(H);
        for (uint32_t y = 0; y < input.height; ++y) col[y] = tmp.row(y)[x * 4];
        resample_axis(col.data(), input.height, 1, out.data(), H, 1, 1);
        for (uint32_t y = 0; y < H; ++y) output.row(y)[x * 4] = clampf(out[y], 0.f, 1.f);
        // green/blue/alpha
        for (int ch = 1; ch < 4; ++ch) {
            for (uint32_t y = 0; y < input.height; ++y) col[y] = tmp.row(y)[x * 4 + ch];
            resample_axis(col.data(), input.height, 1, out.data(), H, 1, 1);
            for (uint32_t y = 0; y < H; ++y) output.row(y)[x * 4 + ch] = ch == 3 ? 1.0f : clampf(out[y], 0.f, 1.f);
        }
    }
    Image sharpened;
    sharpen_image(output, sharpen, sharpened);
    output = std::move(sharpened);
    return true;
}

// ---------------------------------------------------------------------------
// temporal ops
// ---------------------------------------------------------------------------

bool CpuBackend::estimateMotion(const Image& prevLuma, const Image& currLuma, MotionField& mv) {
    const uint32_t W = currLuma.width, H = currLuma.height;
    mv.width = W; mv.height = H;
    mv.xy.assign(size_t(W) * H * 2, 0.f);
    if (prevLuma.width != W || prevLuma.height != H) return false;
    const int block = 16, search = 6;
    std::vector<float> prevL(W * H), currL(W * H);
    for (size_t i = 0; i < W * H; ++i) {
        prevL[i] = Image::luma(&prevLuma.pixels[i * 4]);
        currL[i] = Image::luma(&currLuma.pixels[i * 4]);
    }
    for (uint32_t by = 0; by < (H + block - 1) / block; ++by) {
        for (uint32_t bx = 0; bx < (W + block - 1) / block; ++bx) {
            float best = 1e30f; int bestDx = 0, bestDy = 0;
            for (int dy = -search; dy <= search; dy += 2) {
                for (int dx = -search; dx <= search; dx += 2) {
                    float sad = 0;
                    for (int y = 0; y < block; y += 2) {
                        int cy = int(by * block) + y;
                        if (cy >= int(H)) break;
                        for (int x = 0; x < block; x += 2) {
                            int cx = int(bx * block) + x;
                            if (cx >= int(W)) break;
                            int py = std::clamp(cy + dy, 0, int(H) - 1);
                            int px = std::clamp(cx + dx, 0, int(W) - 1);
                            sad += std::fabs(currL[size_t(cy) * W + cx] - prevL[size_t(py) * W + px]);
                        }
                    }
                    if (sad < best) { best = sad; bestDx = dx; bestDy = dy; }
                }
            }
            for (int y = 0; y < block; ++y) {
                int cy = int(by * block) + y;
                if (cy >= int(H)) break;
                for (int x = 0; x < block; ++x) {
                    int cx = int(bx * block) + x;
                    if (cx >= int(W)) break;
                    mv.xy[(size_t(cy) * W + cx) * 2 + 0] = float(bestDx);
                    mv.xy[(size_t(cy) * W + cx) * 2 + 1] = float(bestDy);
                }
            }
        }
    }
    return true;
}

bool CpuBackend::reprojectHistory(const Image& historyColor, const MotionField& mv,
                                  Image& reproj, Image& confidence) {
    const uint32_t W = mv.width, H = mv.height;
    reproj = Image(W, H);
    confidence = Image(W, H);
    if (historyColor.width != W || historyColor.height != H) {
        reproj.clear(); confidence.clear();
        return true;   // no valid history yet
    }
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float dx = mv.xy[(size_t(y) * W + x) * 2 + 0];
            float dy = mv.xy[(size_t(y) * W + x) * 2 + 1];
            float sx = float(x) - dx, sy = float(y) - dy;   // where this pixel was last frame
            float* r = reproj.row(y) + x * 4;
            float* cf = confidence.row(y) + x * 4;
            if (sx < -0.5f || sy < -0.5f || sx > float(W) - 0.5f || sy > float(H) - 0.5f) {
                cf[0] = 0.f; continue;
            }
            int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
            float fx = sx - x0, fy = sy - y0;
            float acc[3] = {0, 0, 0};
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    int cx = std::clamp(x0 + i, 0, int(W) - 1);
                    int cy = std::clamp(y0 + j, 0, int(H) - 1);
                    float w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
                    const float* h = historyColor.row(uint32_t(cy)) + uint32_t(cx) * 4;
                    for (int ch = 0; ch < 3; ++ch) acc[ch] += h[ch] * w;
                }
            for (int ch = 0; ch < 3; ++ch) r[ch] = acc[ch];
            // confidence: bilinear weight mass as a proxy for disocclusion
            float edge = std::max({std::fabs(fx - 0.5f), std::fabs(fy - 0.5f)});
            cf[0] = clampf(1.0f - 1.5f * std::max(0.0f, edge - 0.25f), 0.f, 1.f);
        }
    }
    return true;
}

bool CpuBackend::temporalBlend(const Image& current, const Image& reproj,
                               const Image& confidence, float maxBlend, Image& out) {
    out = Image(current.width, current.height);
    for (uint32_t y = 0; y < current.height; ++y) {
        for (uint32_t x = 0; x < current.width; ++x) {
            const float* c = current.row(y) + x * 4;
            const float* r = reproj.row(y) + x * 4;
            const float* cf = confidence.row(y) + x * 4;
            float* d = out.row(y) + x * 4;
            float w = std::min(cf[0], maxBlend);
            for (int ch = 0; ch < 3; ++ch) d[ch] = clampf(c[ch] * (1 - w) + r[ch] * w, 0.f, 1.f);
            d[3] = 1.f;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// game mode (CPU path: our own reprojection scaler + spatial upscale)
// ---------------------------------------------------------------------------

bool CpuBackend::beginGame(uint32_t renderW, uint32_t renderH, uint32_t outputW, uint32_t outputH) {
    gRenderW_ = renderW; gRenderH_ = renderH; gOutW_ = outputW; gOutH_ = outputH;
    gHasHistory_ = false;
    gHistory_ = Image();
    return true;
}

bool CpuBackend::submitGameFrame(const GameFrameInput& in, GameFrameOutput& out) {
    if (!in.color) return false;
    auto t0 = std::chrono::steady_clock::now();

    Image lowRes = *in.color;   // render resolution
    // 1. upscale to output (spatial)
    Image upscaled;
    uint32_t factor = gRenderW_ ? gOutW_ / gRenderW_ : 2;
    if (factor < 1) factor = 1;
    upscaleSpatial(lowRes, upscaled, factor, 0.15f);

    if (in.resetHistory || !gHasHistory_) {
        out.upscaled = std::move(upscaled);
        gHistory_ = out.upscaled;
        gHasHistory_ = true;
    } else {
        // reproject output-res history with engine motion scaled to output res
        MotionField mvOut;
        mvOut.width = gOutW_; mvOut.height = gOutH_;
        mvOut.xy.assign(size_t(gOutW_) * gOutH_ * 2, 0.f);
        if (in.motion && in.motion->width == gRenderW_ && in.motion->height == gRenderH_) {
            float fx = float(gOutW_) / float(gRenderW_);
            float fy = float(gOutH_) / float(gRenderH_);
            for (uint32_t y = 0; y < gOutH_; ++y)
                for (uint32_t x = 0; x < gOutW_; ++x) {
                    uint32_t sx = std::min(x / factor, gRenderW_ - 1), sy = std::min(y / factor, gRenderH_ - 1);
                    mvOut.xy[(size_t(y) * gOutW_ + x) * 2 + 0] = in.motion->xy[(size_t(sy) * gRenderW_ + sx) * 2 + 0] * fx;
                    mvOut.xy[(size_t(y) * gOutW_ + x) * 2 + 1] = in.motion->xy[(size_t(sy) * gRenderW_ + sx) * 2 + 1] * fy;
                }
        }
        Image reproj, conf;
        reprojectHistory(gHistory_, mvOut, reproj, conf);
        temporalBlend(upscaled, reproj, conf, 0.8f, out.upscaled);
        gHistory_ = out.upscaled;
    }
    auto t1 = std::chrono::steady_clock::now();
    out.gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

void CpuBackend::endGame() {
    gHasHistory_ = false;
    gHistory_ = Image();
}

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------

const char* backend_kind_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu:    return "cpu";
        case BackendKind::Metal:  return "metal";
        case BackendKind::Vulkan: return "vulkan";
        case BackendKind::D3D12:  return "d3d12";
    }
    return "unknown";
}

std::unique_ptr<IBackend> create_backend(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu: return std::make_unique<CpuBackend>();
        case BackendKind::Metal:
#if defined(OPENDLSS_HAS_METAL)
            return create_metal_backend();
#else
            log_warn("backend: metal not compiled into this build");
            return nullptr;
#endif
        case BackendKind::Vulkan:
#if defined(OPENDLSS_HAS_VULKAN)
            return create_vulkan_backend();
#else
            log_warn("backend: vulkan not compiled into this build");
            return nullptr;
#endif
        case BackendKind::D3D12:
#if defined(OPENDLSS_HAS_D3D12)
            return create_d3d12_backend();
#else
            log_warn("backend: d3d12 not compiled into this build");
            return nullptr;
#endif
    }
    return nullptr;
}

std::vector<BackendKind> available_backends() {
    std::vector<BackendKind> out{BackendKind::Cpu};
#if defined(OPENDLSS_HAS_METAL)
    out.push_back(BackendKind::Metal);
#endif
#if defined(OPENDLSS_HAS_D3D12)
    out.push_back(BackendKind::D3D12);
#endif
#if defined(OPENDLSS_HAS_VULKAN)
    out.push_back(BackendKind::Vulkan);
#endif
    return out;
}

std::unique_ptr<IBackend> create_preferred_backend() {
#if defined(__APPLE__) && defined(OPENDLSS_HAS_METAL)
    if (auto b = create_backend(BackendKind::Metal)) return b;
#endif
#if defined(_WIN32) && defined(OPENDLSS_HAS_D3D12)
    if (auto b = create_backend(BackendKind::D3D12)) return b;
#endif
#if !defined(_WIN32) && !defined(__APPLE__) && defined(OPENDLSS_HAS_VULKAN)
    if (auto b = create_backend(BackendKind::Vulkan)) return b;
#endif
    return create_backend(BackendKind::Cpu);
}

} // namespace opendlss
