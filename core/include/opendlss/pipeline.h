// Pipeline orchestration: the shared frame semantics of the network family.
//
// Preprocess packs the 16-lane feature field (identical lane order to the
// reference), the head composes the final display frame, and the mode
// drivers (image / video / game) wire the stages together. GPU backends
// implement the same math in shaders; this CPU path is the parity golden.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include "opendlss/backend.h"

namespace opendlss {

// 16 input lanes (order matches the reference preprocess):
//   0-2   three Gaussian lanes, Box-Muller from a hash of the *padded* pixel
//         coordinate and a per-frame seed
//   3     constant 1
//   4-6   the display proxy, centred: f16((f16(c) - 0.5) * 0.125) per channel
//   7-9   the same for the reprojected previous output; copy of 4-6 with no history
//   10    style id / 128
//   11    local tone
//   12-14 structure / skin / auto-mask conditioning
//   15    0
struct ConditionScalars {
    float style = 0.0f, tone = 0.5f, structure = 0.5f, skin = 0.5f, autoMask = 0.5f;
};

// Pack features for the valid rect (proxy/history sRGB 0..1). Outside the
// valid rectangle the caller mirrors the image (see sample_mirrored) while
// noise still uses the padded coordinate.
void preprocess_pack_features(const Geometry& geom,
                              const Image& proxy,        // valid-size, sRGB codes
                              const Image* history,      // valid-size or nullptr
                              const ConditionScalars& cond,
                              uint64_t frameSeed,
                              std::vector<uint16_t>& features /* fieldH*fieldW*16 */);

// Pixel hash used by the noise lanes (must match the shader implementation).
uint32_t preprocess_pixel_hash(uint32_t x, uint32_t y, uint32_t frameIndex, uint64_t seed);

// Mirrored field sampling: coordinate outside [0, valid-1] reflects off the
// valid edge: 2*valid - x - 2 (no edge repeat), per the reference.
int32_t mirror_coord(int32_t v, uint32_t valid);

// Head composition:
//   neural  = clamp(proxy + rgb / 4, 0, 1)
//   weight  = clamp(sigmoid(logit) * blendScale, 0, 1)
//   display = lerp(neural, history, weight)
void head_composite(const Image& proxy, const float* headRGBA, const Image* history,
                    float blendScale, Image& display);

// Adaptive spatial sharpen (CAS-style, used on non-MetalFX paths).
void sharpen_image(const Image& src, float amount, Image& dst);

// ---------------------------------------------------------------------------
// Mode drivers (shared across backends; backend provides the ops).
// ---------------------------------------------------------------------------

struct ImageJob {
    std::string inputPath, outputPath;
    PipelineOptions opts;
    std::string modelDir;     // optional
    BackendKind backend = BackendKind::Cpu;
};

bool run_image_job(const ImageJob& job);

struct VideoJob {
    std::string inputPath, outputPath;   // Y4M in/out (or image sequences)
    PipelineOptions opts;
    std::string modelDir;
    BackendKind backend = BackendKind::Cpu;
    uint32_t maxFrames = 0;              // 0 = all
};

bool run_video_job(const VideoJob& job);

// Y4M reader/writer (12-bit-safe 8-bit 4:2:0 planar; converts to RGBA).
class Y4mReader {
public:
    bool open(const std::string& path);
    bool next(Image& frame);
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    double fps() const { return fps_; }
private:
    std::FILE* f_ = nullptr;
    uint32_t width_ = 0, height_ = 0;
    double fps_ = 24.0;
    std::string params_;
    std::vector<uint8_t> yuv_;
    bool parseHeader();
};

class Y4mWriter {
public:
    bool open(const std::string& path, uint32_t w, uint32_t h, double fps);
    bool write(const Image& frame);
    void close();
private:
    std::FILE* f_ = nullptr;
    uint32_t width_ = 0, height_ = 0;
    std::vector<uint8_t> yuv_;
};

} // namespace opendlss
