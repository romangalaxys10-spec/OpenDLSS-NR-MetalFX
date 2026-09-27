// RGBA f32 image buffer + PNG/PPM I/O (via vendored stb, public domain).
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

namespace opendlss {

struct Image {
    uint32_t width = 0, height = 0;
    std::vector<float> pixels;              // RGBA, row-major, [y * width + x] * 4

    Image() = default;
    Image(uint32_t w, uint32_t h) : width(w), height(h), pixels(size_t(w) * h * 4, 0.f) {}

    float*       row(uint32_t y)       { return pixels.data() + size_t(y) * width * 4; }
    const float* row(uint32_t y) const { return pixels.data() + size_t(y) * width * 4; }
    size_t byteSize() const { return pixels.size() * sizeof(float); }
    bool empty() const { return width == 0 || height == 0; }

    void clear() { pixels.assign(pixels.size(), 0.f); }

    // sRGB code value (0..1) -> luminance-weighted luma approximation.
    static float luma(const float* rgba) {
        return 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2];
    }
};

// PNG (8/16-bit) and PPM/PGM (P5/P6, ASCII header) readers; PNG writer.
// Returns nullopt and logs on failure.
std::unique_ptr<Image> image_load(const std::string& path);
bool image_save_png(const std::string& path, const Image& img, uint32_t bitrate = 8);
bool image_save_pfm(const std::string& path, const Image& img);   // RGB float dump (debug)

// Simple deterministic test-image generators (no external assets needed).
std::unique_ptr<Image> image_synthetic_scene(uint32_t w, uint32_t h);
std::unique_ptr<Image> image_add_gaussian_noise(const Image& src, float sigma, uint64_t seed);
std::unique_ptr<Image> image_downscale_box(const Image& src, uint32_t factor);  // simulate low-res
std::unique_ptr<Image> image_crop(const Image& src, uint32_t x, uint32_t y, uint32_t w, uint32_t h);

// Side-by-side comparison image with labels strip (for demo artifacts).
std::unique_ptr<Image> image_side_by_side(const Image& a, const Image& b);

// PSNR between two images (on RGB), in dB. > 30 is usable, > 40 good.
double image_psnr(const Image& a, const Image& b);

} // namespace opendlss
