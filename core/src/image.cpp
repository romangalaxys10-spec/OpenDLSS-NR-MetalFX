// Image I/O and generators.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/image.h"
#include "opendlss/logging.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

#ifdef OPENDLSS_STB_IMPLEMENTATION_HERE
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#else
#include "../../third_party/stb_image.h"
#include "../../third_party/stb_image_write.h"
#endif

namespace opendlss {

namespace {
uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}
float gauss(uint64_t& s) {
    // Box-Muller; the network's noise lanes use a hash of the pixel, we use a
    // deterministic PRNG stream — the distribution is what matters here.
    float u1 = float((xorshift64(s) >> 11) * (1.0 / 9007199254740992.0));
    float u2 = float((xorshift64(s) >> 11) * (1.0 / 9007199254740992.0));
    if (u1 < 1e-12f) u1 = 1e-12f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}
} // namespace

std::unique_ptr<Image> image_load(const std::string& path) {
    int w = 0, h = 0, comp = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!data) {
        // try 16-bit
        stbi_us* d16 = stbi_load_16(path.c_str(), &w, &h, &comp, 4);
        if (!d16) {
            log_error("image_load: cannot load '%s': %s", path.c_str(), stbi_failure_reason());
            return nullptr;
        }
        auto img = std::make_unique<Image>(uint32_t(w), uint32_t(h));
        for (size_t i = 0; i < img->pixels.size(); ++i)
            img->pixels[i] = float(d16[i]) / 65535.0f;
        stbi_image_free(d16);
        return img;
    }
    auto img = std::make_unique<Image>(uint32_t(w), uint32_t(h));
    for (size_t i = 0; i < img->pixels.size(); ++i)
        img->pixels[i] = float(data[i]) / 255.0f;
    stbi_image_free(data);
    return img;
}

bool image_save_png(const std::string& path, const Image& img, uint32_t bitrate) {
    if (img.empty()) { log_error("image_save_png: empty image"); return false; }
    if (bitrate <= 8) {
        std::vector<stbi_uc> data(img.pixels.size());
        for (size_t i = 0; i < data.size(); ++i) {
            float v = std::clamp(img.pixels[i], 0.0f, 1.0f);
            data[i] = stbi_uc(std::lround(v * 255.0f));
        }
        if (!stbi_write_png(path.c_str(), int(img.width), int(img.height), 4, data.data(), int(img.width) * 4)) {
            log_error("image_save_png: cannot write '%s'", path.c_str());
            return false;
        }
        return true;
    }
    std::vector<stbi_us> data(img.pixels.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float v = std::clamp(img.pixels[i], 0.0f, 1.0f);
        data[i] = stbi_us(std::lround(v * 65535.0f));
    }
    (void)data; // stb_image_write has no 16-bit writer; keep the vector for future use.
    log_warn("image_save_png: 16-bit write unsupported, writing 8-bit");
    return image_save_png(path, img, 8);
}

bool image_save_pfm(const std::string& path, const Image& img) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "PF\n%u %u\n-1.0\n", img.width, img.height);
    for (uint32_t y = 0; y < img.height; ++y) {
        const float* row = img.row(img.height - 1 - y);   // PFM is bottom-up
        for (uint32_t x = 0; x < img.width; ++x) {
            float rgb[3] = { row[x*4+0], row[x*4+1], row[x*4+2] };
            std::fwrite(rgb, 4, 3, f);
        }
    }
    std::fclose(f);
    return true;
}

std::unique_ptr<Image> image_synthetic_scene(uint32_t w, uint32_t h) {
    auto img = std::make_unique<Image>(w, h);
    const float cy = h * 0.5f;
    (void)cy;
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float fx = float(x) + 0.5f, fy = float(y) + 0.5f;
            // Sky gradient
            float sky = 0.15f + 0.35f * (fy / float(h));
            float r = sky * 0.75f, g = sky * 0.85f, b = sky;
            // Sun disc
            float ds = std::sqrt((fx - w * 0.78f) * (fx - w * 0.78f) + (fy - h * 0.2f) * (fy - h * 0.2f));
            if (ds < w * 0.06f) { r = g = b = 1.0f; }
            else if (ds < w * 0.075f) { float t = 1.0f - (ds - w * 0.06f) / (w * 0.015f); r += t * 0.8f; g += t * 0.55f; b += t * 0.2f; }
            // Mountains (two ridges, deterministic pseudo-noise)
            auto ridge = [&](float period, float amp, float base) {
                float v = base + amp * (0.6f * std::sin(fx / period) + 0.4f * std::sin(fx / (period * 0.37f) + 1.7f));
                return v * float(h);
            };
            float ridge1 = ridge(37.0f, 0.05f, 0.62f), ridge2 = ridge(53.0f, 0.07f, 0.78f);
            if (fy > ridge2) { r = 0.16f; g = 0.18f; b = 0.22f; }
            if (fy > ridge1) { r = 0.09f; g = 0.11f; b = 0.13f; }
            // Water band with "specular" ripples
            if (fy > ridge1) {
                float ripple = std::sin(fx * 0.35f + std::sin(fy * 0.9f) * 2.0f) * 0.5f + 0.5f;
                r = 0.05f + 0.03f * ripple; g = 0.07f + 0.04f * ripple; b = 0.10f + 0.05f * ripple;
            }
            float* px = img->row(y) + x * 4;
            px[0] = std::clamp(r, 0.f, 1.f); px[1] = std::clamp(g, 0.f, 1.f);
            px[2] = std::clamp(b, 0.f, 1.f); px[3] = 1.0f;
        }
    }
    return img;
}

std::unique_ptr<Image> image_add_gaussian_noise(const Image& src, float sigma, uint64_t seed) {
    auto out = std::make_unique<Image>(src.width, src.height);
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < src.pixels.size(); i += 4) {
        float n = gauss(s);
        for (int c = 0; c < 3; ++c) out->pixels[i + c] = std::clamp(src.pixels[i + c] + sigma * n, 0.0f, 1.0f);
        out->pixels[i + 3] = src.pixels[i + 3];
    }
    return out;
}

std::unique_ptr<Image> image_downscale_box(const Image& src, uint32_t factor) {
    uint32_t w = src.width / factor, h = src.height / factor;
    auto out = std::make_unique<Image>(w, h);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float acc[4] = {0, 0, 0, 0};
            for (uint32_t dy = 0; dy < factor; ++dy)
                for (uint32_t dx = 0; dx < factor; ++dx) {
                    const float* p = src.row(y * factor + dy) + (x * factor + dx) * 4;
                    for (int c = 0; c < 4; ++c) acc[c] += p[c];
                }
            float* o = out->row(y) + x * 4;
            for (int c = 0; c < 4; ++c) o[c] = acc[c] / float(factor * factor);
        }
    }
    return out;
}

std::unique_ptr<Image> image_crop(const Image& src, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    auto out = std::make_unique<Image>(w, h);
    for (uint32_t yy = 0; yy < h; ++yy)
        for (uint32_t xx = 0; xx < w; ++xx)
            std::memcpy(out->row(yy) + xx * 4, src.row(std::min(y + yy, src.height - 1)) + std::min(x + xx, src.width - 1) * 4, 16);
    return out;
}

std::unique_ptr<Image> image_side_by_side(const Image& a, const Image& b) {
    uint32_t h = std::max(a.height, b.height);
    auto out = std::make_unique<Image>(a.width + b.width + 4, h);
    for (uint32_t y = 0; y < a.height; ++y)
        std::memcpy(out->row(y), a.row(y), size_t(a.width) * 16);
    for (uint32_t y = 0; y < b.height; ++y)
        std::memcpy(out->row(y) + (a.width + 4) * 4, b.row(y), size_t(b.width) * 16);
    return out;
}

double image_psnr(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height) return -1.0;
    double mse = 0.0; size_t n = 0;
    for (size_t i = 0; i < a.pixels.size(); i += 4) {
        for (int c = 0; c < 3; ++c) {
            double d = double(a.pixels[i + c]) - double(b.pixels[i + c]);
            mse += d * d; ++n;
        }
    }
    mse /= double(n);
    if (mse <= 1e-12) return 99.0;
    return 10.0 * std::log10(1.0 / mse);
}

} // namespace opendlss
