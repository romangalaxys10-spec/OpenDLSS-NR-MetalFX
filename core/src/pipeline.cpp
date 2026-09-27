// Pipeline implementation: preprocess packing, head composite, mode drivers.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/pipeline.h"
#include "opendlss/fp16.h"
#include "opendlss/logging.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace opendlss {

// ---------------- noise + preprocess ----------------

// xxhash-style avalanche; identical constants in every shader implementation.
uint32_t preprocess_pixel_hash(uint32_t x, uint32_t y, uint32_t frameIndex, uint64_t seed) {
    uint64_t h = seed ^ (uint64_t(x) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(y) * 0xC2B2AE3D27D4EB4Full)
               ^ (uint64_t(frameIndex) * 0x165667B19E3779F9ull);
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return uint32_t(h ^ (h >> 32));
}

int32_t mirror_coord(int32_t v, uint32_t valid) {
    // 2*valid - x - 2, no edge repeat. For v in [0, valid-1] returns v.
    int32_t n = int32_t(valid);
    if (v >= 0 && v < n) return v;
    return 2 * n - v - 2;
}

void preprocess_pack_features(const Geometry& geom,
                              const Image& proxy,
                              const Image* history,
                              const ConditionScalars& cond,
                              uint64_t frameSeed,
                              std::vector<uint16_t>& features) {
    const uint32_t FW = geom.fieldWidth, FH = geom.fieldHeight;
    const uint32_t VW = geom.validWidth, VH = geom.validHeight;
    features.assign(size_t(FW) * FH * 16, 0);

    for (uint32_t py = 0; py < FH; ++py) {
        for (uint32_t px = 0; px < FW; ++px) {
            // noise hash uses the padded coordinate
            uint32_t h0 = preprocess_pixel_hash(px, py, 0u, frameSeed);
            uint32_t h1 = preprocess_pixel_hash(px, py, 0x9E37u, frameSeed);
            float u1 = float(h0 >> 8) / 16777216.0f;
            float u2 = float(h1 >> 8) / 16777216.0f;
            if (u1 < 1e-9f) u1 = 1e-9f;
            float g0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
            float g1 = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2 + 2.0943951f);
            float g2 = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2 + 4.1887902f);

            // image sample is mirrored off the valid rectangle
            int32_t sx = mirror_coord(int32_t(px), VW);
            int32_t sy = mirror_coord(int32_t(py), VH);
            sx = std::clamp(sx, 0, int32_t(VW) - 1);
            sy = std::clamp(sy, 0, int32_t(VH) - 1);
            const float* p = proxy.row(uint32_t(sy)) + uint32_t(sx) * 4;
            const float* q = history ? history->row(uint32_t(sy)) + uint32_t(sx) * 4 : nullptr;

            float* lane = nullptr;   // filled through f16 writes below
            uint16_t* out = features.data() + (size_t(py) * FW + px) * 16;
            (void)lane;

            out[0] = f32_to_f16(g0);
            out[1] = f32_to_f16(g1);
            out[2] = f32_to_f16(g2);
            out[3] = f32_to_f16(1.0f);
            for (int c = 0; c < 3; ++c)
                out[4 + c] = f32_to_f16(f32_to_f16(std::clamp(p[c], 0.f, 1.f) - 0.5f) * 0.125f);
            if (q) {
                for (int c = 0; c < 3; ++c)
                    out[7 + c] = f32_to_f16(f32_to_f16(std::clamp(q[c], 0.f, 1.f) - 0.5f) * 0.125f);
            } else {
                for (int c = 0; c < 3; ++c) out[7 + c] = out[4 + c];
            }
            out[10] = f32_to_f16(cond.style);
            out[11] = f32_to_f16(cond.tone);
            out[12] = f32_to_f16(cond.structure);
            out[13] = f32_to_f16(cond.skin);
            out[14] = f32_to_f16(cond.autoMask);
            out[15] = f32_to_f16(0.0f);
        }
    }
}

void head_composite(const Image& proxy, const float* headRGBA, const Image* history,
                    float blendScale, Image& display) {
    const uint32_t W = proxy.width, H = proxy.height;
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            const float* p = proxy.row(y) + x * 4;
            const float* h = headRGBA + (size_t(y) * W + x) * 4;
            float* d = display.row(y) + x * 4;
            for (int c = 0; c < 3; ++c) {
                float neural = clampf(p[c] + h[c] / 4.0f, 0.0f, 1.0f);
                if (history) {
                    float weight = clampf(sigmoid(h[3]) * blendScale, 0.0f, 1.0f);
                    d[c] = neural * (1.0f - weight) + (*history).row(y)[x * 4 + c] * weight;
                } else {
                    d[c] = neural;
                }
            }
            d[3] = 1.0f;
        }
    }
}

void sharpen_image(const Image& src, float amount, Image& dst) {
    dst = Image(src.width, src.height);
    if (amount <= 0.0f) { std::memcpy(dst.pixels.data(), src.pixels.data(), src.pixels.size() * 4); return; }
    for (uint32_t y = 0; y < src.height; ++y) {
        for (uint32_t x = 0; x < src.width; ++x) {
            const float* c = src.row(y) + x * 4;
            float blur[3] = {0, 0, 0};
            float wsum = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    int sx = std::clamp(int(x) + dx, 0, int(src.width) - 1);
                    int sy = std::clamp(int(y) + dy, 0, int(src.height) - 1);
                    const float* n = src.row(uint32_t(sy)) + uint32_t(sx) * 4;
                    float w = (dx && dy) ? 1.0f : 2.0f;
                    for (int ch = 0; ch < 3; ++ch) blur[ch] += n[ch] * w;
                    wsum += w;
                }
            }
            float* d = dst.row(y) + x * 4;
            for (int ch = 0; ch < 3; ++ch) {
                float b = blur[ch] / wsum;
                float lo = std::min({c[0], c[1], c[2]}), hi = std::max({c[0], c[1], c[2]});
                // adaptive: suppress sharpening across strong edges to avoid halos
                float adapt = 1.0f - std::clamp((hi - lo) * 3.0f, 0.0f, 1.0f);
                d[ch] = clampf(c[ch] + (c[ch] - b) * amount * adapt, 0.0f, 1.0f);
            }
            d[3] = 1.0f;
        }
    }
}

// ---------------- Y4M ----------------

bool Y4mReader::open(const std::string& path) {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) { log_error("y4m: cannot open %s", path.c_str()); return false; }
    return parseHeader();
}

bool Y4mReader::parseHeader() {
    char magic[10] = {0};
    if (std::fscanf(f_, "%9s", magic) != 1 || std::string(magic) != "YUV4MPEG2") {
        log_error("y4m: bad magic");
        return false;
    }
    int c;
    std::string tok;
    bool gotSize = false, gotFps = false;
    while ((c = std::fgetc(f_)) != '\n' && c != EOF) {
        if (c == ' ') {
            if (tok.empty()) continue;
            char kind = tok[0];
            if (kind == 'W') { width_ = uint32_t(std::atoi(tok.c_str() + 1)); gotSize = true; }
            else if (kind == 'H') { height_ = uint32_t(std::atoi(tok.c_str() + 1)); gotSize = true; }
            else if (kind == 'F') { int num = 0, den = 1; std::sscanf(tok.c_str() + 1, "%d:%d", &num, &den); fps_ = den ? double(num) / double(den) : 24.0; gotFps = true; }
            else params_ += tok + " ";
            tok.clear();
        } else tok += char(c);
    }
    if (!gotSize || !gotFps || !width_ || !height_) { log_error("y4m: incomplete header"); return false; }
    yuv_.resize(size_t(width_) * height_ * 3 / 2);
    return true;
}

bool Y4mReader::next(Image& frame) {
    char tag[6] = {0};
    if (std::fread(tag, 1, 5, f_) != 5) return false;
    if (std::string(tag) != "FRAME") return false;
    // consume rest of the FRAME line
    int c;
    while ((c = std::fgetc(f_)) != '\n' && c != EOF) {}
    if (std::fread(yuv_.data(), 1, yuv_.size(), f_) != yuv_.size()) return false;

    frame = Image(width_, height_);
    const uint8_t* Y = yuv_.data();
    const uint8_t* U = Y + size_t(width_) * height_;
    const uint8_t* V = U + size_t(width_ / 2) * (height_ / 2);
    for (uint32_t y = 0; y < height_; ++y) {
        for (uint32_t x = 0; x < width_; ++x) {
            float yy = float(Y[size_t(y) * width_ + x]) - 16.0f;
            float uu = float(U[(y / 2) * (width_ / 2) + (x / 2)]) - 128.0f;
            float vv = float(V[(y / 2) * (width_ / 2) + (x / 2)]) - 128.0f;
            float r = 1.164f * yy / 219.0f + 1.793f * vv / 224.0f + 0.5f;
            float g = 1.164f * yy / 219.0f - 0.213f * uu / 224.0f - 0.533f * vv / 224.0f + 0.5f;
            float b = 1.164f * yy / 219.0f + 2.112f * uu / 224.0f + 0.5f;
            float* d = frame.row(y) + x * 4;
            d[0] = clampf(r, 0.f, 1.f); d[1] = clampf(g, 0.f, 1.f); d[2] = clampf(b, 0.f, 1.f); d[3] = 1.f;
        }
    }
    return true;
}

bool Y4mWriter::open(const std::string& path, uint32_t w, uint32_t h, double fps) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) return false;
    width_ = w; height_ = h;
    int num = int(std::lround(fps * 1000.0)), den = 1000;
    // reduce
    int a = num, b = den;
    while (b) { int t = a % b; a = b; b = t; }
    if (a > 0) { num /= a; den /= a; }
    std::fprintf(f_, "YUV4MPEG2 W%u H%u F%d:%d Ip A1:1 C420jpeg\n", w, h, num, den);
    yuv_.resize(size_t(w) * h * 3 / 2);
    return true;
}

bool Y4mWriter::write(const Image& frame) {
    uint8_t* Y = yuv_.data();
    uint8_t* U = Y + size_t(width_) * height_;
    uint8_t* V = U + size_t(width_ / 2) * (height_ / 2);
    std::vector<float> uAcc(size_t(width_ / 2) * (height_ / 2), 0.f);
    std::vector<float> vAcc(uAcc.size(), 0.f);
    for (uint32_t y = 0; y < height_; ++y) {
        for (uint32_t x = 0; x < width_; ++x) {
            const float* p = frame.row(y) + x * 4;
            float r = p[0], g = p[1], b = p[2];
            float y8 = 16.0f + 219.0f * (0.2126f * r + 0.7152f * g + 0.0722f * b);
            float u8 = 128.0f + 224.0f * (-0.2126f * r - 0.7152f * g + 0.9278f * b) * 0.564f;
            float v8 = 128.0f + 224.0f * (0.9278f * r - 0.7152f * g - 0.2126f * b) * 0.713f;
            Y[size_t(y) * width_ + x] = uint8_t(std::clamp(y8, 16.f, 235.f) + 0.5f);
            uAcc[(y / 2) * (width_ / 2) + (x / 2)] += clampf(u8, 16.f, 240.f);
            vAcc[(y / 2) * (width_ / 2) + (x / 2)] += clampf(v8, 16.f, 240.f);
        }
    }
    size_t n = uAcc.size();
    for (size_t i = 0; i < n; ++i) { U[i] = uint8_t(uAcc[i] * 0.25f + 0.5f); V[i] = uint8_t(vAcc[i] * 0.25f + 0.5f); }
    std::fwrite("FRAME\n", 1, 6, f_);
    return std::fwrite(yuv_.data(), 1, yuv_.size(), f_) == yuv_.size();
}

void Y4mWriter::close() { if (f_) { std::fclose(f_); f_ = nullptr; } }

} // namespace opendlss
