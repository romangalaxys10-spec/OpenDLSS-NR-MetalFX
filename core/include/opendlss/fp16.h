// Software IEEE-754 binary16 ("half") codec and the network's half bit-tricks.
//
// The DLSS-NR network family publishes every intermediate through half precision
// and accumulates in half; a few of its nonlinearities (the attention
// exponential, the softmax) are specified as *bit operations on the half
// pattern* rather than as float math. Every backend (CPU, Metal, D3D12,
// Vulkan) therefore shares this one implementation so the numerics are
// identical everywhere. Host-endian little is assumed (all four targets).
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

namespace opendlss {

// ---------- round-to-nearest-even f32 -> f16, identical semantics to hardware F16C ----------
inline uint16_t f32_to_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = int32_t((bits >> 23) & 0xFFu) - 127;   // unbiased
    uint32_t man  = bits & 0x007FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFF) {                  // Inf / NaN
        if (man != 0) {
            // NaN: keep top mantissa bit so it stays a NaN in half.
            uint16_t m16 = uint16_t(man >> 13);
            if (m16 == 0) m16 = 1;
            return uint16_t(sign | 0x7C00u | m16);
        }
        return uint16_t(sign | 0x7C00u);
    }
    if (exp > 15)  return uint16_t(sign | 0x7C00u);        // overflow -> Inf
    if (exp >= -14) {                                      // normal
        uint32_t man16 = man >> 13;
        // round to nearest even on the 13 dropped bits
        uint32_t round = (man >> 12) & 1u;
        man16 += round;
        if (man16 >> 10) { man16 = 0; ++exp; }             // mantissa overflow
        if (exp > 15) return uint16_t(sign | 0x7C00u);
        return uint16_t(sign | uint32_t(exp + 15) << 10 | man16);
    }
    if (exp < -25) return sign;                            // underflow to zero (RTNE)
    // subnormal: shift with rounding
    uint32_t shift = uint32_t(-14 - exp);                  // 1..24
    uint32_t manFull = man | 0x00800000u;                  // implicit leading 1
    uint32_t man16  = manFull >> (13 + shift);
    uint32_t rem    = manFull << (19 - shift);             // bits dropped, MSB-aligned
    uint32_t halfUlp = 1u << 31;
    if (rem > halfUlp || (rem == halfUlp && (man16 & 1u))) ++man16;
    return uint16_t(sign | man16);
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {                                          // subnormal -> normalize
            int e = -1;
            uint32_t m = man;
            do { ++e; m <<= 1; } while ((m & 0x0400u) == 0);
            m &= 0x03FFu;
            bits = sign | uint32_t(127 - 15 - e + 1) << 23 | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | uint32_t(exp - 15 + 127) << 23 | (man << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

struct Half {
    uint16_t bits{};
    Half() = default;
    Half(float v) : bits(f32_to_f16(v)) {}
    operator float() const { return f16_to_f32(bits); }
    static Half from_bits(uint16_t b) { Half h; h.bits = b; return h; }
};

// ---------- the network's half bit-tricks ----------
// Attention exponential (window blocks): with s the cosine logit,
//   x = clamp(f16(0.044921875 s + 1.30078125), 1.03125, 1.5693359375)
//   exp = reinterpret_half(bits(x) << 5 ^ 0x8000)   == 2^(1.4375 s - 5.375)
inline Half window_exp_trick(float s) {
    Half x(0.044921875f * s + 1.30078125f);
    float xf = x;                                     // through the half grid
    float lo = 1.03125f, hi = 1.5693359375f;
    if (xf < lo) xf = lo;
    if (xf > hi) xf = hi;
    Half c(xf);
    uint16_t b = uint16_t((uint32_t(c.bits) << 5) & 0xFFFFu);
    b = uint16_t(b ^ 0x8000u);
    return Half::from_bits(b);
}

// Attention exponential (global ViT blocks):
//   x = clamp(f16(0.08953947 s + 1.70936143), 1.439453125, 1.9775390625)
//   exp = reinterpret_half((bits(x) << 4) + 0x4000) == 2^(1.4326 s - 3.6502)
inline Half global_exp_trick(float s) {
    Half x(0.08953947f * s + 1.70936143f);
    float xf = x;
    float lo = 1.439453125f, hi = 1.9775390625f;
    if (xf < lo) xf = lo;
    if (xf > hi) xf = hi;
    Half c(xf);
    uint16_t b = uint16_t((uint32_t(c.bits) << 4) & 0xFFFFu);
    b = uint16_t(b + 0x4000u);
    return Half::from_bits(b);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

} // namespace opendlss
