// E4M3 (FP8, OCP/NVIDIA flavor) codec.
//
//   1 sign | 4 exponent (bias 7) | 3 mantissa
//   max normal = 448, no infinities; S.1111.111 is NaN.
//   Conversions round to nearest-even and saturate to +/-448.
//
// The network family publishes activations and weights on the E4M3 grid
// while accumulating in half. All backends share this codec so the
// quantization grid is identical everywhere.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>

namespace opendlss {

struct E4M3 {
    static constexpr uint8_t kNaN     = 0x7F;
    static constexpr float   kMaxValue = 448.0f;

    // f32 -> E4M3 byte, round-to-nearest-even, saturating.
    static uint8_t encode(float value) {
        if (std::isnan(value)) return kNaN;
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        uint32_t sign = (bits >> 31) << 7;                    // 0x80
        if (std::isinf(value)) return uint8_t(sign | 0x7E);   // saturate to max normal
        int32_t exp = int32_t((bits >> 23) & 0xFFu) - 127;
        uint32_t man = bits & 0x007FFFFFu;

        if (((bits >> 23) & 0xFFu) == 0) {                    // f32 zero/subnormal -> 0
            // E4M3's smallest subnormal is 2^-9; f32 subnormals below that flush to 0.
            if (value == 0.0f) return uint8_t(sign);
            float av = std::fabs(value);
            if (av < 0.5f * min_subnormal()) return uint8_t(sign);
        }

        if (exp < -6) {                                       // subnormal in E4M3
            // |value| = m/8 * 2^-6 = m * 2^-9  ->  m = |value| * 2^9, m in 0..7
            if (exp < -30) return uint8_t(sign);
            const float av = std::fabs(value);
            const float m_f = av * 512.0f;
            uint32_t m = uint32_t(m_f);
            float frac = m_f - float(m);
            if (frac > 0.5f || (frac == 0.5f && (m & 1u))) ++m;
            if (m >= 8) {                                     // rounds up to min normal 2^-6
                return uint8_t(sign | (1u << 3));
            }
            return uint8_t(sign | m);
        }
        if (exp > 8 || (exp == 8 && man > 0x00400000u)) {     // > 448 -> saturate
            // 448 = 1.75 * 2^8 = e8 m110; values that round beyond that saturate.
            return uint8_t(sign | 0x7E);
        }
        uint32_t man3 = man >> 20;
        uint32_t rem  = man & 0x000FFFFFu;
        uint32_t half = 1u << 19;
        if (rem > half || (rem == half && (man3 & 1u))) {
            ++man3;
            if (man3 == 8) { man3 = 0; ++exp; if (exp > 8) return uint8_t(sign | 0x7E); }
        }
        return uint8_t(sign | uint32_t(exp + 7) << 3 | man3);
    }

    static float decode(uint8_t code) {
        uint32_t sign = uint32_t(code & 0x80u) << 24;
        uint32_t exp  = (code >> 3) & 0x0Fu;
        uint32_t man  = code & 0x07u;
        if (exp == 15 && man == 7) {
            (void)sign;
            return std::numeric_limits<float>::quiet_NaN();
        }
        float v;
        if (exp == 0) {
            if (man == 0) v = 0.0f;                           // zero
            else v = std::ldexp(float(man), -9);              // m * 2^-9 subnormal
        } else {
            v = std::ldexp(1.0f + float(man) / 8.0f, int(exp) - 7);
        }
        return sign ? -v : v;
    }

    static constexpr float min_subnormal() { return std::ldexp(1.0f, -9); } // 2^-9
};

// Quantize f32 onto the E4M3 grid and read it back as f32 (what "publication" means).
inline float e4m3_publish(float v) { return E4M3::decode(E4M3::encode(v)); }

} // namespace opendlss
