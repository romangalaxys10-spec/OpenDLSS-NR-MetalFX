// Common.metal — shared helpers for every OpenDLSS-NR Metal kernel.
// The bit-tricks here MUST stay identical to core/include/opendlss/fp16.h and
// e4m3.h (the CPU reference) — they are checked by the parity harness.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.

#include <metal_stdlib>
using namespace metal;

// ---------- pixel hash (matches preprocess_pixel_hash) ----------
static inline uint odl_hash(uint x, uint y, uint frameIndex, uint seed) {
    ulong h = (ulong)seed
            ^ ((ulong)x * 0x9E3779B97F4A7C15UL)
            ^ ((ulong)y * 0xC2B2AE3D27D4EB4FUL)
            ^ ((ulong)frameIndex * 0x165667B19E3779F9UL);
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDUL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53UL;
    h ^= h >> 33;
    return (uint)(h ^ (h >> 32));
}

// ---------- E4M3 (OCP FP8) ----------
// 1 | 4 (bias 7) | 3 ; max normal 448 ; 0xFF/0x7F = NaN ; saturating RTNE encode.
static inline float odl_e4m3_decode(uint8_t code) {
    uint s = (code & 0x80u) ? 0x80000000u : 0u;
    uint e = (code >> 3) & 0x0Fu;
    uint m = code & 0x07u;
    if (e == 15 && m == 7) return NAN;
    float v;
    if (e == 0) v = (m == 0) ? 0.0f : ldexp((float)m, -9);        // m * 2^-9
    else        v = ldexp(1.0f + (float)m / 8.0f, (int)e - 7);
    return s ? -v : v;
}

static inline uint8_t odl_e4m3_encode(float value) {
    if (as_type<uint>(value) == 0x7F800000u || as_type<uint>(value) == 0xFF800000u)
        return (value < 0) ? 0xFE : 0x7E;                          // inf saturates
    if (value != value) return 0x7F;                               // NaN
    bool neg = value < 0.0f;
    float a = fabs(value);
    if (a >= 448.0f) return neg ? 0xFE : 0x7E;
    // subnormal grid m * 2^-9
    float m_f = a * 512.0f;
    if (m_f < 8.0f) {
        uint m = (uint)m_f;
        float frac = m_f - (float)m;
        if (frac > 0.5f || (frac == 0.5f && (m & 1u))) ++m;
        if (m >= 8) return neg ? 0x88 : 0x08;                      // min normal 2^-6
        return (uint8_t)((neg ? 0x80u : 0u) | m);
    }
    // normal: exponent e in [-6, 8], mantissa 3 bits, RTNE
    int e = (int)floor(log2(a));
    float mant = a / ldexp(1.0f, e);                               // [1, 2)
    float q = mant * 8.0f;
    uint mi = (uint)q;                                             // 8..16
    float frac = q - (float)mi;
    if (frac > 0.5f || (frac == 0.5f && (mi & 1u))) ++mi;
    if (mi >= 16) { mi >>= 1; ++e; }                               // carry
    mi -= 8;                                                       // 3-bit mantissa
    if (e > 8 || (e == 8 && mi > 6)) return neg ? 0xFE : 0x7E;     // saturate 448
    return (uint8_t)((neg ? 0x80u : 0u) | (uint(e + 7) << 3) | mi);
}

// publication = quantize onto the E4M3 grid, keep f32
static inline float odl_pub_e4m3(float v) { return odl_e4m3_decode(odl_e4m3_encode(v)); }

// ---------- attention exponent bit-trick (window blocks) ----------
// x = clamp(half(0.044921875 s + 1.30078125), 1.03125, 1.5693359375)
// exp = reinterpret_half((bits(x) << 5) ^ 0x8000) == 2^(1.4375 s - 5.375)
static inline half odl_window_exp(float s) {
    half x = half(0.044921875h * half(s) + 1.30078125h);
    float xf = float(x);
    xf = clamp(xf, 1.03125f, 1.5693359375f);
    ushort b = (ushort)(as_type<ushort>(half(xf)) << 5);
    b ^= 0x8000u;
    return as_type<half>(b);
}

// global ViT variant:
// x = clamp(half(0.08953947 s + 1.70936143), 1.439453125, 1.9775390625)
// exp = reinterpret_half((bits(x) << 4) + 0x4000) == 2^(1.4326 s - 3.6502)
static inline half odl_global_exp(float s) {
    half x = half(0.08953947h * half(s) + 1.70936143h);
    float xf = float(x);
    xf = clamp(xf, 1.439453125f, 1.9775390625f);
    ushort b = (ushort)(as_type<ushort>(half(xf)) << 4);
    b = (ushort)(b + 0x4000u);
    return as_type<half>(b);
}

// ---------- the 16 input lanes ----------
struct OdlConditioning {
    float style;      // lane 10
    float tone;       // lane 11
    float structure;  // lane 12
    float skin;       // lane 13
    float autoMask;   // lane 14
};

// mirrored field sampling: 2*valid - x - 2 off the valid rect (no edge repeat)
static inline int odl_mirror(int v, uint valid) {
    if (v >= 0 && v < (int)valid) return v;
    return 2 * (int)valid - v - 2;
}
