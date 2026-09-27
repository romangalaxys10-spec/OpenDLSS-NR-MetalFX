// Common.hlsli — shared helpers for every OpenDLSS-NR D3D12 kernel.
//
// The bit-tricks here MUST stay identical to core/include/opendlss/fp16.h and
// e4m3.h (the CPU reference) — they are checked by the parity harness. Every
// function is a line-by-line port of the C++ reference:
//
//   odl_f16_bits_to_f32()  <- opendlss::f16_to_f32   (exact; also available as
//                             the f16tof32 intrinsic, which is exact by spec,
//                             but the bit math keeps one code path for every
//                             profile)
//   odl_f32_to_f16_bits()  <- opendlss::f32_to_f16   (RTNE with subnormals;
//                             implemented in bit math because the rounding
//                             mode of the legacy f32tof16 intrinsic is not
//                             guaranteed identical on every driver)
//   odl_e4m3_encode/decode <- opendlss::E4M3::encode/decode
//   odl_pub_e4m3           <- opendlss::e4m3_publish
//   odl_window_exp         <- opendlss::window_exp_trick
//   odl_global_exp         <- opendlss::global_exp_trick
//   odl_hash               <- opendlss::preprocess_pixel_hash (uint64 math is
//                             emulated with uint2 pairs so the kernel runs on
//                             every profile d3dcompiler can emit)
//
// Token tensors are stored as IEEE-754 binary16 patterns packed two per
// 32-bit word (little-endian: lane 2*i in bits 0..15, lane 2*i+1 in bits
// 16..31) inside ByteAddressBuffer / RWByteAddressBuffer resources. All
// kernel writes go through OdlStoreHalf2 so each thread owns whole dwords —
// this network's channel counts are always even (32..1024), which keeps
// every per-token row dword aligned.
//
// Shader model: written for SM 6.x (DXC, cs_6_0) and kept compatible with
// the d3dcompiler cs_5_0 profile used for runtime compilation — no 16-bit
// scalar types, no wave intrinsics, no dynamically-sized groupshared.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.

#ifndef ODL_COMMON_HLSLI
#define ODL_COMMON_HLSLI

// ---------------------------------------------------------------------------
// f32 <-> f16 (binary16), bit-exact ports of core/include/opendlss/fp16.h
// ---------------------------------------------------------------------------

// Decode binary16 -> float. Exact (half values are a subset of float).
float OdlF16BitsToFloat(uint h)
{
    uint sign = (h & 0x8000u) << 16;
    uint exp  = (h >> 10) & 0x1Fu;
    uint man  = h & 0x03FFu;
    uint bits;
    if (exp == 0)
    {
        if (man == 0)
        {
            bits = sign;
        }
        else
        {
            // subnormal -> normalize
            int e = -1;
            uint m = man;
            [loop] do { ++e; m <<= 1; } while ((m & 0x0400u) == 0);
            m &= 0x03FFu;
            bits = sign | (uint(127 - 15 - e + 1) << 23) | (m << 13);
        }
    }
    else if (exp == 31)
    {
        bits = sign | 0x7F800000u | (man << 13);
    }
    else
    {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    return asfloat(bits);
}

// Round-to-nearest-even f32 -> binary16 pattern. Port of f32_to_f16().
uint OdlFloatToF16Bits(float value)
{
    uint bits = asuint(value);
    uint sign = (bits >> 16) & 0x8000u;
    int  exp  = int((bits >> 23) & 0xFFu) - 127;
    uint man  = bits & 0x007FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu)
    {                       // Inf / NaN
        if (man != 0u)
        {
            uint m16 = man >> 13;
            if (m16 == 0u) m16 = 1u;
            return sign | 0x7C00u | m16;
        }
        return sign | 0x7C00u;
    }
    if (exp > 15) return sign | 0x7C00u;                    // overflow -> Inf
    if (exp >= -14)
    {                     // normal
        uint man16 = man >> 13;
        uint round = (man >> 12) & 1u;
        man16 += round;
        if (man16 >> 10 != 0u) { man16 = 0u; ++exp; }       // mantissa overflow
        if (exp > 15) return sign | 0x7C00u;
        return sign | (uint(exp + 15) << 10) | man16;
    }
    if (exp < -25) return sign;                             // underflow to zero
    // subnormal: shift with rounding
    uint shift   = uint(-14 - exp);                         // 1..24
    uint manFull = man | 0x00800000u;                       // implicit leading 1
    uint man16   = manFull >> (13 + shift);
    uint rem     = manFull << (19 - shift);                 // bits dropped, MSB aligned
    uint halfUlp = 1u << 31;
    if (rem > halfUlp || (rem == halfUlp && (man16 & 1u) != 0u)) ++man16;
    return sign | man16;
}

// round-trip through the half grid (the "half(...)" casts of the reference)
float OdlHalf(float v) { return OdlF16BitsToFloat(OdlFloatToF16Bits(v)); }

// ---------------------------------------------------------------------------
// E4M3 (OCP FP8) codec — port of core/include/opendlss/e4m3.h.
// 1 sign | 4 exponent (bias 7) | 3 mantissa; max normal 448; no infinities;
// S.1111.111 (0x7F/0xFF) is NaN; saturating round-to-nearest-even encode.
// ---------------------------------------------------------------------------

float OdlE4m3Decode(uint code)
{
    uint sign = (code & 0x80u) != 0u ? 0x80000000u : 0u;
    uint e    = (code >> 3) & 0x0Fu;
    uint m    = code & 0x07u;
    if (e == 15u && m == 7u) return asfloat(0x7FC00000u);   // quiet NaN
    float v;
    if (e == 0u)
    {
        if (m == 0u) v = 0.0f;
        else         v = exp2(-9.0f) * float(m);            // m * 2^-9 subnormal
    }
    else
    {
        v = exp2(float(int(e) - 7)) * (1.0f + float(m) / 8.0f);
    }
    return sign != 0u ? -v : v;
}

uint OdlE4m3Encode(float value)
{
    uint fbits = asuint(value);
    bool nan   = (fbits & 0x7F800000u) == 0x7F800000u && (fbits & 0x007FFFFFu) != 0u;
    if (nan) return 0x7Fu;
    bool inf = (fbits & 0x7FFFFFFFu) == 0x7F800000u;
    uint sign = (fbits >> 31) << 7;                          // 0x80
    if (inf) return sign | 0x7Eu;                            // saturate to max normal
    int  exp = int((fbits >> 23) & 0xFFu) - 127;
    uint man = fbits & 0x007FFFFFu;

    if (((fbits >> 23) & 0xFFu) == 0u)
    {                 // f32 zero / subnormal
        if (value == 0.0f) return sign;
        float av = abs(value);
        if (av < 0.5f * exp2(-9.0f)) return sign;            // below min subnormal
    }

    if (exp < -6)
    {                                        // subnormal in E4M3
        if (exp < -30) return sign;
        float av  = abs(value);
        float m_f = av * 512.0f;                             // m = |value| * 2^9
        uint  m   = uint(m_f);
        float frac = m_f - float(m);
        if (frac > 0.5f || (frac == 0.5f && (m & 1u) != 0u)) ++m;
        if (m >= 8u) return sign | (1u << 3);                // rounds up to min normal 2^-6
        return sign | m;
    }
    if (exp > 8 || (exp == 8 && man > 0x00400000u))
    {          // > 448 -> saturate
        return sign | 0x7Eu;
    }
    uint man3 = man >> 20;
    uint rem  = man & 0x000FFFFFu;
    uint half = 1u << 19;
    if (rem > half || (rem == half && (man3 & 1u) != 0u))
    {
        ++man3;
        if (man3 == 8u) { man3 = 0u; ++exp; if (exp > 8) return sign | 0x7Eu; }
    }
    return sign | (uint(exp + 7) << 3) | man3;
}

// publication = quantize onto the E4M3 grid, keep f32
float OdlPubE4m3(float v) { return OdlE4m3Decode(OdlE4m3Encode(v)); }

// ---------------------------------------------------------------------------
// attention exponent bit-tricks — ports of fp16.h window_exp_trick /
// global_exp_trick. Each round-trips through the half grid exactly as the
// C++ reference does.
// ---------------------------------------------------------------------------

// window blocks: x = clamp(half(0.044921875 s + 1.30078125), 1.03125, 1.5693359375)
//                exp = reinterpret_half((bits(x) << 5) ^ 0x8000) == 2^(1.4375 s - 5.375)
float OdlWindowExp(float s)
{
    float x  = OdlHalf(0.044921875f * s + 1.30078125f);
    float xf = clamp(x, 1.03125f, 1.5693359375f);
    uint  b  = (OdlFloatToF16Bits(xf) << 5) & 0xFFFFu;
    b ^= 0x8000u;
    return OdlF16BitsToFloat(b);
}

// global ViT blocks: x = clamp(half(0.08953947 s + 1.70936143), 1.439453125, 1.9775390625)
//                    exp = reinterpret_half((bits(x) << 4) + 0x4000) == 2^(1.4326 s - 3.6502)
float OdlGlobalExp(float s)
{
    float x  = OdlHalf(0.08953947f * s + 1.70936143f);
    float xf = clamp(x, 1.439453125f, 1.9775390625f);
    uint  b  = (OdlFloatToF16Bits(xf) << 4) & 0xFFFFu;
    b += 0x4000u;                                            // wrapping uint add
    return OdlF16BitsToFloat(b);
}

inline float OdlSigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// SiLU (swish) used by every FFN: v / (1 + exp(-v)), f32 (as in the reference)
inline float OdlSilu(float v) { return v / (1.0f + exp(-v)); }

// ---------------------------------------------------------------------------
// pixel hash — port of core/src/pipeline.cpp preprocess_pixel_hash.
// 64-bit ops are emulated with uint2 (x = lo, y = hi) so the kernels stay
// compilable by d3dcompiler (cs_5_0) as well as DXC (cs_6_x).
// ---------------------------------------------------------------------------

uint2 OdlU64Mul(uint2 a, uint2 b)                 // low 64 bits of a*b
{
    uint lo = a.x * b.x;
    uint hi = a.x * b.y + a.y * b.x + mul_hi(a.x, b.x);
    return uint2(lo, hi);
}

uint2 OdlU64Xor(uint2 a, uint2 b) { return uint2(a.x ^ b.x, a.y ^ b.y); }

uint2 OdlU64Shr(uint2 v, uint s)                  // logical shift right
{
    if (s == 0u) return v;
    if (s >= 64u) return uint2(0u, 0u);
    if (s >= 32u) return uint2(v.y >> (s - 32u), 0u);
    return uint2((v.x >> s) | (v.y << (32u - s)), v.y >> s);
}

// seed is passed as (lo, hi) root constants; frameIndex folds like the CPU.
uint OdlHash(uint x, uint y, uint frameIndex, uint2 seed)
{
    uint2 h = OdlU64Xor(seed, OdlU64Mul(uint2(x, 0u), uint2(0x7F4A7C15u, 0x9E3779B9u)));
    h = OdlU64Xor(h, OdlU64Mul(uint2(y, 0u),         uint2(0x27D4EB4Fu, 0xC2B2AE3Du)));
    h = OdlU64Xor(h, OdlU64Mul(uint2(frameIndex, 0u), uint2(0x9E3779F9u, 0x165667B1u)));
    h = OdlU64Xor(h, OdlU64Shr(h, 33u)); h = OdlU64Mul(h, uint2(0xED558CCDu, 0xFF51AFD7u));
    h = OdlU64Xor(h, OdlU64Shr(h, 33u)); h = OdlU64Mul(h, uint2(0x1A85EC53u, 0xC4CEB9FEu));
    h = OdlU64Xor(h, OdlU64Shr(h, 33u));
    return h.x ^ h.y;                                          // (uint)(h ^ (h >> 32))
}

// mirrored field sampling: 2*valid - x - 2 off the valid rect (no edge repeat)
int OdlMirror(int v, uint valid)
{
    if (v >= 0 && v < int(valid)) return v;
    return 2 * int(valid) - v - 2;
}

// ---------------------------------------------------------------------------
// packed-f16 byte-address-buffer access.
// half lane i lives in dword i>>1, bits 0..15 when i is even, 16..31 when odd.
// Loads may read any lane; stores MUST be whole dwords (OdlStoreHalf2) so
// threads never race on a shared word — all of this network's row strides
// (channels) are even, which keeps per-token rows dword aligned.
// ---------------------------------------------------------------------------

float OdlLoadHalf(ByteAddressBuffer buf, uint index)
{
    uint dword = buf.Load((index >> 1) << 2);
    return OdlF16BitsToFloat((index & 1u) != 0u ? (dword >> 16) : (dword & 0xFFFFu));
}

// reads both lanes of the dword containing `index` (index must be even)
float2 OdlLoadHalf2(ByteAddressBuffer buf, uint index2)
{
    uint dword = buf.Load((index2 >> 1) << 2);
    return float2(OdlF16BitsToFloat(dword & 0xFFFFu),
                  OdlF16BitsToFloat(dword >> 16));
}

uint OdlPackHalf2(float a, float b)
{
    return OdlFloatToF16Bits(a) | (OdlFloatToF16Bits(b) << 16);
}

// writes both lanes of dword `dwordIndex` (i.e. half lanes 2*dwordIndex, +1)
void OdlStoreHalf2(RWByteAddressBuffer buf, uint dwordIndex, float a, float b)
{
    buf.Store(dwordIndex << 2, OdlPackHalf2(a, b));
}

uint2 OdlLoadU64(ByteAddressBuffer buf, uint dwordIndex)
{
    return buf.Load2(dwordIndex << 2);
}

#endif // ODL_COMMON_HLSLI
