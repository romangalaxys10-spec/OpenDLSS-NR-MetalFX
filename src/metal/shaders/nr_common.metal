// nr_common.metal — shared numeric foundation of the Metal NR kernels.
// Metal port of shaders/vulkan-glsl/common.glsl + attention_common.glsl helpers.
//
// Every function here reproduces the exact bit-level rounding rules of the network's
// publication points (see docs/original-design/numerics.md and src/common/numeric.h).
// All Metal kernels include this file and use ONLY these helpers for half / E4M3
// conversions, so every kernel publishes identically to the CPU reference and the
// Vulkan (NVIDIA) implementation.
//
// Supported build flags (Metal function constants / -D):
//   NR_EXACT_HALF_CONVERSIONS=0  lets the compiler use its own RNE half conversions
//                                (bit-identical to roundF16 on Apple GPUs; kept switchable
//                                for A/B verification against the portable path).

#ifndef NR_COMMON_METAL
#define NR_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Bit-level f16 <-> f32
// ---------------------------------------------------------------------------

static inline uint nr_f32Bits(float v) { return as_type<uint>(v); }
static inline float nr_f32FromBits(uint b) { return as_type<float>(b); }

// IEEE RNE f32 -> f16 bit pattern (the reference roundF16).
static inline uint nr_f16Bits(float value) {
  uint bits = as_type<uint>(value);
  uint sign = (bits >> 16) & 0x8000u;
  uint exponent = (bits >> 23) & 0xffu;
  uint mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) return sign | (mantissa ? 0x7e00u : 0x7c00u);
  int halfExponent = (int)exponent - 112;
  if (halfExponent >= 31) return sign | 0x7c00u;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    uint normalized = mantissa | 0x800000u;
    uint shift = (uint)(14 - halfExponent);
    uint rounded = normalized >> shift;
    uint remainder = normalized & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) rounded += 1u;
    return sign | rounded;
  }
  uint rounded = mantissa >> 13;
  uint remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
    rounded += 1u;
    if (rounded == 0x400u) { rounded = 0u; halfExponent += 1; }
  }
  if (halfExponent >= 31) return sign | 0x7c00u;
  return sign | ((uint)halfExponent << 10) | rounded;
}

static inline float nr_f16ToF32(uint bits) {
  uint sign = (bits & 0x8000u) << 16;
  uint exponent = (bits >> 10) & 0x1fu;
  uint mantissa = bits & 0x3ffu;
  if (exponent == 0u) {
    if (mantissa == 0u) return as_type<float>(sign);
    int shift = 0;
    while (!(mantissa & 0x400u)) { mantissa <<= 1; shift += 1; }
    mantissa &= 0x3ffu;
    return as_type<float>(sign | ((uint)(113 - shift) << 23) | (mantissa << 13));
  }
  if (exponent == 31u) return as_type<float>(sign | 0x7f800000u | (mantissa << 13));
  return as_type<float>(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

// f32 -> RNE f16 -> f32, on the bit pattern (common.glsl roundF16).
static inline float roundF16(float value) {
  uint bits = as_type<uint>(value);
  uint sign = bits & 0x80000000u;
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) return value;                                  // inf / nan
  if (magnitude >= 0x477ff000u) return as_type<float>(sign | 0x7f800000u);     // >= 65520 overflows to inf
  if (magnitude < 0x38800000u) {                                               // half subnormal grid 2^-24
    float scaled = rint(as_type<float>(magnitude) * 16777216.0f);
    return as_type<float>(sign) + as_type<float>(sign | as_type<uint>(scaled * 0.000000059604644775390625f));
  }
  uint lsb = (magnitude >> 13) & 1u;
  uint rounded = (magnitude + 0xfffu + lsb) & ~0x1fffu;
  return as_type<float>(sign | rounded);
}

static inline half nr_f16FromBits(uint bits) { return as_type<half>((ushort)(bits & 0xffffu)); }
static inline uint nr_halfBits(half value) { return (uint)as_type<ushort>(value); }

// ---------------------------------------------------------------------------
// E4M3FN (FP8) publication
// ---------------------------------------------------------------------------

static inline uint nr_shiftRne(uint value, uint shift) {
  uint quotient = value >> shift;
  uint remainder = value & ((1u << shift) - 1u);
  uint halfway = 1u << (shift - 1u);
  return quotient + ((remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u)) ? 1u : 0u);
}

// An already half-rounded value to its E4M3FN code: RNE, finite saturation at 448, NaN -> +0.
// The sign of a zero survives (code 0x80), as the hardware conversion does.
static inline uint e4m3CodeFromF16Bits(uint halfBits) {
  bool isNan = (halfBits & 0x7c00u) == 0x7c00u && (halfBits & 0x03ffu) != 0u;
  if (isNan) return 0u;
  if ((halfBits & 0x7fffu) == 0u) return (halfBits >> 8u) & 0x80u;
  uint negative = (halfBits & 0x8000u) != 0u ? 0x80u : 0u;
  uint exponent = (halfBits >> 10) & 0x1fu;
  uint mantissa = halfBits & 0x03ffu;
  uint code;
  if (exponent == 31u) {
    code = 0x7eu;                     // inf saturates
  } else if (exponent <= 8u) {        // E4M3 subnormal
    uint significand = exponent == 0u ? mantissa : 1024u + mantissa;
    uint shift = exponent == 0u ? 15u : 16u - exponent;
    code = min(nr_shiftRne(significand, shift), 8u);
  } else {
    uint e4Exponent = exponent - 8u;
    uint e4Mantissa = nr_shiftRne(mantissa, 7u);
    if (e4Mantissa == 8u) { e4Mantissa = 0u; e4Exponent += 1u; }
    code = (e4Exponent > 15u || (e4Exponent == 15u && e4Mantissa > 6u)) ? 0x7eu : ((e4Exponent << 3u) | e4Mantissa);
  }
  return negative | code;
}

static inline uint e4m3CodeFromF32(float value) { return e4m3CodeFromF16Bits(nr_f16Bits(value)); }
static inline uchar e4m3Code(half value) { return (uchar)e4m3CodeFromF16Bits(nr_halfBits(value)); }

// E4M3FN byte -> float (exact; NaN code 0x7f/0xff decodes to NaN).
static inline float e4m3ToF32(uint bits) {
  bool negative = (bits & 0x80u) != 0u;
  uint exponent = (bits >> 3u) & 0x0fu;
  uint mantissa = bits & 0x07u;
  float value;
  if (exponent == 0u) value = (float)mantissa * 0.001953125f;
  else if (exponent == 15u && mantissa == 7u) value = 0.0f;   // NaN code -> caller decides (decode path uses NaN)
  else value = (1.0f + (float)mantissa * 0.125f) * as_type<float>((exponent + 120u) << 23);
  return negative ? -value : value;
}

// Decode where NaN codes read back as 0, as in the reference and the Vulkan kernels.
static inline float e4m3HwToF32(uint code) {
  float value = ((code & 0x7fu) == 0x7fu) ? 0.0f : e4m3ToF32(code);
  return value;
}

// E4M3 byte -> half (exact: every E4M3 value is representable in f16).
static inline half e4m3ToHalf(uint code) {
  if ((code & 0x7fu) == 0x7fu) return (half)0.0;   // NaN -> 0 like the reference
  return nr_f16FromBits(nr_f16Bits(e4m3ToF32(code)));
}

// Two halves -> two E4M3 codes packed little-endian (NaN of either -> both-safe per-half guard).
static inline ushort e4m3Pair(half2 value) {
  half2 clean = select(value, half2(0.0h), isnan(value));
  ushort lo = (ushort)e4m3CodeFromF16Bits(nr_halfBits(clean.x));
  ushort hi = (ushort)e4m3CodeFromF16Bits(nr_halfBits(clean.y));
  return (ushort)(lo | (hi << 8));
}

static inline uint e4x4(half2 a, half2 b) {
  return (uint)e4m3Pair(a) | ((uint)e4m3Pair(b) << 16);
}

// ---------------------------------------------------------------------------
// The network's cubic SiLU (half FMAs; identical on all 65536 half inputs to the
// f32 reference mpCubicSilu, verified by tests/test_reference)
// ---------------------------------------------------------------------------

static inline half siluHalf(half value) {
  half bounded = clamp(value, (half)-4.0h, (half)4.0h);
  half inner = fma((half)-0.055908203125h, abs(bounded), (half)0.447265625h);
  half polynomial = fma(bounded, inner, (half)0.89453125h);
  half result = value * polynomial;
  return result;
}

static inline half2 siluPair(half2 value) {
  half2 bounded = clamp(value, half2(-4.0h), half2(4.0h));
  half2 inner = fma(half2(-0.055908203125h), abs(bounded), half2(0.447265625h));
  half2 polynomial = fma(bounded, inner, half2(0.89453125h));
  return value * polynomial;
}

// ---------------------------------------------------------------------------
// Window / global attention helpers
// ---------------------------------------------------------------------------

// natural token order -> physical (4x4 tiles of 16) order
static inline uint tiledToken(uint token) {
  uint x = token & 7u, y = token >> 3u;
  return ((y >> 2u) * 32u) + ((x >> 2u) * 16u) + ((y & 3u) * 4u) + (x & 3u);
}

static inline uint inverseTiledToken(uint token) {
  uint tile = token >> 4u, within = token & 15u;
  uint x = ((tile & 1u) * 4u) + (within & 3u);
  uint y = ((tile >> 1u) * 4u) + (within >> 2u);
  return (y * 8u) + x;
}

// The network's exponential approximation: f32 affine, half clamp, then the
// exponent-field remap (bits << 5) + 0x8000. Scalar-exact form (the packed SIMD
// variant compilers generate from this is bit-identical because every step is a
// half publication).
static inline half expWeight(half score) {
  float affine32 = fma((float)score, 0.044921875f, 1.30078125f);
  half affine = clamp((half)affine32, (half)1.03125h, (half)1.5693359375h);
  uint bits = nr_halfBits(affine);
  uint exponentialBits = ((bits << 5u) + 0x8000u) & 0xffffu;
  return nr_f16FromBits(exponentialBits);
}

// Global-ViT variant (different constants and shift 4).
static inline half vitExpWeight(half score) {
  half affine = fma(score, (half)0.08953946828842163, (half)1.7093614339828491);
  affine = clamp(affine, (half)1.439453125h, (half)1.9775390625h);
  uint bits = nr_halfBits(affine);
  uint exponentialBits = ((bits << 4u) + 0x4000u) & 0xffffu;
  return nr_f16FromBits(exponentialBits);
}

// Cosine normalization of one 32-channel vector held as 16 packed pairs (channels 2i, 2i+1):
// r[c] = fma(v[c], v[c], f16(v[c+16]^2)) for c < 16, tree stride 8/4/2/1, every level
// published as half exactly like the reference reduction array.
static inline half inverseNorm16(thread half2* p) {
  half2 r[8];
  for (uint m = 0u; m < 8u; ++m) {
    half2 highSquare = p[m + 8u] * p[m + 8u];
    r[m] = fma(p[m], p[m], highSquare);
  }
  half2 s8[4];
  for (uint m = 0u; m < 4u; ++m) s8[m] = r[m] + r[m + 4u];
  half2 s4a = s8[0] + s8[2];
  half2 s4b = s8[1] + s8[3];
  half2 s2 = s4a + s4b;
  half s1 = s2.x + s2.y;
  return (half)(1.0 / rsqrt((float)s1));
}

// round_f16(residual * scale) for eight residual/scale pairs (exact products in f32, one rounding).
static inline half2 scaledPair(half2 residual, half2 scale) {
  float2 product = (float2)residual * (float2)scale;
  return as_type<half2>(half2(product.x, product.y));
}

static inline uint packHalves(float2 v) { return as_type<uint>(half2(v.x, v.y)); }

// ---------------------------------------------------------------------------
// Buffer views: Metal kernels read/write through raw device pointers carried in
// MTLBuffer addresses bound as constants, matching the Vulkan 12-binding layout.
// ---------------------------------------------------------------------------

static inline uint2 loadE4Pair(const device uchar* bytes, uint index) {
  uint packed = (uint)bytes[index] | ((uint)bytes[index + 1u] << 8);
  return uint2(packed & 0xffu, packed >> 8u);
}

#endif  // NR_COMMON_METAL
