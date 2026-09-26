// Elementwise tensor operations of the NR graph, selected by MODE.
// v2: one thread per eight consecutive channels of one pixel (16-byte f16 and
// 8-byte E4M3 accesses, packed hardware conversions); every half rounding is
// the bit-level roundF16 or a packed 16-bit conversion, never float(float16_t()).
//
// Port of shaders/vulkan-glsl/ops.comp. MODE selects: f32 -> f16 packing,
// f16 -> E4M3 quantization, the raw-f16 2x2 box pool -> E4M3, the learned post
// blend (2x upsample E4 * scale + adapter E4 * scale -> f16 raw + E4M3), or the
// upsample residual add (f16 projection + E4 skip * scale -> E4M3 [+ f16]).
//
// Vulkan binding i -> [[buffer(i)]]; the push constant block -> constant Push&
// [[buffer(30)]] with identical field order/types (host encodes with setBytes).
// Vulkan constant_id 0 (MODE) -> function_constant(0); the host builds one
// pipeline per MODE key, mirroring the Vulkan spec-constant keying.
//
// local_size_x = 256; the host dispatches a 1D grid, so the GLSL
// gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 65535u * 256u linear-id
// recovery is simply gid.x (spec section 3).

#include "nr_common.metal"

// Vulkan constant_id 0 -> function_constant(0).
struct NrOpsConstants {
  uint MODE [[function_constant(0)]];
};
constant NrOpsConstants kc [[function_constants(NrOpsConstants)]];

constant uint MODE_F32_TO_F16 = 0u;        // input features -> f16 A operand
constant uint MODE_QUANTIZE = 1u;          // f16 -> E4M3
constant uint MODE_DOWNSAMPLE_FP8 = 2u;    // raw f16 2x box pool -> E4M3
constant uint MODE_POST_BLEND = 3u;        // 2x upsample E4 * scale + adapter E4 * scale -> f16 raw + E4M3
constant uint MODE_UPSAMPLE_RESIDUAL = 4u; // f16 projection (low res) + E4 skip * scale -> E4M3 (+ f16 raw)

struct Push {
  uint count;        // output elements (multiple of 8)
  uint channels;     // multiple of 8
  uint inWidth;
  uint inHeight;
  uint outWidth;
  uint outHeight;
  uint auxOffsetA;   // half units in binding 4
  uint auxOffsetB;
  uint dual;         // also publish f16 raw output (binding 6)
};

// Eight scale halves as f32 (aux offsets are not 16-byte aligned in general).
static inline void auxScales(const device ushort* aux16, uint offset, uint channel, thread float* s) {
  for (uint i = 0u; i < 8u; ++i) s[i] = (float)as_type<half>(aux16[offset + channel + i]);
}

// Packed halves -> eight f32 lanes (half -> f32 widening is exact).
static inline void unpack8(uint4 words, thread float* v) {
  for (uint i = 0u; i < 4u; ++i) {
    half2 p = as_type<half2>(words[i]);
    v[2u * i] = (float)p.x;
    v[2u * i + 1u] = (float)p.y;
  }
}

static inline void unpackE4x8(uint2 codes, thread float* v) {
  for (uint i = 0u; i < 8u; ++i) v[i] = e4m3HwToF32((codes[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}

// f32 values already rounded to half -> packed halves (exact) and E4M3 codes.
static inline uint4 packHalves8(thread const float* v) {
  uint4 w;
  for (uint i = 0u; i < 4u; ++i) w[i] = as_type<uint>(half2(v[2u * i], v[2u * i + 1u]));
  return w;
}

// GLSL quantize8: four e4m3HwPair calls packed little-endian; nr_common's
// e4m3Pair carries the same NaN -> +0 guard as the hardware conversion.
static inline uint2 quantize8(uint4 halves) {
  return uint2((uint)e4m3Pair(as_type<half2>(halves.x)) | ((uint)e4m3Pair(as_type<half2>(halves.y)) << 16u),
               (uint)e4m3Pair(as_type<half2>(halves.z)) | ((uint)e4m3Pair(as_type<half2>(halves.w)) << 16u));
}

kernel void nr_ops(
    device const float4* in32x4 [[buffer(0)]],
    device const uint4* in16x8 [[buffer(1)]],
    device const uint2* in8x8 [[buffer(2)]],
    device const uint2* skip8x8 [[buffer(3)]],
    device const ushort* aux16 [[buffer(4)]],
    device uint2* out8x8 [[buffer(5)]],
    device uint4* out16x8 [[buffer(6)]],
    constant Push& pc [[buffer(30)]],
    uint3 gid [[thread_position_in_grid]]) {
  // GLSL: uint group = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 65535u * 256u;
  // (8-channel group; the host dispatches a 1D grid so gid.x is the linear id)
  uint group = gid.x;
  if (group * 8u >= pc.count) return;
  const uint groupsPerPixel = pc.channels / 8u;
  const uint c = (group % groupsPerPixel) * 8u;
  const uint pixel = group / groupsPerPixel;
  if (kc.MODE == MODE_F32_TO_F16) {
    float4 a = in32x4[group * 2u], b = in32x4[group * 2u + 1u];
    out16x8[group] = uint4(as_type<uint>(half2(a.xy)), as_type<uint>(half2(a.zw)), as_type<uint>(half2(b.xy)),
                           as_type<uint>(half2(b.zw)));
  } else if (kc.MODE == MODE_QUANTIZE) {
    out8x8[group] = quantize8(in16x8[group]);
  } else if (kc.MODE == MODE_DOWNSAMPLE_FP8) {
    uint ox = pixel % pc.outWidth, oy = pixel / pc.outWidth;
    uint sx = ox * 2u, sy = oy * 2u;
    uint4 result = uint4(0u);
    if (sx + 1u < pc.inWidth && sy + 1u < pc.inHeight) {
      // The 2x2 pool: three add.f16x2 as (a+b)+(c+d), one mul.f16x2 by
      // 0.25, then cvt.rn.satfinite.e4m3. Every intermediate is a half.
      uint i00 = ((sy * pc.inWidth + sx) * pc.channels + c) / 8u;
      uint i10 = ((sy * pc.inWidth + sx + 1u) * pc.channels + c) / 8u;
      uint i01 = (((sy + 1u) * pc.inWidth + sx) * pc.channels + c) / 8u;
      uint i11 = (((sy + 1u) * pc.inWidth + sx + 1u) * pc.channels + c) / 8u;
      uint4 w00 = in16x8[i00], w10 = in16x8[i10], w01 = in16x8[i01], w11 = in16x8[i11];
      for (uint i = 0u; i < 4u; ++i) {
        half2 top = as_type<half2>(w00[i]) + as_type<half2>(w10[i]);
        half2 bottom = as_type<half2>(w01[i]) + as_type<half2>(w11[i]);
        half2 sum = top + bottom;
        half2 value = sum * (half)0.25h;
        result[i] = as_type<uint>(value);
      }
    }
    out8x8[group] = quantize8(result);
  } else if (kc.MODE == MODE_POST_BLEND) {
    uint ox = pixel % pc.outWidth, oy = pixel / pc.outWidth;
    uint source = ((oy >> 1u) * pc.inWidth + (ox >> 1u)) * pc.channels + c;
    float up[8], adapter[8], scaleA[8], scaleB[8], raw[8];
    unpackE4x8(in8x8[source / 8u], up);
    unpackE4x8(skip8x8[group], adapter);
    auxScales(aux16, pc.auxOffsetA, c, scaleA);
    auxScales(aux16, pc.auxOffsetB, c, scaleB);
    // learned_post_blend: mul.f16x2, mul.f16x2, add.f16x2 with the adapter
    // product contracted into the final add before its half rounding.
    // (The E4M3 * half products are exact in f32, so the f32 spelling of the
    // GLSL reproduces the native half chain bit for bit.)
    for (uint i = 0u; i < 8u; ++i) {
      float inputValue = roundF16(up[i] * scaleA[i]);
      float value = inputValue + adapter[i] * scaleB[i];
      raw[i] = roundF16(value);
    }
    uint4 halves = packHalves8(raw);
    out16x8[group] = halves;
    out8x8[group] = quantize8(halves);
  } else if (kc.MODE == MODE_UPSAMPLE_RESIDUAL) {
    uint ox = pixel % pc.outWidth, oy = pixel / pc.outWidth;
    uint source = ((oy >> 1u) * pc.inWidth + (ox >> 1u)) * pc.channels + c;
    float projected[8], skip[8], scale[8], value[8];
    unpack8(in16x8[source / 8u], projected);
    unpackE4x8(skip8x8[group], skip);
    auxScales(aux16, pc.auxOffsetA, c, scale);
    // residualAfterMatmul + scaleResidual: round_f16(value + residual * scale).
    for (uint i = 0u; i < 8u; ++i) {
      float sum = projected[i] + skip[i] * scale[i];
      value[i] = roundF16(sum);
    }
    uint4 halves = packHalves8(value);
    out8x8[group] = quantize8(halves);
    if (pc.dual != 0u) out16x8[group] = halves;
  }
}
