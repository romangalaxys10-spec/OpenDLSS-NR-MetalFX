// window_attend.metal — fused 8x8 window cosine attention (Metal port of
// shaders/vulkan-glsl/window_attend.comp, unfused reference route).
//
// One workgroup = one (window, head). Four simdgroups.
//   S = Q K^T + prior          16 MMA tiles (16x16x32, C = learned prior tile)
//   w = expWeight(S)          per element, FP16 bit approximation
//   p = e4(w * f16(1/sum))    the specified softmax order, E4M3 weights
//   O = p V                    8 MMA tiles (16x16x32, two k32 steps over 64 keys)
// Keys are kept in physical (tiled) order p = tile*16 + within so the prior,
// the softmax tree and the value K dimension all index the same way.
//
// The NV cooperative matrices map to 2x2 grids of Apple 8x8 simdgroup matrices
// per 16x16 tile; every k32 step is two k16 planes in the same K order. E4M3
// operands are dequantized (exactly) when staged into threadgroup memory.
// All rounding goes through nr_common.metal (bit-identical publications).

#include "nr_common.metal"

struct WindowAttendPush {
  uint width;
  uint height;
  uint channels;   // heads * 32
  uint heads;
  uint shiftX;     // window origin offset (0 or 4)
  uint shiftY;
  uint windowsX;   // ceil((width + shiftX) / 8)
  uint windowCount;
};

kernel void nr_window_attend(
    device const uchar*   qkv       [[buffer(0)]],   // E4M3 [tokens][channels*3]
    device const half*    prior     [[buffer(1)]],   // [heads][64 query][64 physical key]
    device uchar*         attended  [[buffer(5)]],   // E4M3 [tokens][channels]
    constant WindowAttendPush& pc   [[buffer(30)]],
    uint3 grp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint thread = tid;
  const uint subgroup = sg;
  const uint head = grp.x;
  const uint window = grp.y;
  if (window >= pc.windowCount) return;
  const int windowX = int((window % pc.windowsX) * 8u) - int(pc.shiftX);
  const int windowY = int((window / pc.windowsX) * 8u) - int(pc.shiftY);
  const uint stride3 = pc.channels * 3u;
  const uint headBase = head * 96u;

  threadgroup uchar qs[64 * 32];      // [query][component]
  threadgroup uchar ks[64 * 32];      // [physical key][component]
  threadgroup uchar vs[64 * 32];      // [physical key][component]
  threadgroup half priors[64 * 64];
  threadgroup half scores[64 * 64];   // [query][physical key]
  threadgroup uchar weights[64 * 64]; // E4M3 [query][physical key]
  threadgroup half outputs[64 * 32];
  // Dequant-staged MMA operands (exact conversions; reused per tile).
  threadgroup half qh[16 * 40];   // [16 q][32 k] per qTile
  threadgroup half kh[16 * 40];   // [16 n][32 k] per kTile (n-major -> transposed B loads)
  threadgroup half ph[16 * 72];   // [16 q][64 k] per qTile (P weights)
  threadgroup half vh[32 * 24];   // [32 k][16 c] per (kb, cTile) (k-major)

  // Stage Q (query order), K/V (physical key order) and the prior.
  for (uint i = thread; i < 64u * 32u; i += 128u) {
    uint token = i >> 5u, component = i & 31u;
    int qx = windowX + int(token & 7u), qy = windowY + int(token >> 3u);
    uchar q = (uchar)0;
    if (qx >= 0 && qy >= 0 && qx < int(pc.width) && qy < int(pc.height)) {
      q = qkv[(uint(qy) * pc.width + uint(qx)) * stride3 + headBase + component];
    }
    qs[i] = q;
    uint keyLocal = inverseTiledToken(token);
    int kx = windowX + int(keyLocal & 7u), ky = windowY + int(keyLocal >> 3u);
    uchar k = (uchar)0, v = (uchar)0;
    if (kx >= 0 && ky >= 0 && kx < int(pc.width) && ky < int(pc.height)) {
      uint keyBase = (uint(ky) * pc.width + uint(kx)) * stride3 + headBase;
      k = qkv[keyBase + 32u + component];
      v = qkv[keyBase + 64u + component];
    }
    ks[i] = k;
    vs[i] = v;
  }
  for (uint i = thread; i < 64u * 64u; i += 128u) priors[i] = prior[head * 4096u + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // S = Q K^T with the learned prior as the incoming FP16 accumulator.
  // 16 tiles (4x4 of 16x16) over 4 simdgroups; each tile: A = Q [16][32], B = K^T (n-major staged).
  for (uint tile = subgroup; tile < 16u; tile += 4u) {
    uint qTile = tile >> 2u, kTile = tile & 3u;
    for (uint idx = lane; idx < 16u * 32u; idx += 32u) {
      uint r = idx >> 5, c = idx & 31u;
      qh[r * 40u + c] = e4m3ToHalf(qs[qTile * 16u * 32u + idx]);
      kh[r * 40u + c] = e4m3ToHalf(ks[kTile * 16u * 32u + idx]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float16_t, 8, 8> mc[2][2];
    for (uint p = 0u; p < 2u; ++p) {           // k16 planes within k32
      for (uint r = 0u; r < 2u; ++r) {
        auto a = simdgroup_load(qh, (r * 8u) * 40u + p * 8u, (ushort)40u);
        for (uint c = 0u; c < 2u; ++c) {
          auto b = simdgroup_load(kh, (c * 8u) * 40u + p * 8u, (ushort)40u, true);   // [32k][16n] via transpose
          auto priorTile = simdgroup_load(priors, qTile * 16u * 64u + kTile * 16u + (r * 8u) * 64u + c * 8u,
                                          (ushort)64u);
          if (p == 0u) mc[r][c] = simdgroup_multiply_accumulate(a, b, priorTile);
          else mc[r][c] = simdgroup_multiply_accumulate(a, b, mc[r][c]);
        }
      }
    }
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        simdgroup_store(mc[r][c], scores, qTile * 16u * 64u + kTile * 16u + (r * 8u) * 64u + c * 8u, (ushort)64u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Exponential weights, then the specified reduction, reciprocal and E4M3 publication.
  for (uint i = thread; i < 64u * 64u; i += 128u) scores[i] = expWeight(scores[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread < 64u) {
    uint row = thread;
    // the specified softmax reduction (keys in physical order): eight 8-key fragments reduced
    // per lane pair, then the four column pairs, then even/odd halves.
    auto softmaxPair = [&](uint pair, uint parity) -> half {
      uint key = pair * 2u + parity;
      half blocks01 = scores[row * 64u + key] + scores[row * 64u + key + 8u];
      half blocks23 = scores[row * 64u + key + 16u] + scores[row * 64u + key + 24u];
      half blocks45 = scores[row * 64u + key + 32u] + scores[row * 64u + key + 40u];
      half blocks67 = scores[row * 64u + key + 48u] + scores[row * 64u + key + 56u];
      half a = blocks01 + blocks23;
      half b = a + blocks45;
      return b + blocks67;
    };
    half e0 = softmaxPair(0u, 0u) + softmaxPair(1u, 0u);
    half e1 = e0 + softmaxPair(2u, 0u);
    half even = e1 + softmaxPair(3u, 0u);
    half o0 = softmaxPair(0u, 1u) + softmaxPair(1u, 1u);
    half o1 = o0 + softmaxPair(2u, 1u);
    half odd = o1 + softmaxPair(3u, 1u);
    half total = even + odd;
    half reciprocal = (half)(1.0 / float(total));
    for (uint key = 0u; key < 64u; ++key) {
      half normalized = scores[row * 64u + key] * reciprocal;
      weights[row * 64u + key] = (uchar)e4m3CodeFromF16Bits(nr_halfBits(normalized));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // O = P V over the 64 physical keys as two k32 steps (four F13 groups in tile
  // order 0,1,2,3), FP16 accumulation from zero.
  for (uint tile = subgroup; tile < 8u; tile += 4u) {
    uint qTile = tile >> 1u, cTile = tile & 1u;
    for (uint idx = lane; idx < 16u * 64u; idx += 32u) {
      uint r = idx >> 6, c = idx & 63u;
      ph[r * 72u + c] = e4m3ToHalf(weights[qTile * 16u * 64u + idx]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float16_t, 8, 8> mc[2][2];
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        for (ushort i = 0; i < 8; ++i)
          for (ushort j = 0; j < 8; ++j) mc[r][c][i][j] = 0.0h;
    for (uint kb = 0u; kb < 64u; kb += 32u) {
      // Stage V chunk [32k][16c] for this k32 step (k-major).
      for (uint idx = lane; idx < 32u * 16u; idx += 32u) {
        uint k = idx >> 4, c = idx & 15u;
        vh[k * 24u + c] = e4m3ToHalf(vs[(cTile * 16u + c) * 32u + kb + k]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint p = 0u; p < 2u; ++p) {
        for (uint r = 0u; r < 2u; ++r) {
          auto a = simdgroup_load(ph, (r * 8u) * 72u + kb + p * 8u, (ushort)72u);
          for (uint c = 0u; c < 2u; ++c) {
            auto b = simdgroup_load(vh, (p * 8u) * 24u + c * 8u, (ushort)24u);
            mc[r][c] = simdgroup_multiply_accumulate(a, b, mc[r][c]);
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        simdgroup_store(mc[r][c], outputs, qTile * 16u * 32u + cTile * 16u + (r * 8u) * 32u + c * 8u, (ushort)32u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = thread; i < 64u * 32u; i += 128u) {
    uint token = i >> 5u, component = i & 31u;
    int qx = windowX + int(token & 7u), qy = windowY + int(token >> 3u);
    if (qx < 0 || qy < 0 || qx >= int(pc.width) || qy >= int(pc.height)) continue;
    uint index = (uint(qy) * pc.width + uint(qx)) * pc.channels + head * 32u + component;
    attended[index] = (uchar)e4m3CodeFromF16Bits(nr_halfBits(outputs[i]));
  }
}
