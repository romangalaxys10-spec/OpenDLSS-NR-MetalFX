// global_attend.metal — global ViT cosine attention (Metal port of
// shaders/vulkan-glsl/global_attend.comp, unfused reference route).
//
// One simdgroup per (head, 16-query tile). Keys run over the 64-padded token
// count in natural order. Padding keys have zero K/V, so they add nothing to
// the numerator, but their score is still 0 and still exponentiates: the
// denominator therefore carries padding * exp(0), which is subtracted once at
// the end. Dropping them from the sum instead is a different function, not an
// optimization. Unlike the window kernel, the E4M3 weights are unnormalized
// and the reciprocal is applied to the value accumulator afterwards.
//
// Metal: the same 16x16x32 MMA chain through 2x2 grids of 8x8 simdgroup
// matrices; E4M3 operands dequantized (exactly) into threadgroup tiles.
// Threadgroup buffers are sized for PADDED_TOKENS <= 256 (the host guarantees
// this for the single-kernel route; larger token counts use the streamed
// normalize + attend pair).

#include "nr_common.metal"

struct GlobalAttendConstants {
  uint PADDED_TOKENS;   // constant_id 0
};
constant GlobalAttendConstants kc [[function_constants(GlobalAttendConstants)]];

struct GlobalAttendPush {
  uint tokens;
  uint heads;
  uint channels;
};

kernel void nr_global_attend(
    device const uchar*   qkv       [[buffer(0)]],   // E4M3 [paddedTokens][channels*3]
    device uchar*         attended  [[buffer(5)]],   // E4M3 [tokens][channels]
    constant GlobalAttendPush& pc   [[buffer(30)]],
    uint3 grp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (kc.PADDED_TOKENS > 256u) return;   // host contract: single-kernel route caps at 256
  const uint head = grp.x;
  const uint qTile = grp.y;
  const uint stride3 = pc.channels * 3u;
  const uint headBase = head * 96u;
  const uint qBase = qTile * 16u * stride3 + headBase;
  const uint PT = kc.PADDED_TOKENS;

  threadgroup half scores[16 * 256];
  threadgroup uchar weights[16 * 256];
  threadgroup half outputs[16 * 32];
  threadgroup half reciprocals[16];
  threadgroup half stageA[16 * 40];   // [16 rows][32 k] dequant-staged operands
  threadgroup half stageB[32 * 24];   // [32 k][16 n]

  // Stage Q tile [16][32] (rows beyond tokens read the padding rows, which are
  // zeroed by the host's allocation semantics — the GLSL relies on the same).
  for (uint idx = lane; idx < 16u * 32u; idx += 32u) {
    uint r = idx >> 5, c = idx & 31u;
    stageA[r * 40u + c] = e4m3ToHalf(qkv[(size_t)(qTile * 16u + r) * stride3 + headBase + c]);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  // S = Q K^T over every 16-key tile: zero accumulator, one k32 MMA each.
  for (uint kTile = 0u; kTile < PT / 16u; ++kTile) {
    for (uint idx = lane; idx < 16u * 32u; idx += 32u) {
      uint r = idx >> 5, c = idx & 31u;              // r = token within tile, c = channel
      stageB[c * 24u + r] = e4m3ToHalf(qkv[(size_t)(kTile * 16u + r) * stride3 + headBase + 32u + c]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float16_t, 8, 8> mc[2][2];
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        for (ushort i = 0; i < 8; ++i)
          for (ushort j = 0; j < 8; ++j) mc[r][c][i][j] = 0.0h;
    for (uint p = 0u; p < 2u; ++p) {
      for (uint r = 0u; r < 2u; ++r) {
        auto a = simdgroup_load(stageA, (r * 8u) * 40u + p * 8u, (ushort)40u);
        for (uint c = 0u; c < 2u; ++c) {
          auto b = simdgroup_load(stageB, (p * 8u) * 24u + c * 8u, (ushort)24u);
          mc[r][c] = simdgroup_multiply_accumulate(a, b, mc[r][c]);
        }
      }
    }
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        simdgroup_store(mc[r][c], scores, kTile * 16u + (r * 8u) * PT + c * 8u, (ushort)PT);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = lane; i < 16u * PT; i += 32u) scores[i] = vitExpWeight(scores[i]);
  simdgroup_barrier(mem_flags::mem_threadgroup);

  if (lane < 16u) {
    // The specified reduction over 64-key fragments, in the GLSL tree order.
    auto softmaxPair = [&](uint row, uint base, uint pair, uint parity) -> half {
      uint key = row * PT + base + pair * 2u + parity;
      half a = scores[key] + scores[key + 8u];
      half b = scores[key + 16u] + scores[key + 24u];
      half c = scores[key + 32u] + scores[key + 40u];
      half d = scores[key + 48u] + scores[key + 56u];
      half ab = a + b;
      half abc = ab + c;
      return abc + d;
    };
    auto softmax64 = [&](uint row, uint base) -> half {
      half e0 = softmaxPair(row, base, 0u, 0u) + softmaxPair(row, base, 1u, 0u);
      half e1 = e0 + softmaxPair(row, base, 2u, 0u);
      half even = e1 + softmaxPair(row, base, 3u, 0u);
      half o0 = softmaxPair(row, base, 0u, 1u) + softmaxPair(row, base, 1u, 1u);
      half o1 = o0 + softmaxPair(row, base, 2u, 1u);
      half odd = o1 + softmaxPair(row, base, 3u, 1u);
      return even + odd;
    };
    half total = 0.0h;
    for (uint base = 0u; base < PT; base += 64u) total = total + softmax64(lane, base);
    uint padding = PT - pc.tokens;
    if (padding > 0u) {
      half correction = (half)(float(vitExpWeight((half)0.0h)) * float(padding));
      total = total - correction;
    }
    reciprocals[lane] = (half)(1.0 / float(total));
  }
  for (uint i = lane; i < 16u * PT; i += 32u) {
    weights[i] = (uchar)e4m3CodeFromF16Bits(nr_halfBits(scores[i]));
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  // O = P V: the E4M3 weights are unnormalized; the reciprocal lands on the
  // value accumulator in the publication below.
  for (uint idx = lane; idx < 16u * 32u; idx += 32u) {
    uint r = idx >> 5, c = idx & 31u;
    stageA[r * 40u + c] = e4m3ToHalf(weights[r * PT + c]);   // P tile [16][PT(32 window)]
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint cTile = 0u; cTile < 2u; ++cTile) {
    simdgroup_matrix<float16_t, 8, 8> mc[2][2];
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        for (ushort i = 0; i < 8; ++i)
          for (ushort j = 0; j < 8; ++j) mc[r][c][i][j] = 0.0h;
    for (uint kb = 0u; kb < PT; kb += 32u) {
      // Stage P chunk [16 rows][32 keys] and V chunk [32 keys][16 channels].
      for (uint idx = lane; idx < 16u * 32u; idx += 32u) {
        uint r = idx >> 5, c = idx & 31u;
        stageA[r * 40u + c] = e4m3ToHalf(weights[r * PT + kb + c]);
      }
      for (uint idx = lane; idx < 32u * 16u; idx += 32u) {
        uint k = idx >> 4, c = idx & 15u;
        stageB[k * 24u + c] = e4m3ToHalf(qkv[(size_t)(kb + k) * stride3 + headBase + 64u + cTile * 16u + c]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint p = 0u; p < 2u; ++p) {
        for (uint r = 0u; r < 2u; ++r) {
          auto a = simdgroup_load(stageA, (r * 8u) * 40u + p * 8u, (ushort)40u);
          for (uint c = 0u; c < 2u; ++c) {
            auto b = simdgroup_load(stageB, (p * 8u) * 24u + c * 8u, (ushort)24u);
            mc[r][c] = simdgroup_multiply_accumulate(a, b, mc[r][c]);
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        simdgroup_store(mc[r][c], outputs, cTile * 16u + (r * 8u) * 32u + c * 8u, (ushort)32u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = lane; i < 16u * 32u; i += 32u) {
    uint row = i >> 5u, component = i & 31u;
    uint token = qTile * 16u + row;
    if (token >= pc.tokens) continue;
    float value = roundF16(float(outputs[i]) * float(reciprocals[row]));
    attended[token * pc.channels + head * 32u + component] = (uchar)e4m3CodeFromF32(value);
  }
}
