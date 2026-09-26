// FP16 GEMM on cooperative matrices for the network's two m16n8k16 boundaries:
// the 16 -> 32 input adapter of the fused pre block and the 32 -> 4 RGBA
// post head. A and B are f16, accumulation is f16 (two eight-product F24
// reductions per k16 step on Ada). C starts at zero.
//
// Port of shaders/vulkan-glsl/gemm_f16.comp. Each 16x16 Vulkan cooperative
// matrix maps to a 2x2 quad of Apple 8x8 simdgroup matrices; each k16
// coopMatMulAdd maps to two 8x8x8 simdgroup mma steps in k order (see the
// numeric note at the bottom of this file). K / TILE_N / FLAGS are Vulkan
// constant_id 0 / 1 / 2 -> function_constant(0) / (1) / (2); the host builds
// one pipeline per key, mirroring the Vulkan spec-constant keying.
//
// local_size_x = 128 (4 simdgroups of 32). The workgroup grid replicates the
// Vulkan dispatch: rowBase uses gl_WorkGroupID.y + gl_WorkGroupID.z * 65535,
// so the host splits ceil(rows / 64) across grid y/z the same way. Like the
// Vulkan route, the A rows of the last (possibly partial) row block and the
// padded B columns must be readable in the bound buffers.

#include "nr_common.metal"

// Vulkan constant_id 0 / 1 / 2 -> function_constant(0) / (1) / (2).
struct NrGemmF16Constants {
  uint K [[function_constant(0)]];       // reduction length in elements (multiple of 16)
  uint TILE_N [[function_constant(1)]];  // padded N (multiple of 16)
  uint FLAGS [[function_constant(2)]];
};
constant NrGemmF16Constants kc [[function_constants(NrGemmF16Constants)]];

constant uint F_QUANTIZE = 16u;   // main output E4M3 bytes (binding 5)
constant uint F_DUAL = 32u;       // f16 to binding 2 and E4M3 to binding 5
constant uint F_OUT_F32 = 256u;   // f32 output to binding 7

struct Push {
  uint rows;
  uint N;             // valid output columns
  uint inputStride;   // A row stride (elements)
  uint outputStride;  // output row stride (elements)
};

typedef simdgroup_matrix<half, 8, 8> sg_half8x8;

kernel void nr_gemm_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out16 [[buffer(2)]],
    device uchar* out8 [[buffer(5)]],
    device float* out32 [[buffer(7)]],
    constant Push& pc [[buffer(30)]],
    uint3 grp [[threadgroup_position_in_grid]],
    uint subgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint TILES = kc.TILE_N / 16u;
  // Metal threadgroup arrays cannot be sized by function constants, so the
  // tileC staging below is fixed at the maximum: TILE_N <= 64 (host caps
  // paddedN at 32 today; 64 leaves headroom). This guard is dead for every
  // configuration the host can build.
  if (TILES > 4u) return;
  const uint rowBase = (grp.y + grp.z * 65535u) * 64u + subgroup * 16u;
  const uint tileBase = subgroup * 16u * kc.TILE_N;

  // GLSL: shared float16_t tileC[4u * 16u * TILE_N] — see the size note above.
  threadgroup half tileC[4u * 16u * 64u];

  // One 16x16 cooperative-matrix accumulator == four 8x8 simdgroup tiles.
  // Only the first TILES entries are used (TILE_N <= 64).
  sg_half8x8 acc[4][2][2];   // [tile][row quadrant][column quadrant]
  sg_half8x8 ma[2][2];       // A quadrants [row][k]
  sg_half8x8 mb[2][2];       // B quadrants [k][column]

  if (kc.K == 0u) {
    // Degenerate K = 0 (not reachable from the model): the Vulkan accumulator
    // stays at its zero initializer, so tileC must still publish zeros.
    if (lane == 0u) {
      for (uint t = 0u; t < TILES; ++t) {
        for (uint r = 0u; r < 16u; ++r) {
          for (uint c = 0u; c < kc.TILE_N; ++c) {
            tileC[tileBase + t * 16u + r * kc.TILE_N + c] = (half)0.0h;
          }
        }
      }
    }
  } else {
    for (uint kb = 0u; kb < kc.K; kb += 16u) {
      // coopMatLoad(ma, a, rowBase * pc.inputStride + kb, pc.inputStride, RowMajor)
      for (uint rq = 0u; rq < 2u; ++rq) {
        for (uint kq = 0u; kq < 2u; ++kq) {
          simdgroup_load(ma[rq][kq], a + (rowBase + rq * 8u) * pc.inputStride + kb + kq * 8u, (ulong)pc.inputStride);
        }
      }
      for (uint t = 0u; t < TILES; ++t) {
        // coopMatLoad(mb, b, kb * TILE_N + t * 16u, TILE_N, RowMajor)
        for (uint kq = 0u; kq < 2u; ++kq) {
          for (uint cq = 0u; cq < 2u; ++cq) {
            simdgroup_load(mb[kq][cq], b + (kb + kq * 8u) * kc.TILE_N + t * 16u + cq * 8u, (ulong)kc.TILE_N);
          }
        }
        // acc[t] = coopMatMulAdd(ma, mb, acc[t]), decomposed into two k8 mma
        // steps in k order (kq = 0: kb + 0..7, kq = 1: kb + 8..15).
        for (uint kq = 0u; kq < 2u; ++kq) {
          for (uint rq = 0u; rq < 2u; ++rq) {
            for (uint cq = 0u; cq < 2u; ++cq) {
              if (kb == 0u && kq == 0u) {
                // The Vulkan accumulator starts at zero and A*B + 0 == A*B, so
                // the first step of every chain is a plain multiply (the same
                // mma instruction with a zero accumulator).
                simdgroup_multiply(acc[t][rq][cq], ma[rq][kq], mb[kq][cq]);
              } else {
                simdgroup_multiply_accumulate(acc[t][rq][cq], ma[rq][kq], mb[kq][cq], acc[t][rq][cq]);
              }
            }
          }
        }
      }
    }
  }
  // coopMatStore(acc[t], tileC, tileBase + t * 16u, TILE_N, RowMajor)
  for (uint t = 0u; t < TILES; ++t) {
    for (uint rq = 0u; rq < 2u; ++rq) {
      for (uint cq = 0u; cq < 2u; ++cq) {
        simdgroup_store(acc[t][rq][cq], tileC + tileBase + t * 16u + rq * 8u * kc.TILE_N + cq * 8u, (ulong)kc.TILE_N);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = lane; i < 16u * kc.TILE_N; i += 32u) {
    uint r = i / kc.TILE_N, c = i % kc.TILE_N;
    uint row = rowBase + r;
    if (row >= pc.rows || c >= pc.N) continue;
    uint index = row * pc.outputStride + c;
    half value = tileC[tileBase + i];
    if ((kc.FLAGS & F_OUT_F32) != 0u) out32[index] = (float)value;
    else if ((kc.FLAGS & F_QUANTIZE) != 0u) out8[index] = e4m3Code(value);
    else out16[index] = value;
    if ((kc.FLAGS & F_DUAL) != 0u) out8[index] = e4m3Code(value);
  }
}

// Numeric note (port spec section 4.6): Ada rounds each 16-product k16 group
// through its internal F24 accumulator; Apple's simdgroup mma publishes a half
// rounding every 8 products. The k order is unchanged and the difference is
// bounded by the next E4M3 / F16 publication; do not "fix" it by reordering.
