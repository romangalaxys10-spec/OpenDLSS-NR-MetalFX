// gemm_fp8.metal — the FP8 GEMM workhorse of the NR network (Metal port of
// shaders/vulkan-glsl/gemm_fp8.comp, unfused reference route).
//
// D = A(E4M3, rows x K) * B(E4M3, K x N) accumulated in HALF through Apple
// simdgroup MMA as a chain of k32 steps in K order (two k16 planes per step),
// exactly where the Vulkan kernel issues one k32 coopMatMulAdd per tile.
// The prologue seeds the accumulator with the residual/skip vector; the
// epilogue applies the publication order (cubic SiLU, E4M3 quantization, f16)
// element by element with the shared bit-exact conversions.
//
// Differences vs the NVIDIA route (documented, METAL_PORT_SPEC.md §4.6):
//  - DEPTH register prefetch and KSTEP=64 double-staging are NVIDIA scheduling
//    features; Metal v1 stages one k32 step per barrier pass. No arithmetic
//    change: the accumulation chain over K is identical.
//  - Apple half MMA rounds per 8-product plane inside a k16 half where Ada uses
//    its F13 fixed-point grouping; the difference is bounded by the next E4M3/F16
//    publication (every output element is re-quantized between kernels).
//  - PARTITION (in-kernel partial sums every 256/512/1024 K values) keeps the
//    `total` tiles in threadgroup memory instead of a second register array —
//    the same half adds in the same order.

#include "nr_common.metal"

struct GemmFp8Constants {
  uint K;             // constant_id 0
  uint TILES;         // constant_id 1  (BN = 16 * TILES, 1..8)
  uint FLAGS;         // constant_id 2
  uint PARTITION;     // constant_id 3  (0 | 256 | 512 | 1024)
  uint SPLITK;        // constant_id 9
};
constant GemmFp8Constants kc [[function_constants(GemmFp8Constants)]];

constant uint F_RESIDUAL         = 1u;
constant uint F_SCALE_RESIDUAL   = 2u;
constant uint F_SILU             = 8u;
constant uint F_QUANTIZE         = 16u;
constant uint F_DUAL             = 32u;
constant uint F_RESIDUAL_E4      = 64u;
constant uint F_BROADCAST_INPUT  = 128u;

struct GemmFp8Push {
  uint rows;
  uint N;
  uint Nmatrix;
  uint weightColumnOffset;
  uint inputStride;
  uint inputColumnBase;
  uint outputStride;
  uint outputColumnOffset;
  uint auxHalfOffset;   // per-column f16 scale vector, half units (16-byte aligned)
  uint batches;
  uint columnGroups;
  uint splitStride;     // SPLITK > 1: f16 elements per partition slice
};

constant uint SUBGROUPS = 4u;          // 16-row tiles per workgroup; BM = 64
constant uint THREADS   = 128u;
constant uint BM        = 16u * SUBGROUPS;
constant uint MAX_TILES = 8u;
constant uint KSTEP     = 32u;
constant uint AT_STRIDE = 40u;         // A staging row stride (32 + bank spread)
constant uint BT_STRIDE = 40u;
constant uint EP_STRIDE = 8u * 16u + 8u;   // epilogue staging row stride (128 + 8)

threadgroup half atTile[BM * AT_STRIDE];
threadgroup half btTile[MAX_TILES * 16u * BT_STRIDE];
threadgroup half epTile[SUBGROUPS * 16u * EP_STRIDE];
threadgroup half totalTile[SUBGROUPS * 16u * EP_STRIDE];   // PARTITION != 0 only

// E4M3 byte -> half, staged 4 bytes at a time (exact conversion).
static inline void stageBytes4(device const uchar* src, threadgroup half* dst) {
  uchar4 bytes = *(device const uchar4*)src;
  dst[0] = e4m3ToHalf(bytes.x);
  dst[1] = e4m3ToHalf(bytes.y);
  dst[2] = e4m3ToHalf(bytes.z);
  dst[3] = e4m3ToHalf(bytes.w);
}

// ---- Publication of sixteen accumulator halves (GLSL publish16 semantics).
static inline void publish16(uint4 v0, uint4 v1, uint index, uint flags,
                             device uint4* out16x8, device uint4* out8x16) {
  if ((flags & F_SILU) != 0u) {
    v0 = uint4(as_type<uint>(siluPair(as_type<half2>(v0.x))), as_type<uint>(siluPair(as_type<half2>(v0.y))),
               as_type<uint>(siluPair(as_type<half2>(v0.z))), as_type<uint>(siluPair(as_type<half2>(v0.w))));
    v1 = uint4(as_type<uint>(siluPair(as_type<half2>(v1.x))), as_type<uint>(siluPair(as_type<half2>(v1.y))),
               as_type<uint>(siluPair(as_type<half2>(v1.z))), as_type<uint>(siluPair(as_type<half2>(v1.w))));
  }
  if ((flags & (F_QUANTIZE | F_DUAL)) != 0u) {
    uint c0 = (uint)e4m3Pair(as_type<half2>(v0.x)) | ((uint)e4m3Pair(as_type<half2>(v0.y)) << 16);
    uint c1 = (uint)e4m3Pair(as_type<half2>(v0.z)) | ((uint)e4m3Pair(as_type<half2>(v0.w)) << 16);
    uint c2 = (uint)e4m3Pair(as_type<half2>(v1.x)) | ((uint)e4m3Pair(as_type<half2>(v1.y)) << 16);
    uint c3 = (uint)e4m3Pair(as_type<half2>(v1.z)) | ((uint)e4m3Pair(as_type<half2>(v1.w)) << 16);
    out8x16[index >> 4u] = uint4(c0, c1, c2, c3);
  }
  if ((flags & F_QUANTIZE) == 0u) {
    out16x8[index >> 3u] = v0;
    out16x8[(index >> 3u) + 1u] = v1;
  }
}

kernel void nr_gemm_fp8(
    device const uchar*   a4         [[buffer(0)]],   // E4M3 [rows][inputStride]
    device const uchar*   b4         [[buffer(1)]],   // E4M3 [batch][K/32][Nmatrix][32]
    device uint4*         partials   [[buffer(2)]],   // SPLITK > 1: f16 partials
    device const uchar*   residualB  [[buffer(3)]],   // residual: f16 words or E4 bytes (byte reads)
    device const ushort*  aux        [[buffer(4)]],   // per-column f16 scales
    device uint4*         out8x16    [[buffer(5)]],   // E4 output words
    device uint4*         out16x8    [[buffer(6)]],   // f16 output words
    constant GemmFp8Push& pc         [[buffer(30)]],
    uint3 grp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint thread = tid;
  const uint split = grp.x % kc.SPLITK;
  const uint columnLinear = grp.x / kc.SPLITK;
  const uint batch = columnLinear / pc.columnGroups;
  const uint columnGroup = columnLinear % pc.columnGroups;
  const uint blockRow = (grp.y + grp.z * 65535u) * BM;
  const uint rowBase = blockRow + sg * 16u;
  const uint colBase = columnGroup * (kc.TILES * 16u);
  const uint outColumn = pc.outputColumnOffset + batch * pc.N;
  const bool rowsValid = rowBase < pc.rows;
  const uint bn = kc.TILES * 16u;
  const uint epiBase = sg * 16u * EP_STRIDE;

  const uint aColumnBase = pc.inputColumnBase + (((kc.FLAGS & F_BROADCAST_INPUT) != 0u) ? 0u : batch * kc.K);
  const uint steps = kc.K / KSTEP / kc.SPLITK;
  const uint kBegin = split * steps;

  simdgroup_matrix<float16_t, 8, 8> zero;
  for (ushort i = 0; i < 8; ++i)
    for (ushort j = 0; j < 8; ++j) zero[i][j] = 0.0h;

  simdgroup_matrix<float16_t, 8, 8> acc[MAX_TILES][2][2];
  for (uint t = 0u; t < kc.TILES; ++t)
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c) acc[t][r][c] = zero;

  // ---- Partition totals: zeroed once when PARTITION is active.
  const bool usePartition = kc.PARTITION != 0u;
  if (usePartition) {
    for (uint idx = thread; idx < SUBGROUPS * 16u * EP_STRIDE; idx += THREADS) totalTile[idx] = 0.0h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Residual prologue (split 0 only), GLSL pass structure preserved.
  const bool seedResidual = (kc.FLAGS & F_RESIDUAL) != 0u && split == 0u;
  if (seedResidual && (kc.FLAGS & (F_SCALE_RESIDUAL | F_RESIDUAL_E4)) != 0u && rowsValid) {
    for (uint pass = 0u; pass < kc.TILES; pass += 4u) {
      const uint passTiles = (kc.TILES - pass) < 4u ? (kc.TILES - pass) : 4u;
      const uint lpr = passTiles == 1u ? 1u : (passTiles == 2u ? 2u : 4u);
      const uint rpi = 32u / lpr;
      const uint quarter = lane % lpr;
      for (uint it = 0u; it < (16u + rpi - 1u) / rpi; ++it) {
        uint r = it * rpi + lane / lpr;
        uint row = rowBase + r;
        if (r < 16u && quarter < passTiles && row < pc.rows) {
          uint column = colBase + pass * 16u + quarter * 16u;
          float res[16];
          if ((kc.FLAGS & F_RESIDUAL_E4) != 0u) {
            size_t base = (size_t)row * pc.outputStride + outColumn + column;
            for (uint i = 0u; i < 16u; ++i) res[i] = e4m3HwToF32(residualB[base + i]);
          } else {
            device const uchar* src = residualB + ((size_t)row * pc.outputStride + outColumn + column) * 2u;
            for (uint i = 0u; i < 8u; ++i) {
              ushort pair = (ushort)src[2u * i] | ((ushort)src[2u * i + 1u] << 8);
              half2 h = as_type<half2>(pair);
              res[2u * i] = h.x;
              res[2u * i + 1u] = h.y;
            }
          }
          if ((kc.FLAGS & F_SCALE_RESIDUAL) != 0u) {
            for (uint i = 0u; i < 16u; ++i) {
              float scale = (float)as_type<half>(aux[pc.auxHalfOffset + column + i]);
              res[i] = roundF16(res[i] * scale);   // exact f32 product, single half rounding
            }
          }
          for (uint i = 0u; i < 16u; ++i)
            epTile[epiBase + r * EP_STRIDE + quarter * 16u + i] = (half)res[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint t = 0u; t < passTiles; ++t)
        for (uint r = 0u; r < 2u; ++r)
          for (uint c = 0u; c < 2u; ++c)
            acc[pass + t][r][c] = simdgroup_load(epTile, epiBase + (t * 16u + r * 8u) * EP_STRIDE + c * 8u,
                                                 (ushort)EP_STRIDE);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else if (seedResidual && rowsValid) {
    // Plain f16 residual: stage the simdgroup's four 16x16 tiles, then load.
    uint rowClamped = min(rowBase, pc.rows - 1u);
    for (uint t = 0u; t < kc.TILES; ++t) {
      uint column = colBase + t * 16u;
      for (uint idx = lane; idx < 16u * 16u; idx += 32u) {
        uint r = idx >> 4, c = idx & 15u;
        uint row = min(rowBase + r, pc.rows - 1u);   // GLSL: whole-tile load from (possibly padding) rows
        (void)rowClamped;
        size_t addr = ((size_t)row * pc.outputStride + outColumn + column + c) * 2u;
        ushort pair = (ushort)residualB[addr] | ((ushort)residualB[addr + 1u] << 8);
        epTile[epiBase + r * EP_STRIDE + c] = as_type<half>(pair);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = 0u; r < 2u; ++r)
        for (uint c = 0u; c < 2u; ++c)
          acc[t][r][c] = simdgroup_load(epTile, epiBase + (r * 8u) * EP_STRIDE + c * 8u, (ushort)EP_STRIDE);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // prologue staging aliases the operand buffers

  // ---- K loop: stage one k32 step, two k16 MMA planes per tile, K order preserved.
  size_t bBatchBase = (size_t)batch * (kc.K / KSTEP) * pc.Nmatrix;
  for (uint step = 0u; step < steps; ++step) {
    // Stage A rows (clamped like the GLSL).
    for (uint idx = thread; idx < BM; idx += THREADS) {
      uint row = min(blockRow + idx, pc.rows - 1u);
      device const uchar* src = a4 + (size_t)row * pc.inputStride + aColumnBase + (kBegin + step) * KSTEP;
      threadgroup half* dst = atTile + idx * AT_STRIDE;
      for (uint w = 0u; w < KSTEP; w += 4u) stageBytes4(src + w, dst + w);
    }
    // Stage B tile: [n][k] half from the tile-major weights.
    size_t tileBase = bBatchBase + (size_t)(kBegin + step) * pc.Nmatrix + pc.weightColumnOffset + colBase;
    for (uint idx = thread; idx < bn; idx += THREADS) {
      device const uchar* src = b4 + (tileBase + idx) * KSTEP;
      threadgroup half* dst = btTile + idx * BT_STRIDE;
      for (uint w = 0u; w < KSTEP; w += 4u) stageBytes4(src + w, dst + w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rowsValid) {
      for (uint t = 0u; t < kc.TILES; ++t) {
        for (uint p = 0u; p < 2u; ++p) {          // k16 planes within the k32 step
          for (uint r = 0u; r < 2u; ++r) {
            auto a = simdgroup_load(atTile, (sg * 16u + r * 8u) * AT_STRIDE + p * 8u, (ushort)AT_STRIDE);
            for (uint c = 0u; c < 2u; ++c) {
              auto b = simdgroup_load(btTile, (t * 16u + c * 8u) * BT_STRIDE + p * 8u, (ushort)BT_STRIDE, true);
              acc[t][r][c] = simdgroup_multiply_accumulate(a, b, acc[t][r][c]);
            }
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- In-kernel partition boundary: total = (first ? stagedAcc : total + stagedAcc); acc = 0.
    if (usePartition && rowsValid) {
      uint kb = step * KSTEP;
      if (((kb + KSTEP) % kc.PARTITION) == 0u || kb + KSTEP >= kc.K) {
        bool first = kb < kc.PARTITION;
        for (uint t = 0u; t < kc.TILES; ++t) {
          for (uint r = 0u; r < 2u; ++r)
            for (uint c = 0u; c < 2u; ++c)
              simdgroup_store(acc[t][r][c], epTile, epiBase + (r * 8u) * EP_STRIDE + t * 16u + c * 8u,
                              (ushort)EP_STRIDE);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          for (uint idx = lane; idx < 16u * 16u; idx += 32u) {
            uint rr = idx >> 4, cc = idx & 15u;
            uint slot = epiBase + rr * EP_STRIDE + t * 16u + cc;
            half accv = epTile[slot];
            totalTile[slot] = first ? accv : (half)(totalTile[slot] + accv);
            epTile[slot] = 0.0h;
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          for (uint r = 0u; r < 2u; ++r)
            for (uint c = 0u; c < 2u; ++c) acc[t][r][c] = zero;
        }
      }
    }
  }

  if (usePartition && rowsValid) {
    // Load the partition totals back into the accumulators.
    for (uint t = 0u; t < kc.TILES; ++t)
      for (uint r = 0u; r < 2u; ++r)
        for (uint c = 0u; c < 2u; ++c)
          acc[t][r][c] = simdgroup_load(totalTile, epiBase + (r * 8u) * EP_STRIDE + t * 16u + c * 8u,
                                        (ushort)EP_STRIDE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // operand buffers dead: epilogue reuses epTile
  if (!rowsValid) return;

  // ---- Stage the accumulators once; all publishing reads epTile.
  for (uint t = 0u; t < kc.TILES; ++t)
    for (uint r = 0u; r < 2u; ++r)
      for (uint c = 0u; c < 2u; ++c)
        simdgroup_store(acc[t][r][c], epTile, epiBase + (r * 8u) * EP_STRIDE + t * 16u + c * 8u, (ushort)EP_STRIDE);
  simdgroup_barrier(mem_flags::mem_threadgroup);

  if (kc.SPLITK > 1u) {
    // Raw f16 partials of this partition: GLSL word math
    // index = split*splitStride + row*N + batch*N*rows + column; 16 halves -> two uint4 words.
    for (uint t = 0u; t < kc.TILES; ++t) {
      for (uint rr = lane; rr < 16u; rr += 32u) {
        uint row = rowBase + rr;
        if (row >= pc.rows) continue;
        half hs[16];
        for (uint i = 0u; i < 16u; ++i) hs[i] = epTile[epiBase + rr * EP_STRIDE + t * 16u + i];
        uint4 w0 = uint4(as_type<uint>(half2(hs[0], hs[1])), as_type<uint>(half2(hs[2], hs[3])),
                         as_type<uint>(half2(hs[4], hs[5])), as_type<uint>(half2(hs[6], hs[7])));
        uint4 w1 = uint4(as_type<uint>(half2(hs[8], hs[9])), as_type<uint>(half2(hs[10], hs[11])),
                         as_type<uint>(half2(hs[12], hs[13])), as_type<uint>(half2(hs[14], hs[15])));
        uint index = split * pc.splitStride + row * pc.N + batch * pc.N * pc.rows + colBase + t * 16u;
        partials[index >> 3u] = w0;
        partials[(index >> 3u) + 1u] = w1;
      }
    }
    return;
  }

  // ---- Single-partition epilogue: GLSL pass structure (up to 4 tiles per pass).
  for (uint pass = 0u; pass < kc.TILES; pass += 4u) {
    const uint passTiles = (kc.TILES - pass) < 4u ? (kc.TILES - pass) : 4u;
    const uint lpr = passTiles == 1u ? 1u : (passTiles == 2u ? 2u : 4u);
    const uint rpi = 32u / lpr;
    const uint quarter = lane % lpr;
    for (uint it = 0u; it < (16u + rpi - 1u) / rpi; ++it) {
      uint r = it * rpi + lane / lpr;
      uint row = rowBase + r;
      if (r < 16u && quarter < passTiles && row < pc.rows) {
        uint column = colBase + pass * 16u + quarter * 16u;
        half hs[16];
        for (uint i = 0u; i < 16u; ++i) hs[i] = epTile[epiBase + r * EP_STRIDE + quarter * 16u + i];
        uint4 w0 = uint4(as_type<uint>(half2(hs[0], hs[1])), as_type<uint>(half2(hs[2], hs[3])),
                         as_type<uint>(half2(hs[4], hs[5])), as_type<uint>(half2(hs[6], hs[7])));
        uint4 w1 = uint4(as_type<uint>(half2(hs[8], hs[9])), as_type<uint>(half2(hs[10], hs[11])),
                         as_type<uint>(half2(hs[12], hs[13])), as_type<uint>(half2(hs[14], hs[15])));
        publish16(w0, w1, row * pc.outputStride + outColumn + column, kc.FLAGS, out16x8, out8x16);
      }
    }
  }
}
