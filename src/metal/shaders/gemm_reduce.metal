// Split-K reduction + publication for gemm_fp8 (SPLITK > 1): the partial FP16
// accumulators of the K partitions are summed in partition order
// (((p0 + p1) + p2) + p3 ...), exactly the `total = total + acc` chain of the
// single-workgroup kernel, then published (SiLU, E4M3 quantization, f16).
//
// Port of shaders/vulkan-glsl/gemm_reduce.comp. SPLITK / FLAGS are Vulkan
// constant_id 0 / 1 -> function_constant(0) / (1); the host builds one
// pipeline per key, mirroring the Vulkan spec-constant keying.
//
// local_size_x = 256; the host dispatches a 1D grid, so the GLSL
// gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 65535u * 256u linear-id
// recovery is simply gid.x (spec section 3).

#include "nr_common.metal"

// Vulkan constant_id 0 / 1 -> function_constant(0) / (1).
struct NrGemmReduceConstants {
  uint SPLITK [[function_constant(0)]];
  uint FLAGS [[function_constant(1)]];
};
constant NrGemmReduceConstants kc [[function_constants(NrGemmReduceConstants)]];

constant uint F_SILU = 8u;
constant uint F_QUANTIZE = 16u;
constant uint F_DUAL = 32u;

struct Push {
  uint rows;
  uint N;
  uint splitStride;        // elements per partition slice in the partial buffer
  uint outputStride;
  uint outputColumnOffset;
};

kernel void nr_gemm_reduce(
    device const uint4* partial16x8 [[buffer(0)]],
    device uint4* out16x8 [[buffer(2)]],
    device uint2* out8x8 [[buffer(5)]],
    constant Push& pc [[buffer(30)]],
    uint3 gid [[thread_position_in_grid]]) {
  // GLSL: uint group = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 65535u * 256u;
  // (8 columns per thread; the host dispatches a 1D grid so gid.x is the linear id)
  uint group = gid.x;
  uint groupsPerRow = pc.N / 8u;
  uint row = group / groupsPerRow, column = (group % groupsPerRow) * 8u;
  if (row >= pc.rows) return;
  uint4 total = partial16x8[(row * pc.N + column) >> 3u];
  // The GLSL loop is [[unroll]]-ed; SPLITK is fixed per pipeline and the
  // partition-order accumulation chain (one half2 add per step, published as
  // packed halves) is unchanged.
  for (uint s = 1u; s < kc.SPLITK; ++s) {
    uint4 part = partial16x8[(s * pc.splitStride + row * pc.N + column) >> 3u];
    for (uint i = 0u; i < 4u; ++i) {
      half2 sum = as_type<half2>(total[i]) + as_type<half2>(part[i]);
      total[i] = as_type<uint>(sum);
    }
  }
  if ((kc.FLAGS & F_SILU) != 0u) {
    for (uint i = 0u; i < 4u; ++i) total[i] = as_type<uint>(siluPair(as_type<half2>(total[i])));
  }
  uint index = row * pc.outputStride + pc.outputColumnOffset + column;
  if ((kc.FLAGS & (F_QUANTIZE | F_DUAL)) != 0u) {
    uint2 codes;
    codes.x = (uint)e4m3Pair(as_type<half2>(total[0])) | ((uint)e4m3Pair(as_type<half2>(total[1])) << 16u);
    codes.y = (uint)e4m3Pair(as_type<half2>(total[2])) | ((uint)e4m3Pair(as_type<half2>(total[3])) << 16u);
    out8x8[index >> 3u] = codes;
  }
  if ((kc.FLAGS & F_QUANTIZE) == 0u) out16x8[index >> 3u] = total;
}
