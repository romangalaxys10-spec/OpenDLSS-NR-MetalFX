# Metal Shader Port Specification (GLSL -> MSL)

This document is the binding contract for every kernel in `src/metal/shaders/`.
It exists so that the host code (`src/metal/metal_kernels.mm`) and the kernels
never drift apart, and so every port produces the same bytes as the Vulkan
reference route at every publication point.

## 0. Source of truth

- GLSL originals: `shaders/vulkan-glsl/*.comp` (copied verbatim from the original
  OpenDLSS-NR repository; they compile to SPIR-V for the Vulkan backend).
- Shared numeric helpers: `src/metal/shaders/nr_common.metal`. EVERY kernel
  `#include "nr_common.metal"` and uses only those helpers for f16/E4M3
  conversions, SiLU, the exponential approximations and the tiled-token maps.
  Do not reimplement rounding locally.

## 1. Naming and file layout

- One GLSL kernel `X.comp` -> one `X.metal` with entry point `kernel void nr_X(...)`.
- Keep the GLSL comment header (translated), and add a line
  `// Port of shaders/vulkan-glsl/X.comp <hash-of-contract>` describing what the
  kernel computes so future work can diff against the reference.
- Helpers shared between kernels go in `nr_common.metal` (numeric) or a kernel-
  local `static inline` (private to one kernel).

## 2. Argument conventions (match metal_kernels.mm exactly)

Vulkan binding i (0..11) -> Metal `[[buffer(i)]]`.
Push constant block -> a struct named `Push` with EXACTLY the same field order
and types (uint/float), passed as `constant Push& pc [[buffer(30)]]`; the host
encodes it with `setBytes`.
SiLU table (Vulkan binding 9 in some kernels / separate buffer) -> `[[buffer(9)]]`
as `constant ushort*` unless the kernel document says otherwise.

Specialization constants -> `constant` function constants:

```metal
struct NrConstants {
  uint FLAGS  [[function_constant(0)]];
  uint TILES  [[function_constant(1)]];
  ...
};
constant NrConstants kc [[function_constants(NrConstants)]];
```

Host builds one `MTLFunctionConstantValues` per pipeline key, mirroring the
Vulkan `SpecConstants` keying (`shader:const:value...`).

## 3. Threadgroup layout

Keep the GLSL `local_size` exactly:

| kernel | local size | notes |
| --- | --- | --- |
| gemm_fp8 | 32 * SUBGROUPS (128 or 256) | 4 or 8 simdgroups |
| gemm_f16 | 128 | 4 simdgroups |
| ops | 256 | linear dispatch, 8 channels/thread-group-slot |
| preprocess | (8,8,1) | 2D grid |
| fused_block32 | 256 | two windows per workgroup, 8 simdgroups |
| qkv_attention | 128 | one (window, head) per workgroup |
| window_normalize | 256 | |
| window_attend | 128 | |
| global_normalize | 256 | |
| global_attend | 32 | one simdgroup |
| global_attention | 32 | one simdgroup |
| gemm_mlp | 128 | 4 simdgroups, 16 rows each |
| gemm_reduce | 256 | |

Linear global-index recovery (GLSL `gl_GlobalInvocationID.x + y*65535*256`) is
replaced by Metal's `gid = tgPos + grpPos * tgs`; when the host dispatches
linearly it uses a 1D grid sized `ceil(count / threads)`, so kernels simply use
`gid.x`. Keep the `if (id >= bound) return;` guards with the SAME bounds.

## 4. Numerical contracts (non-negotiable)

1. Every value that crosses a kernel boundary is published through the same
   rounding as the CPU reference:
   - f32 -> f16: `roundF16()` from nr_common.metal (bit-level RNE), or the
     equivalent half conversion when the GLSL uses `packFloat2x16(f16vec2(v))`.
   - f16 -> E4M3: `e4m3CodeFromF16Bits` / `e4m3Pair` (NaN -> +0, saturate 448).
   - E4M3 -> f32/f16: `e4m3HwToF32` / `e4m3ToHalf` (NaN code reads as 0).
2. SiLU is always `siluHalf` / `siluPair` (cubic half-FMA approximation).
3. Window attention softmax uses `expWeight` and the fixed reduction tree order
   of the original kernel; the global ViT uses `vitExpWeight` and its own tree.
4. Cosine normalization uses the packed `fma(low, low, high*high)` + tree order
   (`inverseNorm16`).
5. GEMM accumulation: products of dequantized E4M3 values accumulate through
   Apple simdgroup half MMA (`simdgroup_matrix<float16_t,8,8>` with half
   accumulators), staged per k32 step exactly where the Vulkan kernel performs a
   `coopMatMulAdd` per k32 tile. Do NOT reorder the K loop, do NOT switch to an
   f32 accumulator, do NOT fuse steps: every k32 boundary matches the reference
   accumulation chain.
6. The Ada-specific F13 fixed-point truncation inside a 16-product group is
   hardware behavior that Apple silicon replaces with its own half MMA rounding;
   this is the only numeric difference of the Metal route and it is bounded by
   the next E4M3/F16 publication. Document it, never "fix" it by reordering.

## 5. E4M3 staging pattern for GEMMs

Weights arrive in the model's k32-tile-major layout
(`[batch][K/32][Nmatrix][32]` bytes). The Vulkan kernel stages raw uvec4 words;
the Metal kernel DEQUANTS to half when staging global -> threadgroup (coalesced
`uchar4` loads, `half` stores into a `threadgroup half` tile), then runs
`simdgroup_load` from the half tile. Dequantization is exact, so this does not
change any arithmetic; it keeps the MMA loop free of byte shuffling.

A-matrix activations stay E4M3 in memory ([rows][stride] bytes, row stride =
`inputStride` channels, rows padded to 64) and are dequantized with the same
pattern during staging.

## 6. What NOT to port

- The PTX fast path (`scripts/ptx/`, VK_NV_cuda_kernel_launch) and its counter
  chaining: Metal uses barrier-separated launches; the graph code passes
  `chained = false` everywhere. Kernels must not read or write sync counters.
- `VK_KHR_pipeline_executable_properties` statistics: not applicable.
- Split-K partials stay supported (binding 2 partial buffer + `gemm_reduce`),
  because the graph uses partitioned GEMMs for the ViT/split blocks.

## 7. Verification checklist for every ported kernel

- [ ] Push struct fields match the GLSL push_constant block 1:1 (order, width).
- [ ] Buffer bindings map to the same logical operands as the GLSL bindings.
- [ ] All rounding goes through nr_common.metal helpers.
- [ ] The softmax / normalization tree order is character-identical.
- [ ] Out-of-bounds behavior identical (mirrored-window clamps, `rowsValid`
      guards, padding rows produce the same zeros/NaN pattern).
- [ ] Threadgroup sizes and grid shapes match §3.
- [ ] Compiles standalone: `xcrun -sdk macosx metal -c X.metal` (CI job).
