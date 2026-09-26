# Performance

Measured on the hardware the repo's CI runs on; update your numbers with
`opendlss bench --model <dir> --width W --height H` (host wall time per full frame).

## Reference points

| backend | resolution | note |
| --- | --- | --- |
| Vulkan / RTX 4070 SUPER (upstream, fused + PTX) | 768x768 2.8 ms; 1080p 7.8 ms; 2160p 29.3 ms | the published bit-exact route |
| Metal / Apple silicon (this repo, reference route) | fill in with `opendlss bench` on your M-series | v1.0 runs the unfused decomposition |

## Why the Metal route is a different schedule (and why that is fine)

The upstream speed comes from NVIDIA-specific fusions: one fused dispatch per 32-channel block,
PTX `mma.sync` E4M3 kernels, counter-chained launches without barriers. Those are hardware-shaped;
Metal v1 runs the same arithmetic through the unfused kernel sequence (GEMM -> SiLU/quantize ->
QKV GEMM -> normalize -> attend -> projection) with barriers. Every publication point is
identical, so outputs agree with the Vulkan route within the documented intra-kernel bound
(docs/METAL_PORT_SPEC.md §4.6).

## Roadmap (v1.1)

1. fused_block32.metal (one dispatch per 32-channel block, the megakernel),
2. a fused QKV + normalize + attention kernel and the expert FFN,
3. on-GPU session (no host readback) for the demo/game path,
4. temporal scaler integration with real motion vectors in the demo,
5. streamed ViT route for padded token counts > 256 on Metal.
