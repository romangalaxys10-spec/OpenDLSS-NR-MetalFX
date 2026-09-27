# Changelog

All notable changes to OpenDLSS-NR MetalFX are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-27

The initial end-to-end release: the MetalFX-accelerated port of the
OpenDLSS-NR neural rendering pipeline with Windows and Linux backends,
supporting video, game and image content on Apple Silicon and beyond.

### Added

**Core (cross-platform, C++20)**
- Software IEEE-754 binary16 codec and the network family's half bit-trick
  attention exponentials (`core/include/opendlss/fp16.h`), shared by all
  backends so numerics are identical everywhere.
- E4M3 (OCP FP8) codec with round-to-nearest-even and 448 saturation
  (`core/include/opendlss/e4m3.h`), verified against the reference grid edges
  (min subnormal 2^-9, max normal 448, NaN 0x7F).
- Padded-field geometry — a faithful port of the reference `Geometry::fromValid`
  with the documented alignment quirk, extended to configurable level counts so
  small demo networks work. Verified against the reference's published table
  (1920x1080 → field 1920x1152 with L5 32x20, 512x512 → 576x512, …).
- Model manifest system byte-compatible with the reference layout: `config`,
  `schedule`, `stages` (E4M3-packed files with sha256) and `tensors`
  (name/block/layer/parameter/stage/stageOffset/byteLength), host-side E4M3→f16
  decode once at load.
- The 16-lane preprocess specification (three Gaussian noise lanes from a hash
  of the padded coordinate + per-frame seed, centred proxy and reprojected
  history, style/tone/structure/skin/automask conditioning), with mirrored
  sampling off the valid rectangle.
- Head composition: `neural = clamp(proxy + rgb/4)`,
  `weight = clamp(sigmoid(logit)·blendScale)`, `display = lerp(neural, history)`.
- The IBackend contract (`core/include/opendlss/backend.h`) covering image ops
  (bilateral denoise, spatial upscale + adaptive sharpen), the neural graph,
  video/game temporal ops (motion estimation, reprojection with confidence,
  temporal blend) and the real-time game path.

**Neural rendering graph (CPU reference, `core/backends/cpu/`)**
- Full block semantics: `y = x·ffnScale + FFN(x)`,
  `out = y·attnScale + Proj(Attn(QKV(y)))` with fp16 storage, E4M3
  publications, f32 accumulation, per-width FFN shapes (dense 32-channel,
  grouped wide, 512-branch, ViT 4096-hidden).
- Window attention with the reference's 4-phase shifted-window cycle
  (0,0)/(-4,-4)/(-4,0)/(0,-4), cosine-normalized q/k, learned per-head scale,
  learned 64x64 prior, softmax through the half bit-trick exponential,
  weights published E4M3; out-of-field window tokens contribute zero keys and
  values but keep their prior in the softmax denominator (reference semantics).
- Global ViT attention: unnormalized weights, three-separate-half-multiplies
  q scaling (norm · sqrt(32) · learned), exp(0) padding correction over
  64-padded token counts.
- U-net transitions: half-step 2x2 box pooling → E4M3 → channel GEMM up;
  down-GEMM → nearest 2x upsample → one-f16-FMA skip add; field/L0 adapter
  (block-0 output · adapterScale); data-driven block schedule from the manifest.

**macOS — the MetalFX flagship (`platforms/macos/`)**
- Metal compute port of every graph kernel (`platforms/macos/shaders/`):
  Common.metal (shared E4M3/hash/bit-trick helpers, bit-identical to the CPU),
  Preprocess.metal, Graph.metal (FFN, window/global attention, block skip and
  epilogue, pool, channel GEMM, decoder skip, upsample, head), Media.metal
  (bilateral denoise, Lanczos3 passes, adaptive sharpen, 16x16 block-matching
  motion, reprojection with bilinear confidence, temporal blend, head composite).
- `MetalBackend` (Objective-C++, ARC): MetalFX integration —
  `MTLFXSpatialScaler` for image/video scaling with automatic fallback to the
  in-house Lanczos path, `MTLFXTemporalScaler` for the game path (color/depth/
  motion/exposure/reset feedback, RG16Float motion vectors, RGBA16Float color);
  weight upload pipeline, PSO cache, threadgroup-sized window attention.
- SwiftUI + MetalKit viewer (`platforms/macos/viewer/OpenDLSSViewer`) rendering
  a synthetic scene through the live temporal scaler with per-frame stats.

**Windows (`platforms/windows/`)**
- D3D12 + DirectML backend: factory/device (WARP fallback), runtime-compiled
  HLSL 6.x compute shaders (complete port of the shader set, Common.hlsli with
  bit-identical E4M3/half helpers), typed-UAV token buffers, root signatures +
  PSO cache, fence-synced dispatch, DirectML-accelerated channel GEMMs with a
  graceful HLSL fallback, full IBackend implementation.

**Linux (`platforms/linux/`)**
- Vulkan compute backend: dependency-free bootstrap (`vk_boot.h` — dlopen +
  vkGetInstanceProcAddr table with ABI-checked structures, no SDK headers
  required), 20 GLSL 450 compute shaders (bit-identical helpers in
  Common.glsl), SSBO token buffers + storage images, CMake glslangValidator
  integration producing SPIR-V at build time, full IBackend implementation.

**Tooling & integration**
- `apps/cli` — the `opendlss-nr` tool: `info`, `geometry` (reference parity
  helper), `image`, `video`, `game`, `demo`, `bench`.
- `sdk/` — stable C ABI (`opendlss_nr_sdk.h`) for engine integration:
  create/resolve/destroy with per-frame color/depth/motion/exposure/jitter and
  conditioning scalars.
- `tools/make_demo_weights.py` — generates the complete untrained demo model
  (manifest + E4M3-packed stage + sha256) with a real 7-block schedule
  including a global-attention block.
- `tools/export_weights_pytorch.py` — checkpoint → manifest export path for
  trained models.
- Demos: synthetic noisy-scene image demo, procedural Y4M video generator +
  temporal video demo, synthetic game-loop demo, C engine example, and
  `demos/run_all.sh` driving all of them.
- Tests: `tests/test_core.cpp` (fp16/E4M3 codec edges, geometry parity,
  graph determinism, composite bounds) wired to CTest; AddressSanitizer-clean.
- Scripts: per-platform installers, build helpers, demo runner.
- GitHub Actions CI: macOS (Apple Silicon runners building Metal + MetalFX),
  Ubuntu (core + CPU + Vulkan with glslang), Windows (core + CPU + D3D12),
  plus an ASan job.

### Fixed (during development, found by the ASan/test gates)
- Manifest `onField` flag was not parsed — field-level blocks (the reference's
  blocks 0/70 class) ran at level-0 dimensions, corrupting the ladder.
- E4M3 subnormal decode exponent was -12 instead of -9 and the encode scale
  was 4096 instead of 512 (both caught by codec edge tests).
- Multi-head window attention overwrote per-head q/k/v scratch (single-head
  path was unaffected).
- Encoder/decoder transition branches used the wrong depth comparison after
  the field was assigned its correct "shallower than level 0" ordering.
