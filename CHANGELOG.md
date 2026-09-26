# Changelog

All notable changes to OpenDLSS-NR-MetalFX. The project follows semantic versioning;
the Vulkan/NVIDIA backend is carried unchanged from upstream and its fixes are tracked there.

## [1.0.0] - 2026-09-27

First cross-platform release. One repository, three operating systems.

### Added - macOS (Apple silicon, the flagship)

- `src/metal/` — a complete Metal backend for the DLSS 5 NR architecture:
  - `nr_metal_context.mm`: device/queue/buffer/pipeline host (shared-memory storage,
    function-constant specialization mirroring the Vulkan keying).
  - `nr_metal_model.mm`: manifest + SHA-256-verified stage loading, the host re-layout of the
    MMA-fragment-ordered weights (byte math identical to the Vulkan port).
  - `src/metal/shaders/*.metal`: MSL ports of the reference kernel route — FP8 GEMM with
    Apple simdgroup-matrix MMA (k32 chain, residual seeding, SiLU, E4M3 quantization, split-K +
    reduce), f16 GEMM, the five elementwise ops, Gaussian preprocess, window/global cosine
    attention with the network's exact softmax trees. Porting contract in docs/METAL_PORT_SPEC.md.
  - `nr_metal_graph.mm`: the 71-block network (pre adapter, encoder 32/64/128/256, split-512,
    global ViT, decoder chain, post block + RGBA head), barrier-separated production schedule.
  - `nr_metalfx.mm`: MetalFX Spatial (Adaptive/Bilinear) and Temporal scaler wrappers.
- Realtime demo `apps/demo-game-macos`: procedural scene → MetalFX upscale → NR pass → MTKView,
  the game path end to end.
- Scripts: `scripts/macos/{build.sh,run_demo.sh,process_image.sh,process_video.sh}`,
  metallib precompilation, app-bundle assembly; Homebrew formula + pkg packaging.

### Added - Windows and Linux

- Windows: CMake build of the original Vulkan route (cooperative-matrix FP8 GLSL + PTX fast path),
  `opendlss.exe` and the original `dlss5vk.exe` parity tool; Inno Setup packaging.
- Linux: first build support for the Vulkan route (CMake + `scripts/linux/build.sh`,
  portable toolchain fetch, glslang SPIR-V compilation), `.deb` and AppImage packaging.
- GitHub Actions CI on macOS 14 (arm64, compiles Metal + runs tests), windows-2022, ubuntu-24.04;
  release workflow producing per-OS artifacts.

### Added - product surface (all platforms)

- `opendlss` CLI: `image`, `video` (FFmpeg), `bench`, `profile`, `devices`, `selftest`.
- C SDK `libopendlss` (`include/opendlss/opendlss.h`): session create/process/destroy, stable ABI
  for game engines and tools.
- Shared frame pipeline (`src/common/nr_frame*`) used by the CLI, SDK and demo alike, with the
  composition rule `out = clamp((head/32 + centred)*8 + 0.5, 0, 1)`.
- Portable CPU reference (`src/common/nr_cpu_reference.*`) of the network arithmetic for
  tests and parity bisecting on every platform.
- Test suite: exhaustive f16↔f32 round-trip (all 65536 codes), exhaustive E4M3 decode/re-encode
  (all 256 codes), the 65536-entry SiLU table, Ada FP8/F16 MMA contracts, geometry/layout rules.

### Carried unchanged from upstream

- The Vulkan bit-exact route: `src/vulkan/*`, `shaders/vulkan-glsl/*`, `ptxgen/*`, the Filament
  demo (`apps/demo-filament/`), the WebGPU port (`ports/browser-webgpu/`) and the original design
  documents (`docs/original-design/`).

### Known limitations (roadmap for 1.1)

- Metal v1 runs the reference (unfused) kernel decomposition — correctness first; the fused
  megakernel routes (fused_block32, qkv_attention, expert-FFN) remain Vulkan/NVIDIA performance
  work. See docs/PERFORMANCE.md.
- The Metal ViT route caps padded token counts at 256 (resolutions up to ~4K with the standard
  field geometry); larger token counts take the Vulkan streamed route.
- The demo composes through host readback (shared memory); keeping the whole loop on-GPU is the
  v1.1 work item.
