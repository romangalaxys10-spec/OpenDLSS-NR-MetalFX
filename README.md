# OpenDLSS-NR-MetalFX

**Cross-platform neural rendering, MetalFX-accelerated on Apple silicon.**
Metal + MetalFX for macOS (M1/M2/M3/M4), the original bit-exact Vulkan route for Windows and
Linux — one repository, one CLI, one C SDK.

This is the full cross-platform continuation of
[maanHimself/OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR): a Vulkan reimplementation
of NVIDIA's DLSS 5 Neural Rendering network (the same 71-block Swin / ViT architecture as
DLSS-NR build 310.8.0, FP8 E4M3 activations with FP16 accumulation, ~141 MiB of weights). The
original runs on Windows + NVIDIA Ada. This repository carries it to every modern desktop:

| platform | backend | status |
| --- | --- | --- |
| macOS 13+ (Apple silicon M1/M2/M3/M4) | **Metal compute + MetalFX Spatial/Temporal** (`src/metal/`) | new in this repo |
| Windows 10+ (NVIDIA Ada or newer) | Vulkan cooperative-matrix FP8 + PTX fast path (unchanged from upstream) | carried, proven |
| Linux (Vulkan 1.3, cooperative matrix) | same Vulkan route, CMake + shell builds, deb/AppImage | new in this repo |

**You supply the weights**, as a model directory in the layout the original project defines
([docs/original-design/weights.md](docs/original-design/weights.md)). Nothing in this repository
produces them; nothing here changes what the network computes on the Vulkan route.

---

## What the network is

A generative neural rendering network (NVIDIA's term): it re-renders the frame the engine already
drew — adjusting tone, structure and skin under a style setting, generating detail from injected
noise. It takes one rendered frame (a low-dynamic-range proxy), three lanes of Gaussian noise, the
previous frame's output reprojected, and five conditioning scalars, and produces an RGB residual
plus a temporal-blend logit per pixel. It is *not* an upscaler by itself; MetalFX provides the
upscaling half of the pipeline. NVIDIA describes the model in
[DLSS 5: Generative Neural Rendering](https://research.nvidia.com/labs/adlr/DLSS5/files/DLSS5_Report.pdf).
The full graph and its exact arithmetic are documented in
[docs/original-design/network.md](docs/original-design/network.md) and
[docs/original-design/numerics.md](docs/original-design/numerics.md).

## The three product paths

```
images:   source.png ────────────────► NR network ─► out.png            (opendlss image)
video:    frames ──► FFmpeg ─► NR per frame ──► FFmpeg ─► out.mp4       (opendlss video)
games:    low-res render ─► MetalFX Temporal/Spatial ─► NR pass ─► present
          (apps/demo-game-macos, the C API in include/opendlss/opendlss.h)
```

- **Images** — `opendlss image in.png out.png --style 64 --grain 7`. The CLI, the C SDK and the
  demo all go through one frame pipeline (`src/common/nr_frame*`), so every surface behaves the
  same.
- **Video** — `opendlss video in.mp4 out.mp4` (FFmpeg build). Per-frame network with the
  production frame schedule; container frame rate preserved; audio copied.
- **Games** — the realtime demo renders a procedural scene at half resolution, upscales with
  MetalFX, then runs the NR pass, at interactive rates on M-series GPUs. Engines integrate the
  same path through the C API or by driving `nr::metal::Graph` directly (docs/GAMES.md).

## Quick start

macOS (Apple silicon):

```bash
git clone https://github.com/maanHimself/OpenDLSS-NR-MetalFX.git
cd OpenDLSS-NR-MetalFX
scripts/macos/build.sh            # CMake + precompiled Metal shaders (opendlss.metallib)
scripts/macos/build.sh demo       # + demo-game-macos.app
build/opendlss-cli selftest       # CPU numeric contracts
build/opendlss-cli devices
# with a model at models/nr (see docs/INSTALL.md):
build/opendlss-cli image input.png output.png --style 64
scripts/macos/run_demo.sh
```

Windows (NVIDIA, original route):

```powershell
git clone https://github.com/maanHimself/OpenDLSS-NR-MetalFX.git
cd OpenDLSS-NR-MetalFX
powershell -File scripts\windows\fetch_tools.ps1
powershell -File scripts\windows\build.ps1        # opendlss.exe + dlss5vk.exe
build\Release\dlss5vk.exe bench --model models\nr --width 768 --height 768
```

Linux:

```bash
scripts/linux/build.sh            # fetches glslang/Vulkan-Headers/volk into tools/
./build/opendlss-cli bench --model models/nr --width 1920 --height 1080
```

## What is here

| part | files | notes |
| --- | --- | --- |
| Metal backend | `src/metal/` | MSL ports of the reference kernel route (GEMM, attention, ops), the 71-block graph, MetalFX wrappers; Apple-native simdgroup-matrix MMA with the same publication points |
| Vulkan backend | `src/vulkan/`, `shaders/vulkan-glsl/`, `ptxgen/` | the original proven code, carried unchanged (cooperative-matrix FP8 GLSL + PTX generators + `dlss5vk` parity tool) |
| Frame pipeline | `src/common/` | the shared product surface (image/video/CLI/SDK/demos all use it) |
| C SDK | `include/opendlss/opendlss.h`, `src/sdk/` | stable C API for engines and tools |
| CLI | `src/cli/` | `opendlss` — image, video, bench, profile, devices, selftest |
| Demo | `apps/demo-game-macos/` | the realtime MetalFX + NR game path |
| Filament demo | `apps/demo-filament/` | the original glTF renderer demo (Windows) |
| WebGPU port | `ports/browser-webgpu/` | the original browser implementation (carried) |
| Tests | `tests/` | exhaustive f16/E4M3 conversion tables, SiLU table, Ada MMA contracts, geometry — run in CI on all three OSes |

## Numerics, honestly

The Vulkan/NVIDIA route is the original, bit-exact implementation and stays untouched. The Metal
route re-implements the same contracts with Apple-native arithmetic: every value that crosses a
kernel boundary is published through the identical bit-level E4M3/F16 rounding rules (verified
against the CPU reference by the test suite), and the accumulation chain over K is identical. The
one difference is inside a k16 half-product group, where Apple's simdgroup MMA replaces Ada's F13
fixed-point grouping; the next E4M3/F16 publication bounds the effect (every block state is
re-quantized, so drift cannot accumulate). docs/METAL_PORT_SPEC.md is the binding porting
contract, and docs/PERFORMANCE.md reports the measured numbers.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) — models, dependencies, installing (pkg/deb/AppImage/zip)
- [docs/BUILDING.md](docs/BUILDING.md) — CMake targets, toolchains, CI
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — repo map, backends, data flow
- [docs/METALFX.md](docs/METALFX.md) — spatial vs temporal scalers, how they compose with NR
- [docs/GAMES.md](docs/GAMES.md) — the engine integration contract (C API + Metal direct)
- [docs/VIDEO.md](docs/VIDEO.md) — the FFmpeg pipeline and the low-field/high-proxy workflow
- [docs/IMAGES.md](docs/IMAGES.md) — the image path and conditioning knobs
- [docs/METAL_PORT_SPEC.md](docs/METAL_PORT_SPEC.md) — GLSL→MSL porting contract
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — measurements and the fused-route roadmap
- [docs/original-design/](docs/original-design/) — the upstream network/numerics/weights docs

## License

MIT, matching the original repository (see LICENSE and NOTICE; NVIDIA's DLSS report is referenced
for the model description only — no NVIDIA code or weights are included).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
