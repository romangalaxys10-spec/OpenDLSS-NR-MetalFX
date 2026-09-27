# OpenDLSS-NR MetalFX

A MetalFX-accelerated, cross-platform reimplementation of the OpenDLSS-NR neural
rendering pipeline — the DLSS-style network family (Swin/ViT transformer blocks,
FP8(E4M3) publications, FP16 accumulation, temporal reprojection) rebuilt for
**video, games and images** on **Apple Silicon (M1, M2 and later)** with Metal and
MetalFX, plus Windows (D3D12 + DirectML) and Linux (Vulkan) in the same repository.

Based on [maanHimself/OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) —
the bit-exact Vulkan port of NVIDIA's DLSS 5 Neural Rendering network. This project
ports that architecture (same 16-lane input packing, same block semantics, same
padded-field geometry, same head composition) onto Apple's acceleration stack and
wraps it in a complete end-to-end toolchain: CLI, C SDK for engines, viewer,
demos, tests, CI and installation scripts for all three desktop platforms.

```
┌──────────── render/proxy ────────────┐
│  video frame | game color | image    │
└──────────────┬───────────────────────┘
               ▼
   ┌───────────────────────┐
   │ preprocess (16 lanes) │  3 noise lanes · proxy · reprojected history ·
   └──────────┬────────────┘  style/tone/structure/skin/mask conditioning
              ▼
   ┌───────────────────────┐
   │ neural rendering graph│  Swin window-attention blocks + global ViT,
   │ (fp16 + E4M3 grid)    │  U-net transitions, 4-phase shifted windows
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ head + composite      │  residual/4 + sigmoid(logit)·blendScale history lerp
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ MetalFX scaling       │  ◄── MTLFXSpatialScaler (images, video frames)
   │ (Apple Silicon)       │  ◄── MTLFXTemporalScaler (games: motion+depth+exposure)
   └──────────┬────────────┘
               ▼
        display-resolution output
```

## The three platforms, one repository

| Platform | Backend | Acceleration | Entry |
| --- | --- | --- | --- |
| macOS 13+ (Apple Silicon) | `metal+metalfx` | Metal 3 compute + **MTLFXSpatialScaler / MTLFXTemporalScaler** | `platforms/macos/` |
| Windows 10+ | `d3d12` | D3D12 compute + DirectML GEMMs | `platforms/windows/` |
| Linux | `vulkan` | Vulkan compute (GLSL→SPIR-V) | `platforms/linux/` |
| everywhere | `cpu` | software reference numerics (parity golden, CI, fallback) | `core/backends/cpu/` |

Every backend implements the same `IBackend` contract (`core/include/opendlss/backend.h`)
and is verified against the CPU reference numerics: the software fp16 codec, the
E4M3 quantization grid, the attention bit-trick exponentials and the padded-field
geometry are shared, byte-for-byte, by all four implementations.

## What it does

- **Images** — load a PNG, optionally denoise it, run the neural rendering pass
  (resolution-preserving residual + temporal-blend semantics), then upscale 2–4x
  through MetalFX's spatial scaler (or Lanczos3 elsewhere) with adaptive sharpening.
- **Video** — Y4M in, Y4M out (ffmpeg converts to/from mp4, scripts included).
  Every frame gets block-matching motion estimation, history reprojection,
  the 16-lane preprocess with per-frame seeded noise, the neural graph with
  reprojected-history lanes, the head composite against history, then per-frame
  MetalFX scaling. Temporal stability comes from the same blend-logit feedback
  loop the reference network uses.
- **Games** — the real-time path. An engine submits color + depth + motion
  vectors + exposure + jitter per frame (`sdk/include/opendlss_nr_sdk.h`, or the
  `game` CLI command against synthetic frames). On Apple Silicon this drives
  **MTLFXTemporalScaler** — Apple's DLSS-class temporal upscaler — with our
  neural/denoise passes composing around it. Windows/Linux use the same contract
  on our own reprojection scaler.

## Quick start

```bash
git clone <this repo> && cd opendlss-nr-metalfx
scripts/install_macos.sh      # or install_linux.sh / install_windows.ps1
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 tools/make_demo_weights.py --out models/demo-nr   # untrained demo model
./build/opendlss-nr info
./build/opendlss-nr demo --width 480 --height 270 --scale 2 --model models/demo-nr
demos/run_all.sh build/opendlss-nr     # image + video + game + bench demos
```

CLI:

```
opendlss-nr info                      list backends & MetalFX capability
opendlss-nr geometry --width 1920 --height 1080
                                      print the padded field + levels (reference parity)
opendlss-nr image  --in a.png --out b.png [--scale 2] [--model dir]
opendlss-nr video  --in a.y4m --out b.y4m [--scale 2] [--model dir]
opendlss-nr game   [--frames N] [--out f.png]
opendlss-nr bench  [--width W --height H]
```

## The demo model

`tools/make_demo_weights.py` generates a complete, structurally valid model
directory (7 blocks: field → L0 → L1 → ViT → L1 → L0 → field, one global-attention
block, 82 E4M3-packed tensors) with **untrained, near-identity weights** so every
demo, test and CI job exercises the real machinery — preprocess, attention,
transitions, head — with zero proprietary data. Swap in trained weights with the
same manifest layout ([docs/WEIGHTS.md](docs/WEIGHTS.md)) to get real quality;
`tools/export_weights_pytorch.py` shows the export path from a training run.

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | per-platform install, Xcode/SPM, CI |
| [docs/USAGE.md](docs/USAGE.md) | CLI reference, all options, conditioning |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | repo layout, backend contract, data flow |
| [docs/METALFX.md](docs/METALFX.md) | the MetalFX integration in depth |
| [docs/NETWORK.md](docs/NETWORK.md) | the graph: lanes, blocks, transitions, head |
| [docs/WEIGHTS.md](docs/WEIGHTS.md) | manifest + tensor layouts, training/export |
| [docs/GAMES.md](docs/GAMES.md) | engine integration via the C SDK |
| [docs/VIDEO.md](docs/VIDEO.md) | video pipeline, Y4M/ffmpeg workflow |
| [docs/IMAGES.md](docs/IMAGES.md) | image pipeline |
| [demos/README.md](demos/README.md) | what each demo shows, expected artifacts |

## Quality gates

- `tests/test_core.cpp` — 70+ checks: fp16 round-trips, E4M3 grid edges,
  **geometry parity against the reference's documented table**
  (1920x1080 → field 1920x1152, L5 32x20 …), neural-graph determinism and
  composite bounds. Runs under CTest on all three platforms in CI.
- The CPU backend is the parity golden: GPU kernels must match it (the
  numerics helpers are literally the same bit math on every platform).
- AddressSanitizer-clean in development; ASan job runs in CI.

## Performance expectations

The CPU reference executes the demo graph at reference-fidelity speed
(hundreds of ms/frame — it exists for correctness, not speed). The GPU paths
dispatch the same graph as parallel compute: on M1/M2-class hardware the
kernels are sized for the demo network in well under a millisecond of GPU
time, and MetalFX's scalers run in fixed-function-class performance. Full-size
networks (71 blocks, 141 MiB of weights) run when a manifest of that shape is
supplied; the executor is data-driven and hardware cost scales with it.

## Requirements

- macOS 13+ on Apple Silicon (M1/M2/…) with Xcode 15+ — Metal + MetalFX
- Windows 10+ with a DX12 GPU (Feature Level 11_0+) — Visual Studio 2022
- Linux with a Vulkan 1.1+ driver — CMake 3.20+, glslangValidator (or the script)
- C++20 toolchain on all platforms; Python 3 only for tooling/demos

## Not affiliated with NVIDIA or Apple

This project is not affiliated with, endorsed by, or supported by NVIDIA or
Apple. It contains no NVIDIA software, weights, headers, or instructions for
obtaining them, and no Apple SDK code — it links the public Metal/MetalFX
frameworks Apple ships with the OS. You are responsible for the licenses that
apply to whatever model data you use with it.

## License

MIT for everything in this repository ([LICENSE](LICENSE)). Third-party
components are listed in [NOTICE](NOTICE) (vendored stb headers, public domain).

## Credits

- [maanHimself/OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) — the
  reference architecture, numerics specification and geometry rules this project
  ports. Its docs are the ground truth this implementation was written against.
- NVIDIA Research's DLSS 5 report for publishing the network family's design.
- Apple for Metal and MetalFX.
