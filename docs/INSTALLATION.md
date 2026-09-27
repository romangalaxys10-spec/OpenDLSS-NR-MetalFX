# Installation

## macOS — Apple Silicon (M1, M2 and later) — the MetalFX path

Requirements: macOS 13.0 (Ventura) or newer on Apple Silicon, Xcode 15+ (or the
Command Line Tools), CMake 3.20+, Python 3.10+.

```bash
# one-time
xcode-select --install          # if you don't have the toolchain
brew install cmake              # or use scripts/install_macos.sh

# clone + build
git clone <this repo> && cd opendlss-nr-metalfx
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# demo model (untrained, structurally complete)
python3 tools/make_demo_weights.py --out models/demo-nr

# run
./build/opendlss-nr info        # should list metal+metalfx with M-series device
./build/opendlss-nr demo --width 480 --height 270 --scale 2 --model models/demo-nr
```

`scripts/install_macos.sh` automates all of the above (it also checks that the
Mac is Apple Silicon and macOS ≥ 13, where MetalFX lives).

MetalFX notes:
- `MTLFXSpatialScaler` (images, video frames): macOS 13+, all Apple Silicon.
- `MTLFXTemporalScaler` (games): macOS 13+, Apple7 family and later
  (M1 and later are fine). The backend reports both in `opendlss-nr info`
  and falls back to in-house kernels when either is unavailable.

### The SwiftUI viewer

```bash
cd platforms/macos/viewer/OpenDLSSViewer
swift run
```

### Framework link notes

The backend links Metal, MetalFX, Foundation, AppKit and MetalKit — all system
frameworks, nothing to download. Metal shaders compile from
`platforms/macos/shaders/*.metal` at runtime (and via the default library when
one is bundled), so there is no shader build step.

## Windows 10/11

Requirements: Visual Studio 2022 (or Build Tools) with the C++ desktop
workload, CMake 3.20+, a DX12 GPU (Feature Level 11_0+; WARP fallback works
but is slow), optionally DirectML (ships with Windows 10 1903+).

```powershell
git clone <this repo> && cd opendlss-nr-metalfx
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
python tools/make_demo_weights.py --out models/demo-nr
build\opendlss-nr.exe info
build\opendlss-nr.exe demo --width 480 --height 270 --scale 2 --model models\demo-nr
```

`scripts/install_windows.ps1` does the same and locates VS via vswhere.
HLSL shaders compile at runtime from `shaders_hlsl/` (CMake copies them next
to the executable); the shader directory can be overridden with
`OPENDLSS_D3D_SHADER_DIR`.

## Linux

Requirements: a Vulkan 1.1+ driver (NVIDIA/AMD/Intel), CMake 3.20+, a C++20
compiler, `glslangValidator` (package `glslang-tools` on Debian/Ubuntu,
`glslang` on Arch/Fedora), Python 3.

```bash
sudo apt install cmake g++ glslang-tools          # Debian/Ubuntu example
git clone <this repo> && cd opendlss-nr-metalfx
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 tools/make_demo_weights.py --out models/demo-nr
./build/opendlss-nr info
```

GLSL → SPIR-V compilation happens at build time into `build/shaders_spv/`
and is copied next to the binary; override the search location with
`OPENDLSS_VK_SHADER_DIR`. The Vulkan loader is loaded dynamically
(`libvulkan.so.1`) — no SDK headers are needed to build.

## Every platform

- Tests: `ctest --test-dir build` (or run `build/test_core` directly with
  `OPENDLSS_TEST_MODEL=models/demo-nr`).
- Demos: `demos/run_all.sh build/opendlss-nr` (Windows: run the same commands
  individually; see `demos/README.md`).
- ffmpeg (optional) converts the Y4M demo videos to mp4.

## CI

`.github/workflows/ci.yml` builds and tests on:
- `macos-14` (Apple Silicon runner — Metal + MetalFX compile for real),
- `ubuntu-22.04` (core + CPU + Vulkan shader compilation),
- `windows-2022` (core + CPU + D3D12),
- plus an AddressSanitizer job on Linux.
