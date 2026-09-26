# Building

CMake >= 3.24, one `CMakeLists.txt` for all platforms.

| target | what |
| --- | --- |
| `opendlss` | shared library: the C SDK + frame pipeline, linked to the platform backend |
| `opendlss-cli` | the `opendlss` tool (image/video/bench/profile/devices/selftest) |
| `dlss5vk` | Windows/Linux: the original parity/verify/bench tool (unchanged) |
| `demo-game-macos` | the realtime MTKView demo |
| `opendlss-tests` | the numeric/reference/geometry suite |

Options: `OPENDDLSS_WITH_VIDEO` (default ON; needs FFmpeg pkg-config), `OPENDDLSS_BUILD_TESTS`,
`OPENDDLSS_BUILD_DEMO`. Platform specifics:

- **macOS**: `scripts/macos/build.sh` wraps `cmake -B build -DCMAKE_OSX_ARCHITECTURES=arm64` plus
  the metal compile step (`xcrun metal -c src/metal/shaders/*.metal` -> `metallib`). The runtime
  also falls back to compiling the .metal sources in-process when no metallib is next to the binary.
- **Windows**: `scripts/windows/build.ps1` (VS 2022 x64). The Vulkan tree loads `.spv` files compiled
  by glslang (scripts do it; `DLSS5VK`-style env `OPENDLSS_SHADERS` points at the directory).
- **Linux**: `scripts/linux/build.sh` fetches the portable toolchain into `tools/` when needed.

CI (`.github/workflows/ci.yml`) builds all three and runs the tests + `selftest`; the macOS job
additionally compiles every MSL kernel with the real `metal` compiler, which is the shader build gate.
