# Install

## 1. The model

The network is not this repository's to ship: **you supply the weights** as a model directory
(`manifest.json` + `model/*.stage`), in exactly the layout the original project defines
([original-design/weights.md](original-design/weights.md)). By default every tool looks at
`models/nr` next to the binary (or `--model <dir>`, or `OPENDLSS_MODEL`). The tools verify every
stage's SHA-256 at load and refuse a different block count than 71.

## 2. Packages

| platform | package | install |
| --- | --- | --- |
| macOS | `OpenDLSS-NR-MetalFX` release zip (universal ops) / Homebrew formula (`packaging/macos/opendlss.rb`) | unzip to /usr/local; `brew install ./opendlss.rb` |
| Windows | release zip + Inno Setup installer (`packaging/windows/opendlss.iss`) | run the installer or unzip; add `bin` to PATH |
| Linux | `.deb` (`packaging/linux/make_deb.sh`), AppImage (`packaging/linux/make_appimage.sh`) | `sudo dpkg -i opendlss_1.0.0_arm64.deb` |

Contents: `bin/opendlss` (CLI), `bin/dlss5vk` (Windows/Linux parity tool), `lib/libopendlss`
(the C SDK), `include/opendlss/opendlss.h`, `share/opendlss/shaders/*.spv` (Vulkan), and on macOS
`share/opendlss/opendlss.metallib` + `demo-game-macos.app`.

## 3. Dependencies

- macOS: macOS 13+ (MetalFX spatial) / 14+ (temporal), Apple silicon (M1 or newer), Xcode 15
  toolchain. Nothing else.
- Windows: NVIDIA Ada (or newer) exposing `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`,
  `VK_EXT_shader_float8` (+ `VK_NV_cuda_kernel_launch` for the PTX fast path); VS 2022 Build Tools;
  `scripts/windows/fetch_tools.ps1` fills `tools/` (glslang, Vulkan-Headers, volk, CMake, Ninja).
- Linux: Vulkan 1.3 + `VK_KHR_cooperative_matrix` (tested families: NVIDIA Ada, RDNA3);
  `scripts/common/fetch_tools.sh`; FFmpeg dev packages for the video pipeline
  (`libavformat-dev libavcodec-dev libswscale-dev libavutil-dev`).
