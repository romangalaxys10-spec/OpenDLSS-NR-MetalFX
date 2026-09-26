#!/usr/bin/env bash
# build.sh - Linux build (Vulkan backend). Runs fetch_tools.sh when tools/ is empty.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -d tools/Vulkan-Headers ] || scripts/common/fetch_tools.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
# Compile the GLSL kernels to SPIR-V for runtime loading (dlss5vk + CLI).
mkdir -p build/shaders
for f in shaders/vulkan-glsl/*.comp; do
  tools/glslang/glslangValidator -V --target-env vulkan1.3 -Ishaders/vulkan-glsl "$f" -o "build/shaders/$(basename "${f%.comp}").spv"
done
echo "build complete: build/ (opendlss, dlss5vk, opendlss-tests)"
