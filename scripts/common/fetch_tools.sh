#!/usr/bin/env bash
# fetch_tools.sh - portable Vulkan toolchain for the Windows/Linux backend
# (glslang for SPIR-V, Vulkan-Headers, volk; mirrors scripts/windows/fetch_tools.ps1).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools && cd tools
fetch() { local url=$1 dir=$2; if [ ! -d "$dir" ]; then echo "fetching $dir"; (mkdir -p "$dir" && cd "$dir" && curl -sL "$url" | tar xz --strip-components=1); fi; }
fetch "https://github.com/KhronosGroup/glslang/releases/download/main-tot/glslang-main-linux-x86_64-Release.tar.gz" glslang || fetch "https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-linux-x86_64-Release.tar.gz" glslang
fetch "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.363.tar.gz" Vulkan-Headers
if [ ! -d volk ]; then git clone --depth 1 https://github.com/zeux/volk.git volk; fi
echo "tools ready: $(ls)"
