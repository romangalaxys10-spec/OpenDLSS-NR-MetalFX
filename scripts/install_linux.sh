#!/usr/bin/env bash
# install_linux.sh — one-command setup on Linux (Vulkan path).
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "== OpenDLSS-NR MetalFX — Linux installer =="

# 1. packages (best effort across distros)
PKGS="cmake g++ python3 glslang-tools"
if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq $PKGS || \
            echo "warning: package install failed; ensure cmake/g++/glslangValidator exist"
    fi
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y cmake gcc-c++ python3 glslang || true
elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --noconfirm cmake gcc python glslang || true
else
    echo "note: unknown package manager — ensure cmake, g++, python3, glslangValidator"
fi

# 2. build
cd "$HERE"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"

# 3. demo model
python3 tools/make_demo_weights.py --out models/demo-nr

# 4. smoke test (CPU always; Vulkan appears when a driver + loader exist)
./build/opendlss-nr info
ctest --test-dir build --output-on-failure

echo
echo "done. try:"
echo "  ./build/opendlss-nr demo --width 480 --height 270 --scale 2 --model models/demo-nr"
echo "  demos/run_all.sh build/opendlss-nr"
