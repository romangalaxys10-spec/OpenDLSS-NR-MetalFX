#!/usr/bin/env bash
# install_macos.sh — one-command setup on Apple Silicon macOS.
# Verifies the platform (Apple Silicon + macOS 13+ for MetalFX), installs
# CMake if needed, builds the project, generates the demo model and runs the
# smoke test.
#
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "== OpenDLSS-NR MetalFX — macOS installer =="

# 1. platform check
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: this script is for macOS; use install_linux.sh there" >&2
    exit 1
fi
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "warning: MetalFX targets Apple Silicon (arm64); this Mac reports $ARCH" >&2
    echo "         the build will still work (Metal falls back, MetalFX off)" >&2
fi
MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if (( MAJOR < 13 )); then
    echo "error: MetalFX requires macOS 13 (Ventura) or newer; found $(sw_vers -productVersion)" >&2
    exit 1
fi
echo "  macOS $(sw_vers -productVersion) on $ARCH — MetalFX-capable"

# 2. toolchain
if ! command -v cmake >/dev/null 2>&1; then
    echo "  cmake missing — installing via Homebrew (or install it manually)"
    if ! command -v brew >/dev/null 2>&1; then
        echo "error: Homebrew not found. Install from https://brew.sh and retry" >&2
        exit 1
    fi
    brew install cmake
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 required (part of Xcode CLT / developer tools)" >&2
    exit 1
fi

# 3. build
cd "$HERE"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(sysctl -n hw.ncpu)"

# 4. demo model
python3 tools/make_demo_weights.py --out models/demo-nr

# 5. smoke test
./build/opendlss-nr info
ctest --test-dir build --output-on-failure

echo
echo "done. try:"
echo "  ./build/opendlss-nr demo --width 480 --height 270 --scale 2 --model models/demo-nr"
echo "  demos/run_all.sh build/opendlss-nr"
