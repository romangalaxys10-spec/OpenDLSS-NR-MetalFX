#!/usr/bin/env bash
# build.sh — configure + build for the current platform.
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
TYPE="${1:-Release}"
cmake -B build -DCMAKE_BUILD_TYPE="$TYPE" "$@"
cmake --build build -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "binary: build/opendlss-nr"
