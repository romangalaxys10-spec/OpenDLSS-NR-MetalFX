#!/usr/bin/env bash
# process_video.sh - NR-process a video through the Metal backend (FFmpeg).
#   scripts/macos/process_video.sh input.mp4 output.mp4 [--style 64]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x build/opendlss-cli ] || scripts/macos/build.sh
exec build/opendlss-cli video "$@"
