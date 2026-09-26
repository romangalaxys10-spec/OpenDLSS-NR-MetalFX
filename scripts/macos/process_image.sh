#!/usr/bin/env bash
# process_image.sh - NR-process one image through the Metal backend.
#   scripts/macos/process_image.sh input.png output.png [--style 64] [--grain 7]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x build/opendlss-cli ] || scripts/macos/build.sh
exec build/opendlss-cli image "$@"
