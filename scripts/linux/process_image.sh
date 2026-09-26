#!/usr/bin/env bash
# process_image.sh - NR-process one image through the Vulkan backend.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x build/opendlss-cli ] || scripts/linux/build.sh
exec build/opendlss-cli image "$@"
