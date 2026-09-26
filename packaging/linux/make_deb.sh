#!/usr/bin/env bash
# make_deb.sh - Debian package for the Vulkan backend.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=$(mktemp -d)
mkdir -p "$STAGE/usr/local/bin" "$STAGE/usr/local/lib" "$STAGE/usr/local/include/opendlss" "$STAGE/usr/local/share/opendlss/shaders" "$STAGE/DEBIAN"
cp build/opendlss-cli "$STAGE/usr/local/bin/opendlss"
cp build/dlss5vk "$STAGE/usr/local/bin/" 2>/dev/null || true
cp build/libopendlss.so* "$STAGE/usr/local/lib/" 2>/dev/null || true
cp include/opendlss/opendlss.h "$STAGE/usr/local/include/opendlss/"
cp build/shaders/*.spv "$STAGE/usr/local/share/opendlss/shaders/" 2>/dev/null || true
cat > "$STAGE/DEBIAN/control" <<CTRL
Package: opendlss
Version: 1.0.0
Section: utils
Priority: optional
Architecture: $(dpkg --print-architecture)
Maintainer: OpenDLSS-NR-MetalFX contributors
Description: Neural rendering (DLSS 5 NR architecture) - Vulkan backend
 Neural re-rendering of game/video/image frames; Linux build of OpenDLSS-NR-MetalFX.
CTRL
dpkg-deb --build "$STAGE" build/opendlss_1.0.0_$(dpkg --print-architecture).deb
echo "package: build/opendlss_1.0.0_$(dpkg --print-architecture).deb"
