#!/usr/bin/env bash
# make_appimage.sh - portable AppImage (needs linuxdeploy; optional).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x linuxdeploy ] || curl -sL -o linuxdeploy https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-$(uname -m).AppImage && chmod +x linuxdeploy
mkdir -p AppDir/usr/bin AppDir/usr/share/opendlss/shaders
cp build/opendlss-cli AppDir/usr/bin/opendlss
cp build/shaders/*.spv AppDir/usr/share/opendlss/shaders/ 2>/dev/null || true
./linuxdeploy --appdir AppDir -e AppDir/usr/bin/opendlss -d /dev/null -o appimage
echo "AppImage ready"
