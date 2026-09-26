#!/usr/bin/env bash
# make_pkg.sh - macOS installer package (productbuild).
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=$(mktemp -d)/root
mkdir -p "$STAGE/usr/local/bin" "$STAGE/usr/local/lib" "$STAGE/usr/local/include/opendlss" "$STAGE/usr/local/share/opendlss"
cp build/opendlss-cli "$STAGE/usr/local/bin/opendlss"
cp build/libopendlss*dylib "$STAGE/usr/local/lib/" 2>/dev/null || true
cp include/opendlss/opendlss.h "$STAGE/usr/local/include/opendlss/"
[ -f build/opendlss.metallib ] && cp build/opendlss.metallib "$STAGE/usr/local/share/opendlss/"
mkdir -p build/pkg
xcrun pkgbuild --root "$(dirname "$STAGE")" --identifier org.opendlss.nr --version 1.0.0 --install-location / build/pkg/opendlss-macos.pkg
echo "package: build/pkg/opendlss-macos.pkg"
