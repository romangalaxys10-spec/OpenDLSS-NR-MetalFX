#!/usr/bin/env bash
# build.sh - macOS build (Apple silicon, Metal + MetalFX, macOS 13+).
#   scripts/macos/build.sh           # release build of everything
#   scripts/macos/build.sh demo      # + the realtime demo app bundle
#   scripts/macos/build.sh metallib  # precompile the Metal shaders
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:-all}"
BUILD=build
cmake -B $BUILD -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build $BUILD -j "$(sysctl -n hw.ncpu)"

if [ "$MODE" = "metallib" ] || [ "$MODE" = "all" ]; then
  xcrun -sdk macosx metal -c src/metal/shaders/*.metal -o $BUILD/opendlss.air
  xcrun -sdk macosx metallib -o $BUILD/opendlss.metallib $BUILD/opendlss.air
  echo "built $BUILD/opendlss.metallib"
fi
if [ "$MODE" = "demo" ] || [ "$MODE" = "all" ]; then
  APP=$BUILD/demo-game-macos.app
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>demo-game-macos</string>
  <key>CFBundleIdentifier</key><string>org.opendlss.nr-realtime</string>
  <key>CFBundleName</key><string>OpenDLSS NR Realtime</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
  cp $BUILD/demo-game-macos "$APP/Contents/MacOS/"
  echo "demo bundle: $APP"
fi
echo "build complete: $BUILD/"
