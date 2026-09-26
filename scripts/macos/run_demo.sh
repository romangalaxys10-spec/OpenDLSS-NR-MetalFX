#!/usr/bin/env bash
# run_demo.sh - launch the realtime demo (expects a model dir, see docs/INSTALL.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL="${OPENDLSS_MODEL:-models/nr}"
export OPENDLSS_MODEL="$MODEL"
[ -f build/demo-game-macos.app/Contents/MacOS/demo-game-macos ] || scripts/macos/build.sh demo
open build/demo-game-macos.app
