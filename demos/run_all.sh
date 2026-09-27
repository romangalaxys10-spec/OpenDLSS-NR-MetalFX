#!/usr/bin/env bash
# run_all.sh — run every demo end to end against the built CLI.
# Usage: demos/run_all.sh [path/to/opendlss-nr]
# Artifacts land in demos/output/ (git-ignored except the committed samples).
#
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${1:-$HERE/../build/opendlss-nr}"
OUT="$HERE/output"
mkdir -p "$OUT"

echo "== OpenDLSS-NR MetalFX demos =="
"$BIN" info

echo
echo "== demo 1: image (synthetic scene, denoise + neural + 2x upscale) =="
"$BIN" demo --width 480 --height 270 --out "$OUT/image_demo.png" --scale 2 \
      --model "$HERE/../models/demo-nr"

echo
echo "== demo 2: video (48 frames, temporal pipeline + 2x upscale) =="
python3 "$HERE/video/generate_video.py" --out "$OUT/noise_input.y4m" \
        --width 320 --height 180 --frames 48 --fps 24 --sigma 0.07
"$BIN" video --in "$OUT/noise_input.y4m" --out "$OUT/video_clean.y4m" \
      --scale 2 --model "$HERE/../models/demo-nr"

echo
echo "== demo 3: game (real-time temporal path, 120 frames) =="
"$BIN" game --frames 120 --out "$OUT/game_frame.png" --scale 2 \
      --model "$HERE/../models/demo-nr"

echo
echo "== demo 4: bench =="
"$BIN" bench --width 640 --height 360 --frames 5 --scale 2

echo
echo "artifacts in $OUT:"
ls -la "$OUT"

# Optional: mp4 conversion when ffmpeg is present
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -i "$OUT/noise_input.y4m" -pix_fmt yuv420p "$OUT/noise_input.mp4"
    ffmpeg -y -loglevel error -i "$OUT/video_clean.y4m" -pix_fmt yuv420p "$OUT/video_clean.mp4"
    echo "mp4 copies written (noise_input.mp4, video_clean.mp4)"
else
    echo "(install ffmpeg to get mp4 copies of the video demo)"
fi
