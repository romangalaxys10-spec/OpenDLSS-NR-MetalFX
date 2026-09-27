# Video

The video pipeline processes every frame through the full temporal chain and
writes standard Y4M — no proprietary containers, converts to/from mp4 with one
ffmpeg command.

## Pipeline per frame

1. **Motion estimation** — 16×16 block matching, ±6 px search, on luma
   (engine/video-grade: on macOS this kernel can be replaced by decoder-side
   motion when available).
2. **Reprojection** — the previous *output* frame is warped by the motion
   field with bilinear filtering and a disocclusion confidence channel.
3. **Preprocess** — the 16-lane packing with the reprojected history in lanes
   7–9 and a per-frame seeded noise field (temporal noise, not per-frame reset,
   so the network's blend logit can accumulate).
4. **Neural graph** (model loaded) or the analytical path: confidence-guided
   temporal blend + bilateral spatial denoise.
5. **Head composite** — residual + sigmoid-logit history blend, which closes
   the feedback loop: the output frame becomes the next frame's history.
6. **MetalFX spatial scaling** + adaptive sharpen → output frame.

## Commands

```bash
# generate a synthetic noisy clip (no assets needed)
python3 demos/video/generate_video.py --out noise.y4m --width 320 --height 180 \
        --frames 48 --fps 24 --sigma 0.07

# process it
opendlss-nr video --in noise.y4m --out clean.y4m --scale 2 --model models/demo-nr

# convert to mp4 (optional)
ffmpeg -i noise.y4m -pix_fmt yuv420p noise.mp4
ffmpeg -i clean.y4m -pix_fmt yuv420p clean.mp4
```

## Real footage

```bash
ffmpeg -i input.mp4 -pix_fmt yuv420p -f yuv4mpegpipe input.y4m
opendlss-nr video --in input.y4m --out output.y4m --scale 2 \
    --model models/demo-nr --denoise 0.5 --max-blend 0.85
ffmpeg -i output.y4m -pix_fmt yuv420p output.mp4
```

| Flag | Guidance |
| --- | --- |
| `--denoise` | 0.4–0.7 for sensor noise; 0 to rely on the network alone |
| `--max-blend` | caps history weight; lower it after cuts or for fast motion |
| `--scale` | 1 keeps resolution (denoise/neural only) — useful for grading |
| `--frames` | cap the frame count for quick tests |

## Notes

- Y4M 4:2:0 (C420jpeg) is the interchange chroma; conversion is BT.601
  limited-range, the same convention ffmpeg uses by default for y4m.
- The pipeline is deterministic given `--seed`: identical inputs produce
  identical outputs (useful for A/B comparisons and regression tests).
- Long clips: the CLI is stream-oriented; memory stays bounded by two frames
  (history + current) plus the model.
