# Video

`opendlss video in.mp4 out.mp4 --style 64` decodes with FFmpeg, converts each frame to RGBA
display codes, runs the NR network per frame (production schedule, fresh noise seed per frame),
and encodes with libx264/libx265 (CRF, `slow` preset) at the source frame rate. Audio copies
through. Options: `--start N`, `--frames N`, `--crf N`.

Two workflows:

1. **1:1 re-rendering** (default): the field equals the source resolution. Full detail, full cost.
2. **Low-field + high-proxy**: run the network at a smaller `valid` field while the preprocess
   kernel samples the full-resolution proxy (`Params.validWidth/Height` < source) — the network's
   low-frequency decisions render at low cost while the proxy carries the detail. The composition
   rule then behaves like a detail transfer. This is the same trick the game path uses, offline.

Frames are processed sequentially; parallelism is per-session across worker processes if you split
the clip (see `--start/--frames`).
