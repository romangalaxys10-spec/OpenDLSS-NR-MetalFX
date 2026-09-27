# Demos

Run everything: `demos/run_all.sh build/opendlss-nr` (artifacts land in
`demos/output/`, git-ignored except committed samples).

## 1. Image — `../apps/cli demo`

A synthetic scene (sky gradient, sun, ridges, water) is rendered clean,
downscaled 2x, polluted with Gaussian sensor noise, then run through the full
image pipeline: bilateral denoise → neural rendering graph (demo model) →
MetalFX/Lanczos 2x upscale → adaptive sharpen. Output: `image_demo.png` plus
the noisy input for A/B.

```bash
opendlss-nr demo --width 480 --height 270 --scale 2 --model models/demo-nr
```

## 2. Video — `video/generate_video.py` + `opendlss-nr video`

A 24 fps procedural clip (scrolling terrain, moving light, temporal noise) is
generated as Y4M at half resolution and processed frame-by-frame through the
temporal pipeline: motion estimate → reprojection → preprocess with history
lanes → neural graph → head composite (feedback loop) → upscale. Output:
`video_clean.y4m` (+ mp4 copies when ffmpeg is installed).

```bash
python3 demos/video/generate_video.py --out noise_input.y4m
opendlss-nr video --in noise_input.y4m --out video_clean.y4m --scale 2 \
    --model models/demo-nr
```

## 3. Game — `opendlss-nr game` and `game/engine_example.c`

The real-time path: a synthetic "engine" renders a scrolling scene at render
resolution with exact camera motion vectors and feeds color + motion through
the temporal scaler (MetalFX's MTLFXTemporalScaler on Apple Silicon) every
frame. The CLI variant writes the final resolved frame; the C example shows
the exact calls an engine makes.

```bash
opendlss-nr game --frames 120 --scale 2 --model models/demo-nr --out game_frame.png
cc demos/game/engine_example.c -I sdk/include -L build -lopendlss_nr_sdk -lm -o engine_example
```

## 4. Viewer (macOS) — `../platforms/macos/viewer/OpenDLSSViewer`

A SwiftUI + MetalKit app rendering the synthetic scene live through the MetalFX
temporal scaler with an on-screen frame-time/status bar.

```bash
cd platforms/macos/viewer/OpenDLSSViewer && swift run
```

## 5. Bench — `opendlss-nr bench`

Times the denoise + upscale path per frame at a chosen resolution — the quick
sanity check that a GPU backend is actually faster than the CPU reference.

## Expected artifacts (committed samples)

`demos/samples/` holds small committed outputs from CI (image demo before/after,
one game frame) so you can see the pipeline's effect without building anything.
