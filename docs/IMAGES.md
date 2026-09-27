# Images

The image pipeline is the simplest end-to-end path and a good first run:
load → denoise → neural rendering pass → MetalFX upscale → sharpen → save.

## Commands

```bash
# fully self-contained synthetic demo (no input file needed)
opendlss-nr demo --width 480 --height 270 --scale 2 \
    --model models/demo-nr --out demo_output.png

# your own image
opendlss-nr image --in photo.png --out photo_4k.png --scale 2 \
    --model models/demo-nr

# analytical only (no model): denoise + MetalFX-class upscale + sharpen
opendlss-nr image --in photo.png --out photo_4k.png --scale 3 --no-neural

# neural rendering pass at native resolution (no scaling): the DLSS-NR role
opendlss-nr image --in render.png --out rendered.png --scale 1 \
    --model models/demo-nr
```

## What each stage contributes

| Stage | Default | Effect |
| --- | --- | --- |
| bilateral denoise | 0.6 | edge-aware noise suppression, luma-guided |
| neural graph | on with a model | the reference's residual + blend semantics (needs weights to shine; the demo model is near-identity) |
| spatial upscale | 2x | MetalFX spatial scaler on Apple Silicon, Lanczos3 elsewhere |
| adaptive sharpen | 0.25 | CAS-style, suppressed across strong edges to avoid halos |

## Geometry notes

Images drive the same padded-field geometry as everything else
(`opendlss-nr geometry --width W --height H` previews it). Inputs smaller than
the model's floor (320 for reference-shaped models, 64 for the demo model) are
rejected by design — the decoder needs whole windows. Scale factors 1–4 are
supported; larger factors chain two passes.

## Batching

For folders, loop the CLI (it is a zero-state process per invocation):

```bash
for f in renders/*.png; do
  opendlss-nr image --in "$f" --out "upscaled/$(basename "$f")" \
      --scale 2 --model models/demo-nr
done
```
