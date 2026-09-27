# Usage

## Commands

### `info`
Lists every backend compiled into the binary with its device, MetalFX
capability flags and numerics support. Use this first to confirm the GPU path
is alive:

```
$ opendlss-nr info
OpenDLSS-NR MetalFX 1.0.0
backends compiled into this build:
  cpu      device: host              neural:1 fp16:1 e4m3:1
  metal    device: Apple M2 Pro      MetalFX[s:1 t:1] neural:1 fp16:1 e4m3:1
```

### `geometry --width W --height H [--levels 6] [--min-field 320]`
Prints the padded field and every level size — the same computation the
reference documents, used as a parity tool:

```
$ opendlss-nr geometry --width 1920 --height 1080
valid 1920x1080 -> field 1920x1152, L0 960x576, L1 480x288, L2 240x144, L3 120x72, L4 60x36, L5 32x20
```

### `image --in a.png --out b.png [options]`
Denoise → neural rendering pass (when a model is loaded) → scale → sharpen.

| Option | Default | Meaning |
| --- | --- | --- |
| `--scale N` | 2 | output scale 1–4 |
| `--model dir` | — | model directory (manifest.json) |
| `--backend X` | preferred | cpu / metal / d3d12 / vulkan |
| `--denoise F` | 0.6 | analytical bilateral strength 0..1 |
| `--sharpen F` | 0.25 | adaptive sharpen 0..1 |
| `--no-neural` | — | skip the neural graph (analytical only) |
| `--no-metalfx` | — | force the in-house Lanczos path on macOS |
| `--style/--tone/--structure/--skin/--automask F` | see below | conditioning |
| `--seed N` | fixed | noise seed base |

### `video --in a.y4m --out b.y4m [options]`
Full temporal pipeline per frame: motion estimate → reproject history →
preprocess (history lanes) → neural graph → head composite against history →
scale → write. Feedback loop keeps temporal stability. `--frames N` caps the
frame count. Y4M is the interchange format; convert with ffmpeg:

```bash
ffmpeg -i input.mp4 -pix_fmt yuv420p -f yuv4mpegpipe input.y4m
opendlss-nr video --in input.y4m --out clean.y4m --scale 2 --model models/demo-nr
ffmpeg -i clean.y4m -pix_fmt yuv420p clean.mp4
```

### `game [--frames N] [--render-width W] [--render-height H] [--out f.png]`
Runs the real-time path on a synthetic scrolling scene with exact camera
motion — the same code path an engine drives through the C SDK. The last
resolved frame is written as a PNG for inspection.

### `bench [--width W --height H --frames N]`
Measures denoise + upscale wall time per frame on the selected backend.

## Conditioning scalars

The network family conditions on five scalars packed into the feature lanes:

| Scalar | Lane | Effect |
| --- | --- | --- |
| `--style` | 10 | style id / 128; selects the rendering style embedding |
| `--tone` | 11 | local tone bias |
| `--structure` | 12 | structure/detail strength |
| `--skin` | 13 | skin rendering weight |
| `--automask` | 14 | auto-exposure/mask conditioning |

Defaults are neutral (0.5, style 0). Trained models respond to these; the
untrained demo model ignores them by construction.

## Environment variables

| Variable | Platforms | Meaning |
| --- | --- | --- |
| `OPENDLSS_VK_SHADER_DIR` | Linux | where the compiled `.spv` files live |
| `OPENDLSS_D3D_SHADER_DIR` | Windows | where the `.hlsl` files live |
| `OPENDLSS_TEST_MODEL` | tests | model dir for the test suite |

## Exit codes

0 success · 1 runtime failure (logged) · 2 usage error.
