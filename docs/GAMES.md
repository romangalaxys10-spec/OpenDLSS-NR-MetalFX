# Games — engine integration

The game path is a real-time temporal pipeline driven by the engine's own
frame data. Integration is one header, five calls.

## The C SDK

```c
#include "opendlss_nr_sdk.h"

OpendlssNrCreateDesc d = {0};
d.renderWidth = 1280;  d.renderHeight = 720;
d.outputScale = 2;                    // → 2560x1440 output
d.backend = "auto";                   // metal on macOS / d3d12 / vulkan / cpu
d.modelDir = NULL;                    // NULL = analytical; "models/my-nr" = neural
d.denoiseStrength = 0.5f; d.sharpen = 0.2f;

OpendlssNrInstance* inst;
opendlss_nr_create(&d, &inst);

// each frame:
OpendlssNrFrameInput in = {0};
in.colorRGBA  = myRenderedColor;      // render-res float RGBA, 0..1
in.depth      = myLinearDepth;        // render-res float, 0..1
in.motionXY   = myMotionVectors;      // render-res float2, prev→curr pixels
in.exposure   = myPreExposure;
in.jitterX    = jitterPattern[frame % 8];  // if the engine jitters, report it
in.resetHistory = cameraCutThisFrame;
OpendlssNrFrameOutput out = {0};
out.colorRGBA = myOutputBuffer;       // output-res, caller-allocated
opendlss_nr_resolve(inst, &in, &out);
// out.outputWidth/Height, out.gpuMilliseconds for on-screen stats

opendlss_nr_destroy(inst);
```

A complete runnable example lives at `demos/game/engine_example.c`.

## What each backend does with the frame

| Backend | Resolution change | Temporal accumulation |
| --- | --- | --- |
| macOS (Apple Silicon) | **MTLFXTemporalScaler** (fixed-function-class) | scaler-owned feedback, exposure-aware |
| Windows (D3D12) | Lanczos3 compute | our reprojection + confidence blend |
| Linux (Vulkan) | Lanczos3 compute | our reprojection + confidence blend |
| CPU (reference) | Lanczos3 | our reprojection + confidence blend |

## Engine checklist

- Render at the lower resolution with the jitter pattern applied to the
  projection matrix (2-phase or 8-phase halton both work with MetalFX).
- Motion vectors: prev→current, in *pixels*, at render resolution, including
  camera and per-object motion. MetalFX expects RG16Float; the SDK accepts
  float2 and converts.
- Depth: linear 0..1 at render resolution (reverse-Z engines: invert).
- Exposure: the same pre-exposure factor the engine tone-maps with.
- Camera cuts / resolution changes: set `resetHistory` to avoid ghosting.
- Alpha: fill 1.0; the pipeline outputs opaque color.

## Try it without an engine

```
opendlss-nr game --frames 120 --render-width 640 --render-height 360 \
                 --scale 2 --model models/demo-nr --out last_frame.png
```

runs the same path against a synthetic scrolling scene with exact camera
motion and writes the final resolved frame.
