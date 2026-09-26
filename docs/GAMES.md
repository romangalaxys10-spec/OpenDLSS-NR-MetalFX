# Games: engine integration

Three levels, pick by appetite:

## 1. C SDK (any engine, any platform)

```c
#include <opendlss/opendlss.h>
OpendlssParams p = {0};
p.sourceWidth = p.sourceHeight = 1440;          /* the frame your engine drew */
OpendlssSession* s = opendlssSessionCreate("models/nr", &p, &reason);
/* per frame: */
opendlssProcessRgba(s, frameRgba, frameStride, outRgba, outStride, &stats);
```

The network consumes display-code-value RGBA and returns composed RGBA8. Wire the per-frame noise
seed to your frame index for stable grain. Thread rule: one session per rendering thread.

## 2. Metal-native (Apple silicon)

Drive `nr::metal::*` directly (see `apps/demo-game-macos/main.mm`): render your frame into an
MTLTexture, run `MetalFxTemporal`/`MetalFxSpatial` (`encodeToCommandBuffer:`), then the NR graph on
the result. The v1.0 demo reads back through shared memory for clarity; keeping the loop on-GPU
(MLTexture proxy variant of `Session::processFrame`) is the v1.1 work item.

## 3. The engine contract (what a full integration adds)

- motion vectors (RG16Float, world-space per pixel) and depth (R32Float) if you want the temporal
  scaler path;
- history reprojection feeding the network's temporal lanes (the original demo's
  `docs/original-design/frame.md` describes the feedback loop);
- UI/tenancy: run the NR pass before tonemapping UI, on the scene only.
