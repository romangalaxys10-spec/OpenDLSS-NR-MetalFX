# The MetalFX integration

MetalFX is Apple's upscaling framework, shipped with macOS 13+ on Apple
Silicon. It provides two scaler objects; this project uses both, each where it
is the right tool, with in-house fallbacks so nothing hard-fails.

## MTLFXSpatialScaler — images and video frames

A fixed-function-class spatial upscaler with sharpening (the NIS-class role).
Used by the `image` and `video` commands for the resolution change.

```objc
MTLFXSpatialScalerDescriptor* d = [MTLFXSpatialScalerDescriptor new];
d.inputWidth = iw; d.inputHeight = ih;
d.outputWidth = ow; d.outputHeight = oh;
d.colorTextureFormat = MTLPixelFormatRGBA16Float;
d.outputTextureFormat = MTLPixelFormatRGBA16Float;
d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModeSRGB;
id<MTLFXSpatialScaler> scaler = [d newSpatialScalerWithDevice:device];

scaler.colorTexture   = srcTexture;
scaler.outputTexture  = dstTexture;
[scaler encodeTransformToCommandBuffer:cb];   // once per format/size change
[scaler encodeColorToCommandBuffer:cb];       // every frame
```

Scalers are cached per `(in→out)` size pair in `MetalBackend::spatialScaler`.
The color path uses RGBA16Float textures; the host converts RGBA32Float
pipeline images on upload (`uploadImageHalf`) and back on read
(`readTextureHalf`). When MetalFX is unavailable (`--no-metalfx`, or a
non-Apple7 device), the same op runs our separable Lanczos3 kernels
(`odl_upscale_h` / `odl_upscale_v`) followed by the adaptive sharpen pass —
visibly close, measurably different, documented honestly.

## MTLFXTemporalScaler — games

Apple's DLSS-class temporal upscaler: it accumulates jittered renders over
time, reprojects history with engine motion vectors, and applies exposure-
aware anti-aliasing at output resolution. The `game` command and the C SDK
drive it with exactly the inputs an engine already produces:

```objc
MTLFXTemporalScalerDescriptor* d = [MTLFXTemporalScalerDescriptor new];
d.inputWidth = renderW;  d.inputHeight = renderH;
d.outputWidth = outW;    d.outputHeight = outH;
d.colorTextureFormat = MTLPixelFormatRGBA16Float;
d.depthTextureFormat = MTLPixelFormatR32Float;
d.motionVectorTextureFormat = MTLPixelFormatRG16Float;
d.supportsFeedback = YES;                 // scaler-owned history buffer
d.isAutoExposureEnabled = NO;             // engines hand us exposure
id<MTLFXTemporalScaler> ts = [d newTemporalScalerWithDevice:device];

// per frame:
ts.colorTexture = color;  ts.depthTexture = depth;
ts.motionVectorTexture = motion;
ts.outputTexture = output;
ts.exposureFactor = exposure;
ts.reset = cameraCut;                     // clears internal history
[ts encodeToCommandBuffer:cb];
```

Contract notes for engine integrators (see also docs/GAMES.md):

- **Motion vectors** are prev→curr, in pixels, at render resolution; the
  demo/synthetic paths write them through a dedicated kernel (`odl_motion`)
  while engines bind their own velocity buffer.
- **Depth** is linear 0..1 at render resolution.
- **Jitter** must be baked into the render by the engine (the SDK's
  `jitterX/jitterY` fields exist for engines that report rather than apply).
- **Camera cuts** set `reset` (the SDK: `resetHistory`) to avoid ghosting.
- The scaler's feedback means output history lives inside the scaler; the
  C SDK additionally keeps a host-visible copy for CPU fallback backends.

## Where the neural graph fits

MetalFX owns scaling and temporal accumulation. The neural rendering pass —
the part ported from OpenDLSS-NR — composes around it:

1. the 16-lane preprocess runs at the render/valid resolution,
2. the transformer graph computes the per-pixel RGB residual + blend logit,
3. the head composite applies the residual and closes the temporal loop,
4. MetalFX scales the result.

This split matches the reference project's own division of labor: the network
re-renders the frame; the upscaler changes its size. On Windows/Linux the
same split runs over DirectML/Vulkan with our reprojection scaler substituting
for the temporal scaler.

## Device capability reporting

`opendlss-nr info` prints, per backend, the MetalFX flags:

```
metal    device: Apple M2 Pro   MetalFX[s:1 t:1] neural:1 fp16:1 e4m3:1
```

`s:` = spatial scaler available, `t:` = temporal scaler available (Apple7
family and later). The Swift viewer surfaces the same state live in its
status bar.
