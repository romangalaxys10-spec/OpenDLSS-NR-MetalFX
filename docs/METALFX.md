# MetalFX

MetalFX is Apple's upscaler family (macOS 13+ spatial, macOS 14+ temporal) and the "FX" in this
repository's name. The two compose with the NR network like this:

| path | scaler | why |
| --- | --- | --- |
| images / video | `MTLFXSpatialScaler` (Adaptive) | detail-preserving spatial upscale before/after the NR pass; deterministic, no history |
| games | `MTLFXTemporalScaler` (preferred) or Spatial | the engine renders low-res with motion vectors + depth; MetalFX temporally accumulates; NR runs on the upscaled frame |

`src/metal/nr_metalfx.mm` wraps both descriptors verbatim (formats, content dimensions, exposure),
and `apps/demo-game-macos` shows the full loop. Engines that already run their own temporal
anti-aliasing can keep it and use NR alone; engines targeting maximum frame rate render at ~1/4
pixel count, let MetalFX temporal reconstruct, and pay the NR pass on the upscaled frame.

Honest notes: the temporal scaler requires *real* motion vectors (the procedural demo scene writes
zeros and therefore takes the spatial route in its default configuration — flipping it to
`MTLFXTemporalScaler` is a two-line change shown in docs/GAMES.md). MetalFX scaler availability is
checked at runtime (`supportsDevice:`); the CLI reports it under `opendlss devices`.
