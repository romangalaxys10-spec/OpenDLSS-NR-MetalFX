# Architecture

One repository, four backends, one numerics contract.

```
opendlss-nr-metalfx/
├── core/                       platform-neutral C++20
│   ├── include/opendlss/       the public API surface
│   │   ├── fp16.h              software half + bit-trick exponentials
│   │   ├── e4m3.h              E4M3 (OCP FP8) codec
│   │   ├── geometry.h/.h       padded-field math (reference port)
│   │   ├── image.h             RGBA f32 buffers, stb PNG I/O, generators
│   │   ├── model.h             manifest + E4M3 stage loading
│   │   ├── backend.h           IBackend contract + backend registry
│   │   └── pipeline.h/.h       16-lane preprocess, head, Y4M, mode drivers
│   ├── src/                    core implementations
│   └── backends/cpu/           the CPU reference (parity golden)
├── platforms/
│   ├── macos/                  Metal + MetalFX backend (Objective-C++),
│   │   ├── shaders/*.metal     kernel set (port of the CPU semantics)
│   │   ├── src/metal_backend.mm
│   │   └── viewer/OpenDLSSViewer/   SwiftUI + MetalKit demo viewer
│   ├── windows/                D3D12 + DirectML backend (C++20/HLSL)
│   └── linux/                  Vulkan compute backend (C++20/GLSL)
├── apps/cli/                   the opendlss-nr command line tool
├── sdk/                        stable C ABI for engines
├── demos/                      image / video / game demos + runner
├── tools/                      model generation + training export
├── tests/                      unit + parity tests (CTest)
├── scripts/                    install/build/demo scripts (3 platforms)
└── .github/workflows/          per-OS CI
```

## The backend contract

Every GPU backend implements `core/include/opendlss/backend.h`:

```cpp
class IBackend {
    // image mode
    virtual bool denoiseSpatial(const Image& in, Image& out, float strength);
    virtual bool upscaleSpatial(const Image& in, Image& out, uint32_t factor, float sharpen);
    // neural graph
    virtual bool loadModel(const std::string& dir);
    virtual bool runNeuralGraph(const Geometry&, const uint16_t* features /*field*16 half*/,
                                float* head /*field*4 f32*/, uint64_t seed);
    // video/game temporal ops
    virtual bool estimateMotion(const Image& prevLuma, const Image& currLuma, MotionField&);
    virtual bool reprojectHistory(const Image& history, const MotionField&,
                                  Image& reproj, Image& confidence);
    virtual bool temporalBlend(const Image& current, const Image& reproj,
                               const Image& confidence, float maxBlend, Image& out);
    // real-time game path (MetalFX temporal scaler on Apple Silicon)
    virtual bool beginGame(renderW, renderH, outputW, outputH);
    virtual bool submitGameFrame(const GameFrameInput&, GameFrameOutput&);
    virtual void endGame();
};
```

The mode drivers in `core/src/pipeline_jobs.cpp` compose these ops into the
image and video pipelines — identical orchestration on every platform; only
the op execution differs. The game path is engine-driven through the C SDK.

## The numerics contract

All four backends share one set of bit-exact primitives:

- `f32 ↔ f16` conversion with round-to-nearest-even (`fp16.h`),
- the attention exponentials — literally bit operations on the half pattern:
  window: `(bits(clamp(half(0.044921875 s + 1.30078125), 1.03125, 1.5693359375)) << 5) ^ 0x8000`,
  ViT: `(bits(clamp(half(0.08953947 s + 1.70936143), 1.439453125, 1.9775390625)) << 4) + 0x4000`,
- the E4M3 quantization grid (RTNE, saturate ±448, NaN 0x7F, subnormals m·2⁻⁹),
- the padded-field geometry (see `geometry.h` — ported rule for rule, including
  the reference's final `fieldW += alignW` quirk),
- the pixel hash seeding the noise lanes (`preprocess_pixel_hash`).

The CPU backend executes the whole graph with these primitives; the GPU
kernels are ports of the same math (Common.metal / Common.glsl /
Common.hlsli carry identical helpers). This makes CPU-vs-GPU parity testing
meaningful and keeps every platform on one specification.

## Data flow, image mode

```
PNG → RGBA f32 → [bilateral denoise] → preprocess(16 lanes, field geometry)
    → neural graph (blocks per manifest schedule) → head 4ch
    → head_composite (residual + history blend) → upscale (MetalFX | Lanczos)
    → adaptive sharpen → PNG
```

## Data flow, video mode

```
Y4M frame n:
  motion(prevLow, curLow) → mv
  reproject(prevDisplay↓scale, mv) → reproj, confidence
  preprocess(curLow, reproj lanes 7-9, seed+n)
  neural graph → head
  head_composite(curLow, head, reproj)  ← the blend logit closes the loop
  upscale → sharpen → Y4M write; prevDisplay = output; prevLow = composited
```

## Data flow, game mode

```
engine frame: color(RGBA16F) + depth(R32F) + motion(RG16F) + exposure + jitter
  macOS: MTLFXTemporalScaler(color, depth, motion, exposure, reset) → output
  other: Lanczos upscale → reproject → confidence-guided temporal blend
optional neural pass composes at render scale (model-dependent)
```

## The manifest pipeline

`tools/make_demo_weights.py` (or any trainer) writes a model directory:
`manifest.json` + E4M3-packed stage files. `Model::load` verifies structure,
decodes E4M3 → f16 once, and exposes tensors by name. The block schedule is
data — the executor walks whatever ladder the manifest declares (field blocks,
pool/upsample transitions, ViT channel doublings), which is why the same
binary runs the 7-block demo and a full 71-block network when weights exist.
