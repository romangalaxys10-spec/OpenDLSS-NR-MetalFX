// nr_frame.h — shared frame pipeline of the product surface (CLI, SDK, demos).
//
// One frame of the NR network: an RGBA f32 proxy (display code values) at the
// source resolution feeds the preprocess kernel, which builds the 16-lane f32
// input features of the padded field; the 71-block graph produces the f32
// RGBA head; the output is composed with the centred proxy residual:
//   out = clamp((head/32 + centred) * 8 + 0.5, 0, 1)
// identical to the parity tool's composition (src/vulkan/main.cpp).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nr {
namespace frame {

struct Params {
  uint32_t validWidth = 0, validHeight = 0;   // the field the network runs on (<= 8192; see Geometry)
  uint32_t sourceWidth = 0, sourceHeight = 0; // proxy resolution (<= valid; 1:1 when equal)
  uint32_t seed = 0;                          // injected-noise seed
  float style = 0.0f;                         // 0..128 style conditioning
  float localTone = 0.0f, localStructure = 0.0f, skinStructure = 0.0f;
  bool autoMask = false;
  bool captureBoundaries = false;             // parity instrumentation
};

struct SessionStats {
  uint32_t dispatches = 0;
  double gpuMilliseconds = 0.0;
  std::string device;
};

class Session {
 public:
  virtual ~Session() = default;
  // Runs one frame: `proxy` = RGBA f32 (sourceWidth*sourceHeight*4 display code values);
  // `out` receives composed RGBA8 (outWidth*outHeight*4) where out = valid (1:1) or
  // the MetalFX spatial target when a scale factor was configured.
  virtual void processFrame(const float* proxy, uint32_t proxyStridePixels, std::vector<uint8_t>& out,
                            SessionStats* stats = nullptr) = 0;
  virtual const Params& params() const = 0;
};

struct SessionOptions {
  std::string modelDirectory;
  Params params;
  // Output composition: scale the valid field up to `outputWidth/Height` with the
  // backend's spatial scaler (MetalFX Adaptive on macOS; nearest on other backends
  // is refused — pass 1:1 there).
  bool upscale = false;
  // Style/toning presets matching the demo scenes (see docs/IMAGES.md).
  bool centredFromProxy = true;
};

std::unique_ptr<Session> createSession(const SessionOptions& options);

// Compose + pack helpers shared by every consumer (CLI, SDK, demos).
// centred = roundF16(roundF16(roundF16(proxy)-0.5)*0.125) per channel.
float composedValue(float head, float centred);
void packRgba8(const float* composed, size_t pixels, std::vector<uint8_t>& out);

}  // namespace frame
}  // namespace nr
