// nr_frame.cpp — shared compose/pack helpers + backend-agnostic session factory.
#include "nr_frame.h"

#include "nr_numeric.h"

namespace nr {
namespace frame {

// neural = clamp((head / 32 + centred) * 8 + 0.5, 0, 1); `inner * 8` is exact,
// matching the parity tool's composition (src/vulkan/main.cpp).
float composedValue(float head, float centred) {
  const float inner = std::fmaf(head, 0.03125f, centred);
  return std::min(std::max(std::fmaf(inner, 8.0f, 0.5f), 0.0f), 1.0f);
}

void packRgba8(const float* composed, size_t pixels, std::vector<uint8_t>& out) {
  out.resize(pixels * 4);
  for (size_t i = 0; i < pixels * 4; ++i) {
    float value = composed[i];
    value = std::min(std::max(value, 0.0f), 1.0f);
    out[i] = (uint8_t)std::lrintf(value * 255.0f);
  }
}

}  // namespace frame
}  // namespace nr

// ---------------------------------------------------------------------------
// Backend instantiations. The Metal session is compiled into the macOS target
// (nr_frame_metal.mm); the Vulkan session into the Windows/Linux target
// (nr_frame_vulkan.cpp). This translation unit carries the factory.
// ---------------------------------------------------------------------------
#include <stdexcept>

namespace nr {
namespace frame {

#if defined(NR_FRAME_BACKEND_METAL)
std::unique_ptr<Session> createSessionMetal(const SessionOptions& options);
#elif defined(NR_FRAME_BACKEND_VULKAN)
std::unique_ptr<Session> createSessionVulkan(const SessionOptions& options);
#endif

std::unique_ptr<Session> createSession(const SessionOptions& options) {
#if defined(NR_FRAME_BACKEND_METAL)
  return createSessionMetal(options);
#elif defined(NR_FRAME_BACKEND_VULKAN)
  return createSessionVulkan(options);
#else
  (void)options;
  throw std::runtime_error("no NR backend compiled into this binary");
#endif
}

}  // namespace frame
}  // namespace nr
