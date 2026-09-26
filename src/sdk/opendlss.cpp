// opendlss.cpp — C API implementation over the shared frame session.
#include "opendlss/opendlss.h"

#include "nr_frame.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

using namespace nr;

namespace {
thread_local std::string g_lastError;

std::unique_ptr<frame::Session>& storage(OpendlssSession* session) {
  return *reinterpret_cast<std::unique_ptr<frame::Session>*>(session);
}
}  // namespace

extern "C" {

const char* opendlssVersion(void) { return "1.0.0"; }

const char* opendlssBackend(const char** reason) {
#if defined(__APPLE__)
  static const char* backend = "metal+metalfx";
#else
  static const char* backend = "vulkan";
#endif
  if (reason) *reason = nullptr;
  return backend;
}

OpendlssSession* opendlssSessionCreate(const char* modelDir, const OpendlssParams* params,
                                       const char** reasonOut) {
  if (reasonOut) *reasonOut = nullptr;
  try {
    frame::SessionOptions options;
    options.modelDirectory = modelDir ? modelDir : "models/nr";
    frame::Params& p = options.params;
    p.validWidth = params->validWidth ? params->validWidth : params->sourceWidth;
    p.validHeight = params->validHeight ? params->validHeight : params->sourceHeight;
    p.sourceWidth = params->sourceWidth;
    p.sourceHeight = params->sourceHeight;
    p.seed = params->seed;
    p.style = params->style;
    p.localTone = params->localTone;
    p.localStructure = params->localStructure;
    p.skinStructure = params->skinStructure;
    p.autoMask = params->autoMask != 0;
    if (!p.validWidth || !p.validHeight || !p.sourceWidth || !p.sourceHeight) {
      if (reasonOut) *reasonOut = "all dimensions must be non-zero";
      return nullptr;
    }
    auto session = new std::unique_ptr<frame::Session>(frame::createSession(options));
    return reinterpret_cast<OpendlssSession*>(session);
  } catch (const std::exception& e) {
    g_lastError = e.what();
    if (reasonOut) *reasonOut = g_lastError.c_str();
    return nullptr;
  }
}

void opendlssSessionDestroy(OpendlssSession* session) {
  if (!session) return;
  delete &storage(session);
}

int opendlssProcessRgba(OpendlssSession* session, const uint8_t* rgbaIn, uint32_t inStrideBytes, uint8_t* rgbaOut,
                        uint32_t outStrideBytes, OpendlssStats* statsOut) {
  if (!session || !rgbaIn || !rgbaOut) return -1;
  try {
    frame::Session& sessionRef = *storage(session);
    const frame::Params& params = sessionRef.params();
    // 8-bit RGBA -> f32 proxy
    std::vector<float> proxy((size_t)params.sourceWidth * params.sourceHeight * 4);
    for (uint32_t y = 0; y < params.sourceHeight; ++y) {
      const uint8_t* row = rgbaIn + (size_t)y * inStrideBytes;
      float* out = &proxy[(size_t)y * params.sourceWidth * 4];
      for (uint32_t x = 0; x < params.sourceWidth * 4; ++x) out[x] = row[x] * (1.0f / 255.0f);
    }
    std::vector<uint8_t> packed;
    auto started = std::chrono::steady_clock::now();
    frame::SessionStats stats;
    sessionRef.processFrame(proxy.data(), params.sourceWidth, packed, &stats);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    for (uint32_t y = 0; y < params.validHeight; ++y) {
      memcpy(rgbaOut + (size_t)y * outStrideBytes, &packed[(size_t)y * params.validWidth * 4],
             params.validWidth * 4);
    }
    if (statsOut) {
      statsOut->dispatches = stats.dispatches;
      statsOut->cpuMilliseconds = ms;
      static std::string device;
      device = stats.device;
      statsOut->device = device.c_str();
    }
    return 0;
  } catch (const std::exception& e) {
    g_lastError = e.what();
    return -1;
  }
}

int opendlssSetSeed(OpendlssSession* session, uint32_t seed) {
  (void)session;
  (void)seed;
  // Seeds are per-params in v1; recreate the session for a different base seed
  // (documented). Returns 0 so callers can treat it as advisory.
  return 0;
}

}  // extern "C"
