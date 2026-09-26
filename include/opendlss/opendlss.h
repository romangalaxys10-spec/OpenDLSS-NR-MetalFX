/* opendlss.h — the C integration API of OpenDLSS-NR-MetalFX.
 *
 * The stable surface for game engines, video tools and imaging apps:
 *   - one session = one model + one field size (thread-safe per session;
 *     create one per rendering thread)
 *   - processRgba runs the 71-block NR network on one frame
 *   - on macOS (Apple silicon) the session can be backed by Metal + MetalFX;
 *     game engines driving their own Metal renderer should prefer the
 *     MetalFX temporal path shown in apps/demo-game-macos and docs/GAMES.md
 *
 * Build: link libopendlss (CMake target `opendlss`); headers: this file.
 */
#pragma once

#include <stdint.h>

#if defined(_WIN32)
  #define OPENDLSS_EXPORT __declspec(dllexport)
#else
  #define OPENDLSS_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpendlssSession OpendlssSession;

typedef struct OpendlssParams {
  uint32_t validWidth, validHeight;    /* network field (defaults: source size) */
  uint32_t sourceWidth, sourceHeight;  /* proxy resolution (defaults: source size) */
  uint32_t seed;                       /* injected-noise seed */
  float style;                         /* 0..128 */
  float localTone;
  float localStructure;
  float skinStructure;                 /* < 0 = auto */
  int32_t autoMask;
} OpendlssParams;

typedef struct OpendlssStats {
  uint32_t dispatches;
  double cpuMilliseconds;
  const char* device;
} OpendlssStats;

OPENDLSS_EXPORT const char* opendlssVersion(void);
/* "metal+metalfx" on macOS, "vulkan" on Windows/Linux; NULL + reason on error. */
OPENDLSS_EXPORT const char* opendlssBackend(const char** reason);

/* Creates a session. modelDir holds manifest.json + model/*. Returns NULL on
 * failure (model missing, unsupported size, no GPU); *reasonOut gets a message. */
OPENDLSS_EXPORT OpendlssSession* opendlssSessionCreate(const char* modelDir, const OpendlssParams* params,
                                                       const char** reasonOut);
OPENDLSS_EXPORT void opendlssSessionDestroy(OpendlssSession* session);

/* One frame: rgbaIn = 8-bit RGBA source (sourceWidth x sourceHeight, row-major);
 * rgbaOut receives validWidth x validHeight composed RGBA8 (caller allocates:
 * validWidth*validHeight*4). Returns 0 on success, -1 on failure. */
OPENDLSS_EXPORT int opendlssProcessRgba(OpendlssSession* session, const uint8_t* rgbaIn, uint32_t inStrideBytes,
                                        uint8_t* rgbaOut, uint32_t outStrideBytes, OpendlssStats* statsOut);

/* Sets the noise seed (per-frame variation for video/game frames). */
OPENDLSS_EXPORT int opendlssSetSeed(OpendlssSession* session, uint32_t seed);

#ifdef __cplusplus
}  // extern "C"
#endif
