// nr_video.h — FFmpeg-backed video transcoding through the NR pipeline
// (decode -> RGBA f32 frames -> NR session -> encode). Compiled when FFmpeg
// is found (CMake option OPENDDLSS_WITH_VIDEO).
#pragma once

#include "nr_frame.h"

#include <functional>
#include <string>

namespace nr {
namespace video {

struct Progress {
  uint32_t frame = 0, total = 0;
};

struct TranscodeOptions {
  std::string modelDirectory;
  std::string input, output;
  nr::frame::Params params;    // validWidth/Height default to the input size
  uint32_t startFrame = 0;     // frames to skip before processing
  uint32_t maxFrames = 0;      // 0 = to the end
  uint32_t crf = 16;           // encode quality (libx264/libx265 CRF)
  bool verbose = false;
};

// Transcodes a video through the NR network, frame by frame, preserving the
// container frame rate. Uses libx264 (or libx265 by extension) with yuv420p.
void transcode(const TranscodeOptions& options, const std::function<void(const Progress&)>& progress = {});

}  // namespace video
}  // namespace nr
