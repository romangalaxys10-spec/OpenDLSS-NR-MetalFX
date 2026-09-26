// opendlss — the unified cross-platform CLI of OpenDLSS-NR-MetalFX.
//
//   opendlss devices                       show the detected backend + GPU
//   opendlss image  <input> <output>       NR-process one image (PNG/JPEG)
//   opendlss video  <input> <output>       NR-process a video (FFmpeg)
//   opendlss bench  --model <dir> ...      frames/second of the full network
//   opendlss profile --model <dir> ...     dispatch counts per stage
//   opendlss selftest                      CPU numeric/reference tests
//
// Backend selection: Metal+MetalFX on macOS (Apple silicon), the bit-exact
// Vulkan route on Windows/Linux/NVIDIA. `--backend vulkan|metal` overrides.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nr_frame.h"
#include "nr_geometry.h"
#include "nr_image.h"
#include "nr_numeric.h"
#include "nr_cpu_reference.h"
#if defined(NR_WITH_VIDEO)
#include "nr_video.h"
#endif

namespace {

void usage() {
  printf(R"(OpenDLSS-NR-MetalFX - neural rendering (DLSS 5 NR architecture) for macOS/Windows/Linux

usage:
  opendlss devices
  opendlss image  <input.(png|jpg)> <output.(png|jpg)> [options]
  opendlss video  <input.(mp4|mkv|mov)> <output.mp4> [options]     (FFmpeg build)
  opendlss bench  --model <dir> --width W --height H [--frames N]
  opendlss profile --model <dir> --width W --height H
  opendlss selftest

common options:
  --model <dir>            model directory (manifest.json + model/*.stages; see docs)
  --style <0..128>         style conditioning (default 0)
  --grain <seed>           injected-noise seed (default 0; per-frame offset for video)
  --tonemap <f>            local tone conditioning (default 0)
  --structure <f>          local structure conditioning (default 0)
  --skin <f>               skin structure conditioning (default -1)
  --automask               enable the auto skin-mask lane pair
  --backend <auto|metal|vulkan>  backend override (default auto)
)");
}

[[noreturn]] void fail(const std::string& message) {
  fprintf(stderr, "opendlss: %s\n", message.c_str());
  exit(1);
}

struct CommonArgs {
  std::string model = "models/nr";
  float style = 0.0f, localTone = 0.0f, localStructure = 0.0f, skinStructure = -1.0f;
  bool autoMask = false;
  uint32_t seed = 0;
};

CommonArgs parseCommon(int& i, char** argv, int argc) {
  CommonArgs args;
  for (; i < argc; ++i) {
    std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fail("missing value for " + key);
      return argv[++i];
    };
    if (key == "--model") args.model = next();
    else if (key == "--style") args.style = std::stof(next());
    else if (key == "--tonemap") args.localTone = std::stof(next());
    else if (key == "--structure") args.localStructure = std::stof(next());
    else if (key == "--skin") args.skinStructure = std::stof(next());
    else if (key == "--automask") args.autoMask = true;
    else if (key == "--grain") args.seed = (uint32_t)std::stoul(next());
    else break;
  }
  return args;
}

nr::frame::Params makeParams(const CommonArgs& common, uint32_t validW, uint32_t validH, uint32_t srcW,
                             uint32_t srcH, uint32_t seed) {
  nr::frame::Params params;
  params.validWidth = validW;
  params.validHeight = validH;
  params.sourceWidth = srcW;
  params.sourceHeight = srcH;
  params.seed = seed;
  params.style = common.style;
  params.localTone = common.localTone;
  params.localStructure = common.localStructure;
  params.skinStructure = common.skinStructure;
  params.autoMask = common.autoMask;
  return params;
}

int cmdDevices() {
#if defined(__APPLE__)
  printf("backend: metal+metalfx (macOS)\n");
#else
  printf("backend: vulkan\n");
#endif
  try {
    nr::frame::SessionOptions options;
    // Device enumeration without a model: a tiny probe session is overkill for
    // v1; report the backend family and availability instead.
    printf("model required for a full device report (run bench with --model)\n");
  } catch (const std::exception& e) {
    printf("probe: %s\n", e.what());
  }
  return 0;
}

int cmdImage(int argc, char** argv) {
  if (argc < 2) fail("image needs <input> <output>");
  std::string input = argv[0], output = argv[1];
  int i = 2;
  CommonArgs common = parseCommon(i, argv, argc);
  nr::image::Image image = nr::image::load(input);
  CommonArgs c = common;
  (void)c;
  nr::frame::SessionOptions options;
  options.modelDirectory = common.model;
  options.params = makeParams(common, image.width, image.height, image.width, image.height, common.seed);
  std::unique_ptr<nr::frame::Session> session = nr::frame::createSession(options);
  std::vector<uint8_t> outRgba;
  auto started = std::chrono::steady_clock::now();
  session->processFrame(image.rgba.data(), image.width, outRgba);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  nr::image::save(output, outRgba.data(), image.width, image.height);
  printf("done: %ux%u -> %s (%.1f ms, %s)\n", image.width, image.height, output.c_str(), ms,
         session->params().seed == common.seed ? "1 frame" : "1 frame");
  return 0;
}

int cmdVideo(int argc, char** argv) {
#if !defined(NR_WITH_VIDEO)
  (void)argc; (void)argv;
  fail("this build has no FFmpeg (video) support; rebuild with OPENDDLSS_WITH_VIDEO=ON");
#else
  if (argc < 2) fail("video needs <input> <output>");
  std::string input = argv[0], output = argv[1];
  int i = 2;
  CommonArgs common = parseCommon(i, argv, argc);
  uint32_t start = 0, frames = 0, crf = 16;
  for (; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--start" && i + 1 < argc) start = (uint32_t)std::stoul(argv[++i]);
    else if (key == "--frames" && i + 1 < argc) frames = (uint32_t)std::stoul(argv[++i]);
    else if (key == "--crf" && i + 1 < argc) crf = (uint32_t)std::stoul(argv[++i]);
    else fail("unknown video option " + key);
  }
  nr::video::TranscodeOptions options;
  options.modelDirectory = common.model;
  options.input = input;
  options.output = output;
  options.startFrame = start;
  options.maxFrames = frames;
  options.crf = crf;
  options.params.style = common.style;
  options.params.localTone = common.localTone;
  options.params.localStructure = common.localStructure;
  options.params.skinStructure = common.skinStructure;
  options.params.autoMask = common.autoMask;
  nr::video::transcode(options, [](const nr::video::Progress& p) {
    if (p.frame % 10 == 0) printf("\rframe %u      ", p.frame);
    fflush(stdout);
  });
  printf("\ndone: %s\n", output.c_str());
  return 0;
#endif
}

int cmdBench(int argc, char** argv) {
  std::string model;
  uint32_t width = 768, height = 768, frames = 40, seed = 0;
  for (int i = 0; i < argc; ++i) {
    std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fail("missing value for " + key);
      return argv[++i];
    };
    if (key == "--model") model = next();
    else if (key == "--width") width = (uint32_t)std::stoul(next());
    else if (key == "--height") height = (uint32_t)std::stoul(next());
    else if (key == "--frames") frames = (uint32_t)std::stoul(next());
    else if (key == "--seed") seed = (uint32_t)std::stoul(next());
    else fail("unknown bench option " + key);
  }
  if (model.empty()) fail("bench needs --model");
  nr::frame::SessionOptions options;
  options.modelDirectory = model;
  options.params = makeParams(CommonArgs{}, width, height, width, height, seed);
  std::unique_ptr<nr::frame::Session> session = nr::frame::createSession(options);
  std::vector<float> proxy((size_t)width * height * 4, 0.5f);
  std::vector<uint8_t> out;
  double best = 1e9, total = 0.0;
  for (uint32_t f = 0; f < frames; ++f) {
    auto started = std::chrono::steady_clock::now();
    session->processFrame(proxy.data(), width, out);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    best = std::min(best, ms);
    total += ms;
  }
  printf("%ux%u over %u frames: best %.2f ms, mean %.2f ms (%.1f fps best)\n", width, height, frames, best,
         total / frames, 1000.0 / best);
  return 0;
}

int cmdProfile(int argc, char** argv) {
  std::string model;
  uint32_t width = 768, height = 768;
  for (int i = 0; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--model" && i + 1 < argc) model = argv[++i];
    else if (key == "--width" && i + 1 < argc) width = (uint32_t)std::stoul(argv[++i]);
    else if (key == "--height" && i + 1 < argc) height = (uint32_t)std::stoul(argv[++i]);
    else fail("unknown profile option " + key);
  }
  if (model.empty()) fail("profile needs --model");
  nr::frame::SessionOptions options;
  options.modelDirectory = model;
  options.params = makeParams(CommonArgs{}, width, height, width, height, 0);
  std::unique_ptr<nr::frame::Session> session = nr::frame::createSession(options);
  std::vector<float> proxy((size_t)width * height * 4, 0.5f);
  std::vector<uint8_t> out;
  nr::frame::SessionStats stats;
  session->processFrame(proxy.data(), width, out, &stats);
  printf("device: %s\ndispatches: %u\n", stats.device.c_str(), stats.dispatches);
  return 0;
}

int cmdSelftest() {
  // The exact conversion rules every backend publishes through.
  using namespace nr;
  int failures = 0;
  auto check = [&](bool ok, const char* what) {
    if (!ok) {
      fprintf(stderr, "FAIL %s\n", what);
      ++failures;
    }
  };
  check(num::f16Bits(0.0f) == 0x0000, "f16 +0");
  check(num::f16Bits(1.0f) == 0x3c00, "f16 1.0");
  check(num::f16Bits(65504.0f) == 0x7bff, "f16 max");
  check(num::f16Bits(65520.0f) == 0x7c00, "f16 overflow -> inf");
  check(num::f16Bits(-2.5f) == 0xc100, "f16 -2.5");
  check(num::f16ToF32(0x3c00) == 1.0f, "f16 -> f32 1.0");
  check(num::e4m3FromF32(0.0f) == 0x00, "e4m3 zero");
  check(num::e4m3FromF32(1.0f) == 0x38, "e4m3 1.0");
  check(num::e4m3FromF32(448.0f) == 0x7e, "e4m3 max");
  check(num::e4m3FromF32(460.0f) == 0x7e, "e4m3 saturate");
  check(num::e4m3FromF32(0.001953125f) == 0x01, "e4m3 min subnormal");
  check(num::e4m3FromF32(-3.5f) == 0xc6, "e4m3 -3.5");
  check(num::e4m3ToF32(0x38) == 1.0f, "e4m3 decode 1.0");
  // SiLU table spot checks (the reference defines every one of the 65536 codes).
  float value = cpuref::mpCubicSilu(1.0f);
  check(std::fabs(value - 1.2861328125f) < 0.001f, "silu(1)");
  check(cpuref::mpCubicSilu(0.0f) == 0.0f, "silu(0)");
  // Geometry: the field alignment rules.
  nr::Geometry geometry = nr::Geometry::fromValid(1920, 1080);
  check(geometry.fullWidth >= 1920 && geometry.levels[0].width % 8 == 0, "geometry 1080p");
  nr::Geometry small = nr::Geometry::fromValid(256, 256);
  check(small.fullWidth == small.fullHeight, "geometry square");
  if (failures == 0) printf("selftest: all checks passed\n");
  else printf("selftest: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 0;
  }
  std::string command = argv[1];
  try {
    if (command == "devices") return cmdDevices();
    if (command == "image") return cmdImage(argc - 2, argv + 2);
    if (command == "video") return cmdVideo(argc - 2, argv + 2);
    if (command == "bench") return cmdBench(argc - 2, argv + 2);
    if (command == "profile") return cmdProfile(argc - 2, argv + 2);
    if (command == "selftest") return cmdSelftest();
    if (command == "--help" || command == "-h" || command == "help") {
      usage();
      return 0;
    }
    fail("unknown command " + command + " (try --help)");
  } catch (const std::exception& e) {
    fprintf(stderr, "opendlss: %s\n", e.what());
    return 1;
  }
}
