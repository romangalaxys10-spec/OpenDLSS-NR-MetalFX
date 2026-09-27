// opendlss-nr — cross-platform command line tool.
//
//   info                      list backends and capabilities
//   geometry W H              print the padded field + levels (parity tool)
//   image                     run the image pipeline
//   video                     run the video pipeline (Y4M in/out)
//   game                      run the real-time (game) pipeline on synthetic frames
//   demo                      self-contained demo: synthetic scene through every stage
//   bench                     measure the pipeline
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/backend.h"
#include "opendlss/pipeline.h"
#include "opendlss/version.h"
#include "opendlss/logging.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

using namespace opendlss;

namespace {

struct Args {
    std::vector<std::string> positional;
    std::map<std::string, std::string> opts;
    std::map<std::string, bool> flags;

    bool has(const std::string& k) const { return flags.count(k) || opts.count(k); }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = opts.find(k);
        return it == opts.end() ? def : it->second;
    }
    float getf(const std::string& k, float def) const {
        auto it = opts.find(k);
        return it == opts.end() ? def : std::strtof(it->second.c_str(), nullptr);
    }
    uint32_t getu(const std::string& k, uint32_t def) const {
        auto it = opts.find(k);
        return it == opts.end() ? def : uint32_t(std::strtoul(it->second.c_str(), nullptr, 10));
    }
    uint64_t getu64(const std::string& k, uint64_t def) const {
        auto it = opts.find(k);
        return it == opts.end() ? def : std::strtoull(it->second.c_str(), nullptr, 10);
    }
};

Args parse_args(int argc, char** argv, int skip) {
    Args a;
    for (int i = skip; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            std::string key = s.substr(2);
            if (key.rfind("no-", 0) == 0) { a.flags[key] = true; continue; }
            if (i + 1 < argc && argv[i + 1][0] != '-' && std::strchr(argv[i + 1], '.') == nullptr
                && !(argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')) {
                // value is a word
            }
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                a.opts[key] = argv[++i];
            } else {
                a.flags[key] = true;
            }
        } else {
            a.positional.push_back(s);
        }
    }
    return a;
}

BackendKind backend_from_string(const std::string& s) {
    if (s == "cpu") return BackendKind::Cpu;
    if (s == "metal") return BackendKind::Metal;
    if (s == "vulkan") return BackendKind::Vulkan;
    if (s == "d3d12") return BackendKind::D3D12;
    return BackendKind::Cpu;
}

PipelineOptions opts_from_args(const Args& a) {
    PipelineOptions o;
    o.denoiseStrength = a.getf("denoise", 0.6f);
    o.sharpen = a.getf("sharpen", 0.25f);
    o.style = a.getf("style", 0.0f);
    o.tone = a.getf("tone", 0.5f);
    o.structure = a.getf("structure", 0.5f);
    o.skin = a.getf("skin", 0.5f);
    o.autoMask = a.getf("automask", 0.5f);
    o.temporalMaxBlend = a.getf("max-blend", 0.85f);
    o.upscaleFactor = a.getu("scale", 2);
    o.useNeural = !a.flags.count("no-neural");
    o.useMetalFx = !a.flags.count("no-metalfx");
    o.seed = a.getu64("seed", 0x5EED1234ull);
    return o;
}

int cmd_info() {
    std::printf("%s %s\n", OPENDLSS_NR_PROJECT_NAME, OPENDLSS_NR_VERSION_STRING);
    std::printf("backends compiled into this build:\n");
    for (BackendKind k : available_backends()) {
        auto b = create_backend(k);
        if (!b) continue;
        const BackendInfo& i = b->info();
        std::printf("  %-8s device: %-28s MetalFX[s:%d t:%d] neural:%d fp16:%d e4m3:%d\n",
                    i.name.c_str(), i.deviceName.c_str(), int(i.metalFxSpatial),
                    int(i.metalFxTemporal), int(i.neuralGraph), int(i.fp16Storage), int(i.e4m3Quant));
        if (!i.details.empty()) std::printf("           %s\n", i.details.c_str());
    }
    return 0;
}

int cmd_geometry(const Args& a) {
    uint32_t w = a.getu("width", 1920), h = a.getu("height", 1080);
    uint32_t levels = a.getu("levels", 6), minField = a.getu("min-field", 320);
    auto g = Geometry::fromValid(w, h, levels, minField);
    if (!g) { std::fprintf(stderr, "geometry: rejected\n"); return 1; }
    std::printf("%s\n", geometry_to_string(*g).c_str());
    return 0;
}

int cmd_image(const Args& a) {
    ImageJob job;
    job.inputPath = a.get("in");
    job.outputPath = a.get("out");
    job.modelDir = a.get("model");
    job.backend = backend_from_string(a.get("backend", "auto"));
    job.opts = opts_from_args(a);
    if (job.inputPath.empty() || job.outputPath.empty()) {
        std::fprintf(stderr, "usage: opendlss-nr image --in <png> --out <png> [--scale N] [--model dir] ...\n");
        return 2;
    }
    if (a.get("backend", "auto") == "auto") {
        auto b = create_preferred_backend();
        if (!b) return 1;
        // run through the same path by using the preferred backend directly
        job.backend = b->info().kind;
    }
    return run_image_job(job) ? 0 : 1;
}

int cmd_video(const Args& a) {
    VideoJob job;
    job.inputPath = a.get("in");
    job.outputPath = a.get("out");
    job.modelDir = a.get("model");
    job.backend = backend_from_string(a.get("backend", "cpu"));
    job.opts = opts_from_args(a);
    job.maxFrames = a.getu("frames", 0);
    if (job.inputPath.empty() || job.outputPath.empty()) {
        std::fprintf(stderr, "usage: opendlss-nr video --in <y4m> --out <y4m> [--scale N] ...\n");
        return 2;
    }
    return run_video_job(job) ? 0 : 1;
}

// Synthetic "game": a moving scene rendered at low res with exact motion vectors,
// fed through the backend's game (temporal scaler) path at 30 fps equivalent.
int cmd_game(const Args& a) {
    auto backend = create_backend(backend_from_string(a.get("backend", "cpu")));
    if (!backend) return 1;
    if (!a.get("model").empty()) backend->loadModel(a.get("model"));

    const uint32_t RW = a.getu("render-width", 320), RH = a.getu("render-height", 180);
    const uint32_t OW = RW * a.getu("scale", 2), OH = RH * a.getu("scale", 2);
    const uint32_t frames = a.getu("frames", 30);
    const float speed = a.getf("speed", 2.5f);

    backend->beginGame(RW, RH, OW, OH);
    double totalMs = 0;
    for (uint32_t f = 0; f < frames; ++f) {
        // synthetic renderer: scrolling ridges + a bouncing light
        Image color(RW, RH);
        MotionField mv;
        mv.width = RW; mv.height = RH;
        mv.xy.assign(size_t(RW) * RH * 2, 0.f);
        const float t = float(f) / 30.0f;
        for (uint32_t y = 0; y < RH; ++y)
            for (uint32_t x = 0; x < RW; ++x) {
                float fx = float(x), fy = float(y);
                float v = 0.5f + 0.4f * std::sin((fx + t * 30.0f * speed) * 0.05f)
                                * std::sin(fy * 0.07f + 1.3f);
                float lx = RW * (0.5f + 0.3f * std::sin(t * 3.1f));
                float ly = RH * (0.5f + 0.3f * std::cos(t * 2.3f));
                float dl = std::sqrt((fx - lx) * (fx - lx) + (fy - ly) * (fy - ly));
                float light = std::exp(-dl / (RW * 0.15f));
                float* px = color.row(y) + x * 4;
                px[0] = std::clamp(v * 0.8f + light, 0.f, 1.f);
                px[1] = std::clamp(v * 0.9f + light * 0.8f, 0.f, 1.f);
                px[2] = std::clamp(v + light * 0.5f, 0.f, 1.f);
                px[3] = 1.f;
                mv.xy[(size_t(y) * RW + x) * 2 + 0] = -speed;      // exact camera motion
                mv.xy[(size_t(y) * RW + x) * 2 + 1] = 0.f;
            }
        GameFrameInput in;
        in.color = &color;
        in.motion = &mv;
        in.resetHistory = (f == 0);
        GameFrameOutput out;
        if (!backend->submitGameFrame(in, out)) { log_error("game: frame %u failed", f); return 1; }
        totalMs += out.gpuMs;
        if (f == frames - 1) {
            std::string outPath = a.get("out", "game_frame.png");
            image_save_png(outPath, out.upscaled);
            log_info("game: wrote last frame to %s", outPath.c_str());
        }
    }
    backend->endGame();
    log_info("game: %u frames, %.2f ms/frame (%s)", frames, totalMs / frames,
             backend->info().name.c_str());
    return 0;
}

int cmd_demo(const Args& a) {
    // 1. synthetic clean scene, downscale, add noise -> "engine proxy"
    const uint32_t W = a.getu("width", 320), H = a.getu("height", 180);
    auto clean = image_synthetic_scene(W, H);
    auto low = image_downscale_box(*clean, a.getu("downscale", 2));
    auto noisy = image_add_gaussian_noise(*low, 0.08f, a.getu64("seed", 1));

    ImageJob job;
    job.inputPath = "";      // in-memory path not used; save first
    job.opts = opts_from_args(a);
    job.modelDir = a.get("model");
    auto backend = create_preferred_backend();
    if (!backend) return 1;
    job.backend = backend->info().kind;

    image_save_png("demo_input.png", *noisy);
    job.inputPath = "demo_input.png";
    job.outputPath = a.get("out", "demo_output.png");
    return run_image_job(job) ? 0 : 1;
}

int cmd_bench(const Args& a) {
    auto backend = create_backend(backend_from_string(a.get("backend", "cpu")));
    if (!backend) return 1;
    if (!a.get("model").empty()) backend->loadModel(a.get("model"));
    const uint32_t W = a.getu("width", 640), H = a.getu("height", 360);
    const uint32_t frames = a.getu("frames", 10);
    auto img = image_add_gaussian_noise(*image_synthetic_scene(W, H), 0.06f, 7);
    Image out;
    double total = 0;
    for (uint32_t f = 0; f < frames; ++f) {
        auto t0 = std::chrono::steady_clock::now();
        Image dn;
        backend->denoiseSpatial(*img, dn, 0.6f);
        backend->upscaleSpatial(dn, out, a.getu("scale", 2), 0.25f);
        auto t1 = std::chrono::steady_clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::printf("bench: %ux%u -> %ux%u, %.2f ms/frame avg over %u frames (%s)\n",
                W, H, out.width, out.height, total / frames, frames, backend->info().name.c_str());
    return 0;
}

void usage() {
    std::printf(
        "%s %s — DLSS-style neural rendering + MetalFX-accelerated scaling\n\n"
        "usage: opendlss-nr <command> [options]\n\n"
        "commands:\n"
        "  info                              list backends and capabilities\n"
        "  geometry --width W --height H     print padded field and level sizes\n"
        "  image    --in a.png --out b.png   image pipeline  [--scale 2] [--model dir]\n"
        "  video    --in a.y4m --out b.y4m   video pipeline  [--scale 2] [--model dir]\n"
        "  game     [--out f.png]            real-time path on synthetic frames\n"
        "  demo     [--out b.png]            synthetic scene through every stage\n"
        "  bench    [--width W --height H]   measure ms/frame\n\n"
        "common options:\n"
        "  --backend cpu|metal|vulkan|d3d12  execution backend (default: preferred)\n"
        "  --model <dir>                     model directory with manifest.json\n"
        "  --scale N                         output scale factor (1,2,3,4)\n"
        "  --denoise F                       analytical denoise strength 0..1\n"
        "  --sharpen F                       adaptive sharpen 0..1\n"
        "  --no-neural                       skip the neural graph\n"
        "  --style F --tone F --structure F --skin F --automask F\n"
        "  --seed N                          noise seed base\n",
        OPENDLSS_NR_PROJECT_NAME, OPENDLSS_NR_VERSION_STRING);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 0; }
    std::string cmd = argv[1];
    Args a = parse_args(argc, argv, 2);
    if (a.has("verbose")) log_set_level(LogLevel::Debug);

    if (cmd == "info")     return cmd_info();
    if (cmd == "geometry") return cmd_geometry(a);
    if (cmd == "image")    return cmd_image(a);
    if (cmd == "video")    return cmd_video(a);
    if (cmd == "game")     return cmd_game(a);
    if (cmd == "demo")     return cmd_demo(a);
    if (cmd == "bench")    return cmd_bench(a);
    if (cmd == "help" || cmd == "--help") { usage(); return 0; }
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
    usage();
    return 2;
}
