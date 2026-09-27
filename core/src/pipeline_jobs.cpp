// Mode drivers: image job and video job. These orchestrate the backend ops
// and the network the same way on every platform; the backend supplies the
// execution (MetalFX on macOS, DirectML on Windows, Vulkan on Linux, CPU ref).
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/pipeline.h"
#include "opendlss/fp16.h"
#include "opendlss/logging.h"

#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace opendlss {

namespace {
ConditionScalars opts_to_cond(const PipelineOptions& o) {
    return ConditionScalars{o.style, o.tone, o.structure, o.skin, o.autoMask};
}
} // namespace

// ---------------------------------------------------------------------------
// image job: load -> denoise -> (neural) -> upscale -> save
// ---------------------------------------------------------------------------
bool run_image_job(const ImageJob& job) {
    auto backend = create_backend(job.backend);
    if (!backend) { log_error("image: backend unavailable"); return false; }
    if (!job.modelDir.empty() && !backend->loadModel(job.modelDir)) return false;

    auto src = image_load(job.inputPath);
    if (!src) return false;
    const auto t0 = std::chrono::steady_clock::now();
    log_info("image: %s %ux%u via %s", job.inputPath.c_str(), src->width, src->height,
             backend->info().name.c_str());

    Image working = *src;

    // 1. optional spatial denoise (analytical)
    if (job.opts.denoiseStrength > 0.0f) {
        Image dn;
        backend->denoiseSpatial(working, dn, job.opts.denoiseStrength);
        working = std::move(dn);
    }

    // 2. neural rendering pass (resolution-preserving residual + blend) when
    //    a model is loaded and enabled. The head output composes over the proxy.
    if (backend->hasNeuralGraph() && job.opts.useNeural) {
        auto geom = Geometry::fromValid(working.width, working.height,
                                        backend->model()->config().levels,
                                        backend->model()->config().minField,
                                        backend->model()->config().windowSize);
        if (!geom) { log_error("image: geometry rejected %ux%u", working.width, working.height); return false; }
        log_info("image: %s", geometry_to_string(*geom).c_str());

        std::vector<uint16_t> features;
        preprocess_pack_features(*geom, working, nullptr, opts_to_cond(job.opts),
                                 job.opts.seed, features);
        std::vector<float> head(size_t(geom->fieldWidth) * geom->fieldHeight * 4, 0.f);
        if (!backend->runNeuralGraph(*geom, features.data(), head.data(), job.opts.seed)) {
            log_error("image: neural graph failed");
            return false;
        }
        Image neural(working.width, working.height);
        head_composite(working, head.data(), nullptr, backend->model()->blendScale(), neural);
        working = std::move(neural);
    }

    // 3. spatial upscale (MetalFX spatial scaler on macOS)
    Image out;
    if (!backend->upscaleSpatial(working, out, job.opts.upscaleFactor, job.opts.sharpen)) {
        log_error("image: upscale failed");
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    log_info("image: done in %.1f ms -> %ux%u",
             std::chrono::duration<double, std::milli>(t1 - t0).count(), out.width, out.height);
    if (!image_save_png(job.outputPath, out)) return false;
    log_info("image: wrote %s", job.outputPath.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// video job: per-frame temporal pipeline
//   motion estimate -> reproject history -> preprocess(+history) -> neural
//   (or analytical blend) -> upscale -> write; feedback the display frame.
// ---------------------------------------------------------------------------
bool run_video_job(const VideoJob& job) {
    auto backend = create_backend(job.backend);
    if (!backend) { log_error("video: backend unavailable"); return false; }
    if (!job.modelDir.empty() && !backend->loadModel(job.modelDir)) return false;

    Y4mReader reader;
    if (!reader.open(job.inputPath)) return false;
    Y4mWriter writer;
    if (!writer.open(job.outputPath, reader.width() * job.opts.upscaleFactor,
                     reader.height() * job.opts.upscaleFactor, reader.fps()))
        return false;

    log_info("video: %s %ux%u @%.2f fps via %s", job.inputPath.c_str(), reader.width(),
             reader.height(), reader.fps(), backend->info().name.c_str());

    Image prevDisplay;         // last finished output-resolution frame (history source)
    Image prevLowLuma;         // previous input frame luma carrier
    uint32_t frameIndex = 0;

    Image frame;
    while (reader.next(frame)) {
        if (job.maxFrames && frameIndex >= job.maxFrames) break;
        const auto t0 = std::chrono::steady_clock::now();

        Image working = frame;

        // 1. motion vs previous input frame (input resolution)
        Image reprojLow;       // history reprojected to input resolution
        Image confidenceLow;
        bool haveHistory = false;
        if (!prevDisplay.empty() && !prevLowLuma.empty()) {
            MotionField mv;
            backend->estimateMotion(prevLowLuma, working, mv);
            // history is output-res; downsample to input res for reprojection
            Image historyLow;
            if (job.opts.upscaleFactor > 1)
                historyLow = *image_downscale_box(prevDisplay, job.opts.upscaleFactor);
            else
                historyLow = prevDisplay;
            if (historyLow.width == working.width && historyLow.height == working.height) {
                Image reproj, conf;
                backend->reprojectHistory(historyLow, mv, reproj, conf);
                reprojLow = std::move(reproj);
                confidenceLow = std::move(conf);
                haveHistory = true;
            }
        }

        // 2. neural pass with the reprojected history as lanes 7-9, or the
        //    analytical path: temporal blend first, then spatial denoise.
        if (backend->hasNeuralGraph() && job.opts.useNeural) {
            auto geom = Geometry::fromValid(working.width, working.height,
                                            backend->model()->config().levels,
                                            backend->model()->config().minField,
                                            backend->model()->config().windowSize);
            if (!geom) { log_error("video: geometry rejected"); return false; }
            std::vector<uint16_t> features;
            preprocess_pack_features(*geom, working, haveHistory ? &reprojLow : nullptr,
                                     opts_to_cond(job.opts), job.opts.seed + frameIndex, features);
            std::vector<float> head(size_t(geom->fieldWidth) * geom->fieldHeight * 4, 0.f);
            if (!backend->runNeuralGraph(*geom, features.data(), head.data(),
                                         job.opts.seed + frameIndex)) {
                log_error("video: neural graph failed at frame %u", frameIndex);
                return false;
            }
            Image neural(working.width, working.height);
            head_composite(working, head.data(), haveHistory ? &reprojLow : nullptr,
                           backend->model()->blendScale(), neural);
            working = std::move(neural);
        } else {
            if (haveHistory) {
                Image blended;
                backend->temporalBlend(working, reprojLow, confidenceLow,
                                       job.opts.temporalMaxBlend, blended);
                working = std::move(blended);
            }
            if (job.opts.denoiseStrength > 0.0f) {
                Image dn;
                backend->denoiseSpatial(working, dn, job.opts.denoiseStrength);
                working = std::move(dn);
            }
        }

        // 3. upscale to output resolution
        Image out;
        if (!backend->upscaleSpatial(working, out, job.opts.upscaleFactor, job.opts.sharpen)) {
            log_error("video: upscale failed at frame %u", frameIndex);
            return false;
        }

        const auto t1 = std::chrono::steady_clock::now();
        if (frameIndex % 8 == 0)
            log_info("video: frame %u %.1f ms", frameIndex,
                     std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (!writer.write(out)) { log_error("video: write failed"); return false; }
        prevDisplay = std::move(out);
        prevLowLuma = std::move(working);
        ++frameIndex;
    }
    writer.close();
    log_info("video: wrote %u frames to %s", frameIndex, job.outputPath.c_str());
    return true;
}

} // namespace opendlss
