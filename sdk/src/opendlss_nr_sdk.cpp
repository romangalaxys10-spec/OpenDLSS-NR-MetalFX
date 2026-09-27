// C SDK implementation over the core pipeline.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss_nr_sdk.h"

#include "opendlss/backend.h"
#include "opendlss/pipeline.h"
#include "opendlss/version.h"

#include <cstring>
#include <string>
#include <chrono>
#include <cstdlib>

using namespace opendlss;

struct OpendlssNrInstance {
    std::unique_ptr<IBackend> backend;
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
    PipelineOptions opts;
    bool gameActive = false;
    std::string backendInfo;
    bool metalFxSpatial = false, metalFxTemporal = false;
    Image currentColor;
    MotionField currentMotion;
    GameFrameInput frameIn;
    GameFrameOutput frameOut;
};

extern "C" const char* opendlss_nr_sdk_version(void) {
    return OPENDLSS_NR_VERSION_STRING;
}

static BackendKind pick_backend(const char* name) {
    std::string s = name ? name : "auto";
    if (s == "cpu") return BackendKind::Cpu;
    if (s == "metal") return BackendKind::Metal;
    if (s == "d3d12") return BackendKind::D3D12;
    if (s == "vulkan") return BackendKind::Vulkan;
    // auto: first available non-CPU backend, else CPU
    auto list = available_backends();
    for (BackendKind k : list) if (k != BackendKind::Cpu) return k;
    return BackendKind::Cpu;
}

extern "C" OpendlssNrStatus opendlss_nr_create(const OpendlssNrCreateDesc* desc,
                                               OpendlssNrInstance** out) {
    if (!desc || !out || desc->renderWidth == 0 || desc->renderHeight == 0) {
        return OPENDLSS_NR_ERR_ARGUMENT;
    }
    if (desc->outputScale < 1 || desc->outputScale > 4) return OPENDLSS_NR_ERR_ARGUMENT;
    auto* inst = new OpendlssNrInstance();
    inst->backend = create_backend(pick_backend(desc->backend));
    if (!inst->backend) { delete inst; return OPENDLSS_NR_ERR_BACKEND; }
    if (!desc->modelDir) {
        // try the default model location next to the executable / repo layout
        for (const char* p : {"models/demo-nr", "../models/demo-nr"}) {
            std::FILE* f = std::fopen((std::string(p) + "/manifest.json").c_str(), "rb");
            if (f) { std::fclose(f); inst->backend->loadModel(p); break; }
        }
    } else if (!inst->backend->loadModel(desc->modelDir)) {
        delete inst;
        return OPENDLSS_NR_ERR_MODEL;
    }
    inst->renderW = desc->renderWidth;
    inst->renderH = desc->renderHeight;
    inst->outW = desc->renderWidth * desc->outputScale;
    inst->outH = desc->renderHeight * desc->outputScale;
    inst->opts.denoiseStrength = desc->denoiseStrength;
    inst->opts.sharpen = desc->sharpen;
    inst->opts.seed = desc->seed ? desc->seed : 0x5EED1234ull;
    inst->opts.upscaleFactor = desc->outputScale;

    const BackendInfo& bi = inst->backend->info();
    inst->backendInfo = bi.name + " / " + bi.deviceName;
    inst->metalFxSpatial = bi.metalFxSpatial;
    inst->metalFxTemporal = bi.metalFxTemporal;

    if (!inst->backend->beginGame(inst->renderW, inst->renderH, inst->outW, inst->outH)) {
        delete inst;
        return OPENDLSS_NR_ERR_INTERNAL;
    }
    inst->gameActive = true;
    *out = inst;
    return OPENDLSS_NR_OK;
}

extern "C" void opendlss_nr_destroy(OpendlssNrInstance* inst) {
    if (!inst) return;
    if (inst->gameActive && inst->backend) inst->backend->endGame();
    delete inst;
}

extern "C" OpendlssNrStatus opendlss_nr_resolve(OpendlssNrInstance* inst,
                                                const OpendlssNrFrameInput* in,
                                                OpendlssNrFrameOutput* out) {
    if (!inst || !inst->backend || !in || !out || !in->colorRGBA) {
        return OPENDLSS_NR_ERR_ARGUMENT;
    }
    if (!inst->gameActive) return OPENDLSS_NR_ERR_INTERNAL;
    if (!out->colorRGBA) return OPENDLSS_NR_ERR_ARGUMENT;

    inst->currentColor = Image(inst->renderW, inst->renderH);
    std::memcpy(inst->currentColor.pixels.data(), in->colorRGBA,
                size_t(inst->renderW) * inst->renderH * 4 * sizeof(float));

    inst->currentMotion.width = inst->renderW;
    inst->currentMotion.height = inst->renderH;
    const size_t mvCount = size_t(inst->renderW) * inst->renderH * 2;
    if (in->motionXY) {
        inst->currentMotion.xy.assign(in->motionXY, in->motionXY + mvCount);
    } else {
        inst->currentMotion.xy.assign(mvCount, 0.0f);
    }

    inst->frameIn = GameFrameInput{};
    inst->frameIn.color = &inst->currentColor;
    inst->frameIn.motion = &inst->currentMotion;
    inst->frameIn.depth = in->depth;
    inst->frameIn.exposure = in->exposure > 0 ? in->exposure : 1.0f;
    inst->frameIn.jitterX = in->jitterX;
    inst->frameIn.jitterY = in->jitterY;
    inst->frameIn.resetHistory = in->resetHistory != 0;

    if (!inst->backend->submitGameFrame(inst->frameIn, inst->frameOut)) {
        return OPENDLSS_NR_ERR_INTERNAL;
    }
    if (inst->frameOut.upscaled.width != inst->outW ||
        inst->frameOut.upscaled.height != inst->outH) {
        return OPENDLSS_NR_ERR_INTERNAL;
    }
    std::memcpy(out->colorRGBA, inst->frameOut.upscaled.pixels.data(),
                size_t(inst->outW) * inst->outH * 4 * sizeof(float));
    out->outputWidth = inst->outW;
    out->outputHeight = inst->outH;
    out->gpuMilliseconds = inst->frameOut.gpuMs;
    return OPENDLSS_NR_OK;
}

extern "C" OpendlssNrStatus opendlss_nr_output_size(const OpendlssNrInstance* inst,
                                                    uint32_t* w, uint32_t* h) {
    if (!inst || !w || !h) return OPENDLSS_NR_ERR_ARGUMENT;
    *w = inst->outW;
    *h = inst->outH;
    return OPENDLSS_NR_OK;
}

extern "C" OpendlssNrStatus opendlss_nr_backend_info(const OpendlssNrInstance* inst,
                                                     const char** info,
                                                     int* metalFxSpatial,
                                                     int* metalFxTemporal) {
    if (!inst || !info) return OPENDLSS_NR_ERR_ARGUMENT;
    *info = inst->backendInfo.c_str();
    if (metalFxSpatial) *metalFxSpatial = inst->metalFxSpatial ? 1 : 0;
    if (metalFxTemporal) *metalFxTemporal = inst->metalFxTemporal ? 1 : 0;
    return OPENDLSS_NR_OK;
}
