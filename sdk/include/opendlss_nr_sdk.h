/* OpenDLSS-NR MetalFX — C integration SDK for game engines and video apps.
 *
 * A stable C ABI over the core pipeline. Engines call:
 *      opendlss_nr_create()            once, with the output scale
 *      opendlss_nr_begin_frame()       each frame, with frame inputs
 *      opendlss_nr_submit()            the rendered color (+ depth/motion)
 *      opendlss_nr_resolve()           the upscaled/denoised output
 *      opendlss_nr_end_frame()
 *      opendlss_nr_destroy()
 *
 * Threading: calls must come from one thread at a time (the GPU context is
 * not internally synchronized). Backend selection: "auto" picks Metal on
 * macOS, D3D12 on Windows, Vulkan on Linux, CPU everywhere else / as
 * fallback. Model directories follow docs/WEIGHTS.md.
 *
 * Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
 */
#ifndef OPENDLSS_NR_SDK_H
#define OPENDLSS_NR_SDK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENDLSS_NR_SDK_VERSION 1

typedef struct OpendlssNrInstance OpendlssNrInstance;

typedef enum OpendlssNrStatus {
    OPENDLSS_NR_OK = 0,
    OPENDLSS_NR_ERR_ARGUMENT = 1,
    OPENDLSS_NR_ERR_BACKEND = 2,
    OPENDLSS_NR_ERR_MODEL = 3,
    OPENDLSS_NR_ERR_INTERNAL = 4
} OpendlssNrStatus;

typedef struct OpendlssNrCreateDesc {
    uint32_t renderWidth;        /* the resolution the engine renders at      */
    uint32_t renderHeight;
    uint32_t outputScale;        /* 1, 2, 3 or 4                              */
    const char* backend;         /* "auto" | "metal" | "d3d12" | "vulkan" | "cpu" */
    const char* modelDir;        /* model directory or NULL (analytical path) */
    float denoiseStrength;       /* 0..1, used when no model is loaded        */
    float sharpen;               /* 0..1                                      */
    uint64_t seed;               /* noise seed base (0 = default)             */
} OpendlssNrCreateDesc;

typedef struct OpendlssNrFrameInput {
    const float* colorRGBA;      /* render-res, row-major, [y*w+x]*4, 0..1    */
    const float* depth;          /* render-res linear depth 0..1, may be NULL */
    const float* motionXY;       /* render-res prev->curr, [y*w+x]*2, may be NULL */
    float exposure;              /* pre-exposure factor (0 = 1.0)             */
    float jitterX, jitterY;      /* sub-pixel jitter used when rendering      */
    int resetHistory;            /* camera cut / first frame / resize         */
    /* conditioning for the neural graph (0.5 defaults when all zero)         */
    float style, tone, structure, skin, autoMask;
} OpendlssNrFrameInput;

typedef struct OpendlssNrFrameOutput {
    float* colorRGBA;            /* output-res, caller-allocated              */
    uint32_t outputWidth;
    uint32_t outputHeight;
    double gpuMilliseconds;      /* wall-clock execution time                 */
} OpendlssNrFrameOutput;

const char* opendlss_nr_sdk_version(void);

OpendlssNrStatus opendlss_nr_create(const OpendlssNrCreateDesc* desc,
                                    OpendlssNrInstance** out);
void opendlss_nr_destroy(OpendlssNrInstance* inst);

/* Resolve one frame. colorRGBA input is consumed; output.colorRGBA is filled. */
OpendlssNrStatus opendlss_nr_resolve(OpendlssNrInstance* inst,
                                     const OpendlssNrFrameInput* in,
                                     OpendlssNrFrameOutput* out);

/* Query the resolved output dimensions (renderSize * outputScale). */
OpendlssNrStatus opendlss_nr_output_size(const OpendlssNrInstance* inst,
                                         uint32_t* w, uint32_t* h);

/* Human-readable backend description for diagnostics / on-screen stats. */
OpendlssNrStatus opendlss_nr_backend_info(const OpendlssNrInstance* inst,
                                          const char** info,
                                          int* metalFxSpatial,
                                          int* metalFxTemporal);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENDLSS_NR_SDK_H */
