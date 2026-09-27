/* engine_example.c — minimal game-engine integration of the OpenDLSS-NR
 * MetalFX SDK. Renders a synthetic frame at half resolution with camera
 * motion + jitter and resolves it to full resolution through the backend
 * (MetalFX temporal scaler on Apple Silicon).
 *
 * Build:  see demos/README.md
 * Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
 */
#include "opendlss_nr_sdk.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define RW 640
#define RH 360
#define SCALE 2

int main(void) {
    OpendlssNrCreateDesc desc = {0};
    desc.renderWidth = RW;
    desc.renderHeight = RH;
    desc.outputScale = SCALE;
    desc.backend = "auto";
    desc.modelDir = NULL;             /* NULL = analytical path; "models/demo-nr" for neural */
    desc.denoiseStrength = 0.5f;
    desc.sharpen = 0.2f;
    desc.seed = 42;

    OpendlssNrInstance* inst = NULL;
    OpendlssNrStatus st = opendlss_nr_create(&desc, &inst);
    if (st != OPENDLSS_NR_OK) {
        fprintf(stderr, "create failed: %d\n", st);
        return 1;
    }

    const char* info = NULL;
    int fxS = 0, fxT = 0;
    opendlss_nr_backend_info(inst, &info, &fxS, &fxT);
    printf("backend: %s  MetalFX spatial=%d temporal=%d\n", info, fxS, fxT);

    uint32_t ow = 0, oh = 0;
    opendlss_nr_output_size(inst, &ow, &oh);
    printf("output: %ux%u\n", ow, oh);

    float* color = malloc(sizeof(float) * RW * RH * 4);
    float* motion = malloc(sizeof(float) * RW * RH * 2);
    float* depth = malloc(sizeof(float) * RW * RH);
    float* outColor = malloc(sizeof(float) * ow * oh * 4);

    const int frames = 120;
    double totalMs = 0;
    for (int f = 0; f < frames; ++f) {
        float t = (float)f / 60.0f;
        /* synthetic renderer: scrolling scene + moving light + exact camera motion */
        for (int y = 0; y < RH; ++y) {
            for (int x = 0; x < RW; ++x) {
                float fx = (float)x, fy = (float)y;
                float v = 0.5f + 0.4f * sinf((fx + t * 30.0f * 2.5f) * 0.05f)
                                  * sinf(fy * 0.07f + 1.3f);
                float lx = (float)RW * (0.5f + 0.3f * sinf(t * 3.1f));
                float ly = (float)RH * (0.5f + 0.3f * cosf(t * 2.3f));
                float dl = sqrtf((fx - lx) * (fx - lx) + (fy - ly) * (fy - ly));
                float light = expf(-dl / ((float)RW * 0.15f));
                float* px = color + ((size_t)y * RW + x) * 4;
                px[0] = v * 0.8f + light; px[1] = v * 0.9f + light * 0.8f;
                px[2] = v + light * 0.5f; px[3] = 1.0f;
                motion[((size_t)y * RW + x) * 2 + 0] = 2.5f;  /* camera pans +2.5 px/frame */
                motion[((size_t)y * RW + x) * 2 + 1] = 0.0f;
                depth[(size_t)y * RW + x] = 0.5f + 0.1f * sinf(fx * 0.01f);
            }
        }
        OpendlssNrFrameInput in = {0};
        in.colorRGBA = color;
        in.motionXY = motion;
        in.depth = depth;
        in.exposure = 1.0f;
        in.jitterX = (f % 2) ? 0.25f : -0.25f;   /* standard 2-phase jitter */
        in.jitterY = 0.0f;
        in.resetHistory = (f == 0);

        OpendlssNrFrameOutput out = {0};
        out.colorRGBA = outColor;
        st = opendlss_nr_resolve(inst, &in, &out);
        if (st != OPENDLSS_NR_OK) {
            fprintf(stderr, "resolve failed at frame %d: %d\n", f, st);
            return 1;
        }
        totalMs += out.gpuMilliseconds;
    }
    printf("%d frames, %.2f ms/frame average\n", frames, totalMs / frames);

    free(color); free(motion); free(depth); free(outColor);
    opendlss_nr_destroy(inst);
    return 0;
}
