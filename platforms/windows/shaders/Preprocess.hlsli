// Preprocess.hlsli — 16-lane feature packing and the input embedding.
// Port of platforms/macos/shaders/Preprocess.metal with the CPU reference
// (core/src/pipeline.cpp preprocess_pack_features, core/backends/cpu/
// neural_graph_cpu.cpp input embedding) as the numeric golden:
//
//   lane 0-2   Gaussian lanes (Box-Muller from the hash of the PADDED coord)
//   lane 3     constant 1
//   lane 4-6   proxy, centred: half((half(clamp(c,0,1)) - 0.5) * 0.125)
//   lane 7-9   reprojected history, else copy of 4-6
//   lane 10    style       (conditioning)
//   lane 11    tone
//   lane 12-14 structure / skin / auto-mask
//   lane 15    0
//
// Outside the valid rect the image is mirrored (2*valid - x - 2, no edge
// repeat); the noise lanes keep using the padded coordinate.
//
// Root signature (buffers/graphics shared "rsBuffers"):
//   t0 proxy    (field-sized RGBA32Float; valid rect at the origin)
//   t1 history  (valid-sized RGBA32Float, or a 1x1 dummy)
//   u0 features [fieldTokens][16] f16 packed
//   t4 features (as SRV for the embed pass), t5 input_proj [C0][16] f16
//   u1 state    [fieldTokens][C0] f16 packed
//   b0 root constants P[16].
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "Common.hlsli"

Texture2D<float4>       OdlProxy    : register(t0);
Texture2D<float4>       OdlHistory  : register(t1);
RWByteAddressBuffer     OdlFeaturesW: register(u0);

ByteAddressBuffer       OdlFeatR    : register(t4);
ByteAddressBuffer       OdlInputProj: register(t5);
RWByteAddressBuffer     OdlStateW   : register(u1);

cbuffer OdlParams : register(b0)
{
    uint4 P[16];
};

// ---------------- odl_preprocess ----------------
// P[0] = (validW, validH, hasHistory, fieldW)
// P[1] = (frameSeedLo, frameSeedHi, -, -)
// P[2] = (bits(style), bits(tone), bits(structure), bits(skin))
// P[3] = (bits(autoMask), -, -, -)
// Dispatch: one thread per padded-field token (fieldW * fieldH).

[numthreads(8, 8, 1)]
void odl_preprocess(uint3 dtid : SV_DispatchThreadID)
{
    const uint fieldW = P[0].w;
    const uint validW = P[0].x, validH = P[0].y;
    const uint px = dtid.x, py = dtid.y;
    if (px >= fieldW) return;                    // grid padded to 8x8 tiles

    const uint2 seed = uint2(P[1].x, P[1].y);

    // three Gaussian lanes from the padded-coordinate hash (frameIndex 0)
    uint  h0 = OdlHash(px, py, 0u, seed);
    uint  h1 = OdlHash(px, py, 0x9E37u, seed);
    float u1 = float(h0 >> 8) / 16777216.0f;
    float u2 = float(h1 >> 8) / 16777216.0f;
    u1 = max(u1, 1e-9f);
    float r1 = sqrt(-2.0f * log(u1));
    float g0 = r1 * cos(6.2831853f * u2);
    float g1 = r1 * cos(6.2831853f * u2 + 2.0943951f);
    float g2 = r1 * cos(6.2831853f * u2 + 4.1887902f);

    // image sample mirrored off the valid rectangle, then clamped into it
    int sx = OdlMirror(int(px), validW);
    int sy = OdlMirror(int(py), validH);
    sx = clamp(sx, 0, int(validW) - 1);
    sy = clamp(sy, 0, int(validH) - 1);
    float4 p = OdlProxy.Load(uint3(uint(sx), uint(sy), 0u));
    float4 q = P[0].z != 0u
             ? OdlHistory.Load(uint3(uint(sx), uint(sy), 0u))
             : p;

    const uint t = py * fieldW + px;
    const uint base = t * 16u;                   // 16 halfs = 8 dwords, even base
    float lanes[16];
    lanes[0] = g0;
    lanes[1] = g1;
    lanes[2] = g2;
    lanes[3] = 1.0f;
    [unroll]
    for (int c = 0; c < 3; ++c)
        lanes[4 + c] = OdlHalf(OdlHalf(clamp(p[c], 0.0f, 1.0f) - 0.5f) * 0.125f);
    [unroll]
    for (int c2 = 0; c2 < 3; ++c2)
        lanes[7 + c2] = OdlHalf(OdlHalf(clamp(q[c2], 0.0f, 1.0f) - 0.5f) * 0.125f);
    lanes[10] = asfloat(P[2].x);
    lanes[11] = asfloat(P[2].y);
    lanes[12] = asfloat(P[2].z);
    lanes[13] = asfloat(P[2].w);
    lanes[14] = asfloat(P[3].x);
    lanes[15] = 0.0f;

    [unroll]
    for (uint d = 0; d < 8u; ++d)
        OdlStoreHalf2(OdlFeaturesW, base / 2u + d, lanes[2 * d], lanes[2 * d + 1]);
}

// ---------------- odl_input_embed ----------------
// state[t][o] = pub_e4m3(sum_i input_proj[o][i] * features[t][i])
// P[0] = (tokens, C, -, -)
// Dispatch: one thread per token.

[numthreads(64, 1, 1)]
void odl_input_embed(uint3 dtid : SV_DispatchThreadID)
{
    const uint tokens = P[0].x;
    const uint C      = P[0].y;
    const uint t = dtid.x;
    if (t >= tokens) return;

    [loop] for (uint o2 = 0; o2 < C; o2 += 2u)
    {
        float acc0 = 0.0f, acc1 = 0.0f;
        [loop] for (uint i = 0; i < 16u; ++i)
        {
            float f  = OdlLoadHalf(OdlFeatR, t * 16u + i);
            acc0 += OdlLoadHalf(OdlInputProj, o2 * 16u + i)       * f;
            acc1 += OdlLoadHalf(OdlInputProj, (o2 + 1u) * 16u + i) * f;
        }
        OdlStoreHalf2(OdlStateW, (t * C + o2) / 2u, OdlPubE4m3(acc0), OdlPubE4m3(acc1));
    }
}
