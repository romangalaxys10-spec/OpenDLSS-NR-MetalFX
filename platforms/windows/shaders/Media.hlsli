// Media.hlsli — the analytical/media kernels: bilateral denoise, Lanczos3
// upscale (separable), CAS-style sharpen, 16x16 block-matching motion
// estimation, reprojection with bilinear confidence, temporal blend and the
// head composite. Ports of platforms/macos/shaders/Media.metal with the CPU
// reference (core/backends/cpu/cpu_backend.cpp, core/src/pipeline.cpp) as the
// numeric golden — where they differ this file follows the CPU:
//   * odl_motion matches the CPU's per-16x16-block matching (search +-6 step
//     2, block sampling step 2, first-best on ties) instead of Metal's
//     per-pixel variant, and uses luma(rgb) like the CPU
//   * odl_upscale_v clamps to [0,1] like the CPU's vertical pass; the
//     horizontal pass stays unclamped; Lanczos weights use the CPU's
//     divide-by-min(scale,1) form
//   * odl_reproject writes the CPU's unclamped bilinear reprojection
//
// Root signature "rsTextures":
//   t0..t3  Texture2D<float4> inputs
//   u0..u3  RWTexture2D<float4> outputs, u4 RWTexture2D<float2> (motion)
//   b0      root constants C[8] (32 floats)
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "Common.hlsli"

Texture2D<float4>         Tex0 : register(t0);
Texture2D<float4>         Tex1 : register(t1);
Texture2D<float4>         Tex2 : register(t2);
Texture2D<float4>         Tex3 : register(t3);

RWTexture2D<float4>       Out0 : register(u0);
RWTexture2D<float4>       Out1 : register(u1);
RWTexture2D<float4>       Out2 : register(u2);
RWTexture2D<float4>       Out3 : register(u3);
RWTexture2D<float2>       OutMv: register(u4);

cbuffer OdlMediaParams : register(b0)
{
    float4 C[8];
}

static const float3 kLumaW = float3(0.2126, 0.7152, 0.0722);

float4 OdlTexLoad(Texture2D<float4> tex, uint2 p, uint2 size)
{
    p = min(p, size - 1u);
    return tex.Load(uint3(p, 0u));
}

// ---------------- bilateral denoise (edge-aware, luma-guided) ----------------
// C[0].x = strength
[numthreads(8, 8, 1)]
void odl_denoise(uint3 dtid : SV_DispatchThreadID)
{
    const uint2 size = uint2(C[1].x, C[1].y);       // source size
    uint2 gid = dtid.xy;
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 c = OdlTexLoad(Tex0, gid, size);
    const float sigmaColor = 0.08f + 0.12f * C[0].x;
    const int radius = C[0].x > 0.75f ? 3 : 2;
    float cl = dot(c.rgb, kLumaW);
    float3 acc = 0; float wsum = 0;
    [loop]
    for (int dy = -radius; dy <= radius; ++dy)
    {
        [loop]
        for (int dx = -radius; dx <= radius; ++dx)
        {
            float4 n = OdlTexLoad(Tex0, uint2(gid.x + dx, gid.y + dy), size);
            float3 dc = n.rgb - c.rgb;
            float dl = dot(n.rgb, kLumaW) - cl;
            float w = exp(-dot(dc, dc) / (2 * sigmaColor * sigmaColor)
                          - dl * dl / (2 * sigmaColor * sigmaColor * 0.25f));
            acc += n.rgb * w; wsum += w;
        }
    }
    Out0[gid] = float4(clamp(acc / wsum, 0.0f, 1.0f), 1.0f);
}

// ---------------- Lanczos3 resampling (separable passes) ----------------
float OdlLanczos3(float x)
{
    x = abs(x);
    if (x < 1e-6f) return 1.0f;
    if (x >= 3.0f) return 0.0f;
    const float pi = 3.14159265358979f;
    return 3.0f * sin(pi * x) * sin(pi * x / 3.0f) / (pi * pi * x * x);
}

// horizontal pass: C[0].x = dstW ; C[1] = (srcW, srcH, dstH, -)
[numthreads(8, 8, 1)]
void odl_upscale_h(uint3 dtid : SV_DispatchThreadID)
{
    const uint dstW = uint(C[0].x);
    const uint2 srcSize = uint2(C[1].x, C[1].y);
    const uint dstH = uint(C[1].z);
    const uint2 gid = dtid.xy;
    if (gid.x >= dstW || gid.y >= dstH) return;

    const float scale = float(srcSize.x) / float(dstW);
    const float support = 3.0f;
    float cx = (float(gid.x) + 0.5f) * scale - 0.5f;
    int lo = int(floor(cx - support + 0.5f)), hi = int(floor(cx + support + 0.5f));
    float4 acc = 0; float wsum = 0;
    [loop]
    for (int s = lo; s <= hi; ++s)
    {
        float w = OdlLanczos3((float(s) - cx) / min(scale, 1.0f));
        if (w == 0) continue;
        uint sx = clamp(uint(s), 0u, srcSize.x - 1u);
        acc += OdlTexLoad(Tex0, uint2(sx, gid.y), srcSize) * w; wsum += w;
    }
    Out0[gid] = acc / wsum;
}

// vertical pass: C[0].x = dstH ; C[1] = (srcW, srcH, -, -); clamps like CPU
[numthreads(8, 8, 1)]
void odl_upscale_v(uint3 dtid : SV_DispatchThreadID)
{
    const uint dstH = uint(C[0].x);
    const uint2 srcSize = uint2(C[1].x, C[1].y);
    const uint dstW = uint(C[1].z);
    const uint2 gid = dtid.xy;
    if (gid.x >= dstW || gid.y >= dstH) return;

    const float scale = float(srcSize.y) / float(dstH);
    const float support = 3.0f;
    float cy = (float(gid.y) + 0.5f) * scale - 0.5f;
    int lo = int(floor(cy - support + 0.5f)), hi = int(floor(cy + support + 0.5f));
    float4 acc = 0; float wsum = 0;
    [loop]
    for (int s = lo; s <= hi; ++s)
    {
        float w = OdlLanczos3((float(s) - cy) / min(scale, 1.0f));
        if (w == 0) continue;
        uint sy = clamp(uint(s), 0u, srcSize.y - 1u);
        acc += OdlTexLoad(Tex0, uint2(gid.x, sy), srcSize) * w; wsum += w;
    }
    float4 o = acc / wsum;
    Out0[gid] = float4(clamp(o.rgb, 0.0f, 1.0f), 1.0f);
}

// ---------------- adaptive sharpen (CAS-style) ----------------
// C[0].x = amount ; C[1] = (w, h, -, -)
[numthreads(8, 8, 1)]
void odl_sharpen(uint3 dtid : SV_DispatchThreadID)
{
    const uint2 size = uint2(C[1].x, C[1].y);
    uint2 gid = dtid.xy;
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 c = OdlTexLoad(Tex0, gid, size);
    float3 blur = 0; float wsum = 0;
    [loop]
    for (int dy = -1; dy <= 1; ++dy)
        [loop]
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            float w = (dx != 0 && dy != 0) ? 1.0f : 2.0f;
            blur += OdlTexLoad(Tex0, uint2(gid.x + dx, gid.y + dy), size).rgb * w;
            wsum += w;
        }
    blur /= wsum;
    float lo = min(min(c.r, c.g), c.b), hi = max(max(c.r, c.g), c.b);
    float adapt = 1.0f - clamp((hi - lo) * 3.0f, 0.0f, 1.0f);
    Out0[gid] = float4(clamp(c.rgb + (c.rgb - blur) * C[0].x * adapt, 0.0f, 1.0f), 1.0f);
}

// ---------------- motion: 16x16 block matching, +/-6 search ----------------
// One thread per 16x16 block (CPU semantics: one motion vector per block).
// C[1] = (currW, currH, prevW, prevH)
// Blocks at the right/bottom edge clamp into the frame, exactly like the CPU.
[numthreads(8, 8, 1)]
void odl_motion(uint3 dtid : SV_DispatchThreadID)
{
    const uint W = uint(C[1].x), H = uint(C[1].y);
    const uint PW = uint(C[1].z), PH = uint(C[1].w);
    const uint bx = dtid.x, by = dtid.y;
    if (bx * 16u >= W || by * 16u >= H) return;

    const int block = 16, search = 6;
    float best = 1e30f; int bestDx = 0, bestDy = 0;
    [loop]
    for (int dy = -search; dy <= search; dy += 2)
    {
        [loop]
        for (int dx = -search; dx <= search; dx += 2)
        {
            float sad = 0;
            bool finished = false;
            [loop]
            for (int y = 0; y < block && !finished; y += 2)
            {
                int cy = int(by * 16u) + y;
                if (cy >= int(H)) { finished = true; break; }
                [loop]
                for (int x = 0; x < block; x += 2)
                {
                    int cx = int(bx * 16u) + x;
                    if (cx >= int(W)) break;
                    float a = dot(OdlTexLoad(Tex1, uint2(cx, cy), uint2(W, H)).rgb, kLumaW);
                    float b = dot(OdlTexLoad(Tex0, uint2(cx + dx, cy + dy), uint2(PW, PH)).rgb, kLumaW);
                    sad += abs(a - b);
                }
            }
            if (sad < best) { best = sad; bestDx = dx; bestDy = dy; }
        }
    }
    // one vector for every pixel of the block (clamped at the frame edges)
    for (uint py = by * 16u; py < min(by * 16u + 16u, H); ++py)
        for (uint px = bx * 16u; px < min(bx * 16u + 16u, W); ++px)
            OutMv[uint2(px, py)] = float2(float(bestDx), float(bestDy));
}

// ---------------- reprojection with bilinear confidence ----------------
// C[0].x = hasHistory ; C[1] = (w, h, histW, histH)
[numthreads(8, 8, 1)]
void odl_reproject(uint3 dtid : SV_DispatchThreadID)
{
    const uint W = uint(C[1].x), H = uint(C[1].y);
    const uint HW = uint(C[1].z), HH = uint(C[1].w);
    const uint2 gid = dtid.xy;
    if (gid.x >= W || gid.y >= H) return;

    float2 m = OdlTexLoad(Tex1, gid, uint2(W, H)).rg;
    float2 s = float2(gid) - m;                     // where this pixel was
    bool oob = !bool(C[0].x) ||
               s.x < -0.5f || s.y < -0.5f ||
               s.x > float(HW) - 0.5f || s.y > float(HH) - 0.5f;
    if (oob)
    {
        Out2[gid] = float4(0, 0, 0, 1);
        Out3[gid] = float4(0, 0, 0, 1);
        return;
    }
    float2 sp = clamp(s, float2(0, 0), float2(float(HW) - 1.0f, float(HH) - 1.0f));
    int2 i0 = int2(floor(sp.x), floor(sp.y));
    float2 f = sp - float2(i0);
    int2 i1 = int2(min(i0.x + 1, int(HW) - 1), min(i0.y + 1, int(HH) - 1));
    float4 h00 = OdlTexLoad(Tex0, uint2(i0), uint2(HW, HH));
    float4 h10 = OdlTexLoad(Tex0, uint2(i1.x, i0.y), uint2(HW, HH));
    float4 h01 = OdlTexLoad(Tex0, uint2(i0.x, i1.y), uint2(HW, HH));
    float4 h11 = OdlTexLoad(Tex0, uint2(i1), uint2(HW, HH));
    float4 r = h00 * (1 - f.x) * (1 - f.y) + h10 * f.x * (1 - f.y)
             + h01 * (1 - f.x) * f.y + h11 * f.x * f.y;
    Out2[gid] = float4(r.rgb, 1.0f);                // CPU leaves this unclamped
    float edge = max(abs(f.x - 0.5f), abs(f.y - 0.5f));
    Out3[gid] = float4(clamp(1.0f - 1.5f * max(0.0f, edge - 0.25f), 0.0f, 1.0f), 0, 0, 1);
}

// ---------------- temporal blend ----------------
// C[0].x = maxBlend ; C[1] = (w, h, -, -)
[numthreads(8, 8, 1)]
void odl_temporal_blend(uint3 dtid : SV_DispatchThreadID)
{
    const uint2 size = uint2(C[1].x, C[1].y);
    uint2 gid = dtid.xy;
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 c = OdlTexLoad(Tex0, gid, size);
    float4 r = OdlTexLoad(Tex1, gid, size);
    float w = min(OdlTexLoad(Tex2, gid, size).r, C[0].x);
    Out0[gid] = float4(clamp(c.rgb * (1 - w) + r.rgb * w, 0.0f, 1.0f), 1.0f);
}

// ---------------- head composite ----------------
//   neural  = clamp(proxy + rgb/4, 0, 1)
//   weight  = clamp(sigmoid(logit) * blendScale, 0, 1)
//   display = lerp(neural, history, weight)
// C[0].x = blendScale ; C[0].y = hasHistory ; C[1] = (w, h, -, -)
[numthreads(8, 8, 1)]
void odl_head_composite(uint3 dtid : SV_DispatchThreadID)
{
    const uint2 size = uint2(C[1].x, C[1].y);
    uint2 gid = dtid.xy;
    if (gid.x >= size.x || gid.y >= size.y) return;

    float4 p = OdlTexLoad(Tex0, gid, size);
    float4 h = OdlTexLoad(Tex1, gid, size);         // rgb = residual/4 domain, a = logit
    float neural[3], outv[3];
    [unroll]
    for (int c = 0; c < 3; ++c)
        neural[c] = clamp(p[c] + h[c] / 4.0f, 0.0f, 1.0f);
    if (C[0].y != 0.0f)
    {
        float w = clamp(OdlSigmoid(h.a) * C[0].x, 0.0f, 1.0f);
        float4 q = OdlTexLoad(Tex2, gid, size);
        [unroll]
        for (int c2 = 0; c2 < 3; ++c2) outv[c2] = neural[c2] * (1 - w) + q[c2] * w;
    }
    else
    {
        [unroll]
        for (int c3 = 0; c3 < 3; ++c3) outv[c3] = neural[c3];
    }
    Out0[gid] = float4(outv[0], outv[1], outv[2], 1.0f);
}
