// Media.metal — the analytical/media kernels: bilateral denoise, Lanczos3
// upscale fallback (MetalFX is used when available), sharpen, motion
// estimation, reprojection and temporal blend.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "Common.metal"

// ---------------- bilateral denoise (edge-aware, luma-guided) ----------------
kernel void odl_denoise(
    texture2d<float, access::read> src  [[texture(0)]],
    constant float&                strength [[buffer(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height()) return;
    float4 c = src.read(gid);
    const float sigmaColor = 0.08f + 0.12f * strength;
    const int radius = strength > 0.75f ? 3 : 2;
    float cl = dot(c.rgb, float3(0.2126, 0.7152, 0.0722));
    float3 acc = 0; float wsum = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            uint2 p((int)gid.x + dx, (int)gid.y + dy);
            p = clamp(p, uint2(0), uint2(src.get_width()-1, src.get_height()-1));
            float4 n = src.read(p);
            float3 dc = n.rgb - c.rgb;
            float dl = dot(n.rgb, float3(0.2126, 0.7152, 0.0722)) - cl;
            float w = exp(-dot(dc,dc) / (2*sigmaColor*sigmaColor)
                          - dl*dl / (2*sigmaColor*sigmaColor*0.25f));
            acc += n.rgb * w; wsum += w;
        }
    }
    dst.write(float4(clamp(acc / wsum, 0.0f, 1.0f), 1.0f), gid);
}

// ---------------- Lanczos3 resampling (separable passes) ----------------
static inline float odl_lanczos3(float x) {
    x = fabs(x);
    if (x < 1e-6f) return 1.0f;
    if (x >= 3.0f) return 0.0f;
    const float pi = 3.14159265358979f;
    return 3.0f * sin(pi*x) * sin(pi*x/3.0f) / (pi*pi*x*x);
}

kernel void odl_upscale_h(                     // horizontal pass, f32 texture -> f32
    texture2d<float, access::read> src [[texture(0)]],
    constant uint&  dstW               [[buffer(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dstW || gid.y >= src.get_height()) return;
    const float scale = float(src.get_width()) / float(dstW);
    const float support = 3.0f;
    float cx = (float(gid.x) + 0.5f) * scale - 0.5f;
    int lo = int(floor(cx - support + 0.5f)), hi = int(floor(cx + support + 0.5f));
    float4 acc = 0; float wsum = 0;
    for (int s = lo; s <= hi; ++s) {
        float w = odl_lanczos3((float(s) - cx) * min(scale, 1.0f));
        if (w == 0) continue;
        uint sx = clamp(uint(s), 0u, src.get_width() - 1);
        acc += src.read(uint2(sx, gid.y)) * w; wsum += w;
    }
    dst.write(acc / wsum, gid);
}

kernel void odl_upscale_v(
    texture2d<float, access::read> src [[texture(0)]],
    constant uint&  dstH               [[buffer(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= dstH) return;
    const float scale = float(src.get_height()) / float(dstH);
    const float support = 3.0f;
    float cy = (float(gid.y) + 0.5f) * scale - 0.5f;
    int lo = int(floor(cy - support + 0.5f)), hi = int(floor(cy + support + 0.5f));
    float4 acc = 0; float wsum = 0;
    for (int s = lo; s <= hi; ++s) {
        float w = odl_lanczos3((float(s) - cy) * min(scale, 1.0f));
        if (w == 0) continue;
        uint sy = clamp(uint(s), 0u, src.get_height() - 1);
        acc += src.read(uint2(gid.x, sy)) * w; wsum += w;
    }
    dst.write(acc / wsum, gid);
}

// ---------------- adaptive sharpen (CAS-style) ----------------
kernel void odl_sharpen(
    texture2d<float, access::read> src   [[texture(0)]],
    constant float& amount               [[buffer(0)]],
    texture2d<float, access::write> dst  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height()) return;
    float4 c = src.read(gid);
    float3 blur = 0; float wsum = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (!dx && !dy) continue;
            uint2 p(clamp((int)gid.x+dx, 0, (int)src.get_width()-1),
                    clamp((int)gid.y+dy, 0, (int)src.get_height()-1));
            float w = (dx && dy) ? 1.0f : 2.0f;
            blur += src.read(p).rgb * w; wsum += w;
        }
    blur /= wsum;
    float lo = min(min(c.r, c.g), c.b), hi = max(max(c.r, c.g), c.b);
    float adapt = 1.0f - clamp((hi - lo) * 3.0f, 0.0f, 1.0f);
    dst.write(float4(clamp(c.rgb + (c.rgb - blur) * amount * adapt, 0.0f, 1.0f), 1.0f), gid);
}

// ---------------- motion: 16x16 block matching, +/-6 search ----------------
kernel void odl_motion(
    texture2d<float, access::read> prev [[texture(0)]],   // luma in rgb
    texture2d<float, access::read> curr [[texture(1)]],
    texture2d<float, access::write> mv  [[texture(2)]],   // RG32Float (dx, dy)
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= curr.get_width() || gid.y >= curr.get_height()) return;
    const int block = 16, search = 6;
    float best = 1e30f; int bestDx = 0, bestDy = 0;
    for (int dy = -search; dy <= search; dy += 2)
        for (int dx = -search; dx <= search; dx += 2) {
            float sad = 0;
            for (int y = 0; y < block; y += 2) {
                int cy = gid.y + y; if (cy >= (int)curr.get_height()) break;
                for (int x = 0; x < block; x += 2) {
                    int cx = gid.x + x; if (cx >= (int)curr.get_width()) break;
                    float a = curr.read(uint2(cx, cy)).r;
                    uint2 p(clamp(cx+dx,0,(int)prev.get_width()-1), clamp(cy+dy,0,(int)prev.get_height()-1));
                    sad += fabs(a - prev.read(p).r);
                }
            }
            if (sad < best) { best = sad; bestDx = dx; bestDy = dy; }
        }
    mv.write(float4(float(bestDx), float(bestDy), 0, 0), gid);
}

// ---------------- reprojection with bilinear confidence ----------------
kernel void odl_reproject(
    texture2d<float, access::read> history [[texture(0)]],
    texture2d<float, access::read> mv      [[texture(1)]],   // RG32Float
    texture2d<float, access::write> reproj [[texture(2)]],
    texture2d<float, access::write> conf   [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= mv.get_width() || gid.y >= mv.get_height()) return;
    float2 m = mv.read(gid).rg;
    float2 s = float2(gid) - m;                     // where this pixel was
    bool oob = s.x < -0.5f || s.y < -0.5f ||
               s.x > float(history.get_width()) - 0.5f ||
               s.y > float(history.get_height()) - 0.5f;
    if (oob || history.get_width() == 0) {
        reproj.write(float4(0, 0, 0, 1), gid);
        conf.write(float4(0, 0, 0, 1), gid);
        return;
    }
    float2 sp = clamp(s, float2(0), float2(history.get_width()-1, history.get_height()-1));
    int2 i0(int(floor(sp.x)), int(floor(sp.y)));
    float2 f(sp.x - i0.x, sp.y - i0.y);
    int2 i1(min(i0.x+1, (int)history.get_width()-1), min(i0.y+1, (int)history.get_height()-1));
    float4 h00 = history.read(uint2(i0.x, i0.y)), h10 = history.read(uint2(i1.x, i0.y));
    float4 h01 = history.read(uint2(i0.x, i1.y)), h11 = history.read(uint2(i1.x, i1.y));
    float4 r = h00*(1-f.x)*(1-f.y) + h10*f.x*(1-f.y) + h01*(1-f.x)*f.y + h11*f.x*f.y;
    reproj.write(float4(clamp(r.rgb,0.0f,1.0f), 1), gid);
    float edge = max(fabs(f.x - 0.5f), fabs(f.y - 0.5f));
    conf.write(float4(clamp(1.0f - 1.5f * max(0.0f, edge - 0.25f), 0.0f, 1.0f), 0, 0, 1), gid);
}

// ---------------- temporal blend ----------------
kernel void odl_temporal_blend(
    texture2d<float, access::read> current [[texture(0)]],
    texture2d<float, access::read> reproj  [[texture(1)]],
    texture2d<float, access::read> conf    [[texture(2)]],
    constant float& maxBlend               [[buffer(0)]],
    texture2d<float, access::write> dst    [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= current.get_width() || gid.y >= current.get_height()) return;
    float4 c = current.read(gid);
    float4 r = reproj.read(gid);
    float w = min(conf.read(gid).r, maxBlend);
    dst.write(float4(clamp(c.rgb * (1-w) + r.rgb * w, 0.0f, 1.0f), 1.0f), gid);
}

// ---------------- head composite ----------------
//   neural  = clamp(proxy + rgb/4, 0, 1)
//   weight  = clamp(sigmoid(logit) * blendScale, 0, 1)
//   display = lerp(neural, history, weight)
kernel void odl_head_composite(
    texture2d<float, access::read> proxy   [[texture(0)]],
    texture2d<float, access::read> headTex [[texture(1)]],   // field-res RGBA32Float
    texture2d<float, access::read> history [[texture(2)]],   // may be empty
    constant float& blendScale             [[buffer(0)]],
    texture2d<float, access::write> dst    [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= proxy.get_width() || gid.y >= proxy.get_height()) return;
    float4 p = proxy.read(gid);
    float4 h = headTex.read(gid);          // rgb = residual/4 domain handled below
    float neural[3], outv[3];
    for (int c = 0; c < 3; ++c)
        neural[c] = clamp(p[c] + h[c] / 4.0f, 0.0f, 1.0f);
    if (history.get_width()) {
        float w = clamp(1.0f / (1.0f + exp(-h.a)) * blendScale, 0.0f, 1.0f);
        float4 q = history.read(gid);
        for (int c = 0; c < 3; ++c) outv[c] = neural[c] * (1-w) + q[c] * w;
    } else {
        for (int c = 0; c < 3; ++c) outv[c] = neural[c];
    }
    dst.write(float4(outv[0], outv[1], outv[2], 1.0f), gid);
}
