// Preprocess.metal — pack the 16-lane feature field.
// Lane order matches the reference preprocess (and core/src/pipeline.cpp):
//   0-2 Gaussian lanes (Box-Muller from hash of the PADDED coordinate + frame seed)
//   3   constant 1
//   4-6 proxy, centred: half((half(c) - 0.5) * 0.125)
//   7-9 reprojected history (copy of 4-6 without history)
//   10  style/128 · 11 tone · 12-14 structure/skin/automask · 15 0
// Outside the valid rect the image is mirrored; noise keeps using the padded coord.
//
// Dispatch: threads over the padded field.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "Common.metal"

kernel void odl_preprocess(
    texture2d<float, access::read>   proxy       [[texture(0)]],   // valid size, RGBA32Float
    texture2d<float, access::read>   history     [[texture(1)]],   // valid size or empty
    constant OdlConditioning&        cond        [[buffer(0)]],
    constant uint2&                  validSize   [[buffer(1)]],
    constant uint&                   frameSeed   [[buffer(2)]],
    device half*                     features    [[buffer(3)]],    // fieldH*fieldW*16
    uint2 gid [[thread_position_in_grid]])
{
    const uint FW = proxy.get_width();   // caller passes the field-sized proxy texture
    const uint FH = proxy.get_height();
    if (gid.x >= FW || gid.y >= FH) return;

    // three Gaussian lanes from the padded-coordinate hash
    uint h0 = odl_hash(gid.x, gid.y, 0u, frameSeed);
    uint h1 = odl_hash(gid.x, gid.y, 0x9E37u, frameSeed);
    float u1 = max(float(h0 >> 8) / 16777216.0f, 1e-9f);
    float u2 = float(h1 >> 8) / 16777216.0f;
    float r1 = sqrt(-2.0f * log(u1));
    float g0 = r1 * cos(6.2831853f * u2);
    float g1 = r1 * cos(6.2831853f * u2 + 2.0943951f);
    float g2 = r1 * cos(6.2831853f * u2 + 4.1887902f);

    // image sample mirrored off the valid rectangle
    int sx = odl_mirror((int)gid.x, validSize.x);
    int sy = odl_mirror((int)gid.y, validSize.y);
    sx = clamp(sx, 0, (int)validSize.x - 1);
    sy = clamp(sy, 0, (int)validSize.y - 1);
    float4 p = proxy.read(uint2(sx, sy));
    float4 q = history.get_width() ? history.read(uint2(sx, sy)) : p;

    device half* o = features + ((size_t)gid.y * FW + gid.x) * 16;
    o[0] = half(g0);
    o[1] = half(g1);
    o[2] = half(g2);
    o[3] = half(1.0h);
    for (int c = 0; c < 3; ++c)
        o[4 + c] = half(half(half(p[c]) - 0.5h) * 0.125h);
    for (int c = 0; c < 3; ++c)
        o[7 + c] = half(half(half(q[c]) - 0.5h) * 0.125h);
    o[10] = half(cond.style);
    o[11] = half(cond.tone);
    o[12] = half(cond.structure);
    o[13] = half(cond.skin);
    o[14] = half(cond.autoMask);
    o[15] = half(0.0h);
}

// input embedding: tokens = input_proj (fieldW x 16) applied per token
// Dispatch: threads over field tokens.
kernel void odl_input_embed(
    device const half* features   [[buffer(0)]],   // [field][16]
    device const half* proj       [[buffer(1)]],   // [C0][16] row-major
    device half*       state      [[buffer(2)]],   // [field][C0]
    constant uint2&    fieldSize  [[buffer(3)]],
    constant uint&     channels   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint tokens = fieldSize.x * fieldSize.y;
    if (gid >= tokens) return;
    device const half* f = features + (size_t)gid * 16;
    device half* out = state + (size_t)gid * channels;
    for (uint o = 0; o < channels; ++o) {
        float s = 0.f;
        device const half* w = proj + (size_t)o * 16;
        for (uint i = 0; i < 16; ++i) s += float(w[i]) * float(f[i]);
        out[o] = half(odl_pub_e4m3(s));
    }
}
