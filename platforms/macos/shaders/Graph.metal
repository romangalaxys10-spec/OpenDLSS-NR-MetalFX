// Graph.metal — the transformer block kernels: FFN, window attention
// (4-phase shifted windows), global ViT attention, block epilogue, and the
// U-net transitions. Layouts match the CPU reference exactly:
//   state  [tokens][C] half (E4M3-published values)
//   weights [out][in] row-major half (host pre-decoded from E4M3)
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "Common.metal"

// ---------------------------------------------------------------------------
// FFN: C=32 dense 32->128->32 ; wide C: C/32 paths C->128->32 + W3 ; ViT dense.
// Dispatch: one threadgroup per token, 32 threads (one per lane group).
// ---------------------------------------------------------------------------
kernel void odl_ffn(
    device const half* x      [[buffer(0)]],    // [tokens][C]
    device const half* w1     [[buffer(1)]],
    device const half* b1     [[buffer(2)]],
    device const half* w2     [[buffer(3)]],
    device const half* b2     [[buffer(4)]],
    device const half* w3     [[buffer(5)]],    // null for dense blocks
    device const half* b3     [[buffer(6)]],
    constant uint&     tokens [[buffer(7)]],
    constant uint&     C      [[buffer(8)]],
    constant uint&     hidden [[buffer(9)]],    // 128 or 4096 (ViT)
    constant uint&     paths  [[buffer(10)]],   // 1 for C=32/ViT, C/32 for wide
    device half*       y      [[buffer(11)]],   // [tokens][C] ffn result (pre-skip)
    threadgroup half*  scratch [[threadgroup(0)]], // [hidden]
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tpg  [[threads_per_threadgroup]])
{
    const uint token = tgid;
    if (token >= tokens) return;
    device const half* xv = x + (size_t)token * C;

    // stage 1: hidden = pub_e4m3(silu(w1 x + b1)) per path
    // each thread handles hidden/tpg rows... simple strided loop
    for (uint row = tid; row < hidden * paths; row += tpg) {
        const uint pathId = row / hidden;
        const uint hrow = row % hidden;
        device const half* w1p = w1 + (size_t(pathId) * hidden + hrow) * C;
        float s = float(b1[pathId * hidden + hrow]);
        device const half* xp = xv + pathId * 32 * (paths > 1 ? 1 : 0);
        // NOTE: for wide blocks each path reads the FULL x [C]; pathId*32 offset
        // only applies to the w2 concat below.
        xp = xv;
        for (uint i = 0; i < C; ++i) s += float(w1p[i]) * float(xv[i]);
        scratch[row] = half(odl_pub_e4m3(silu(s)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 2: per path C_out = w2 * hidden_path (+b2), concat; then optional W3
    for (uint row = tid; row < C * paths; row += tpg) {
        const uint pathId = row / C;
        const uint crow = row % C;
        device const half* w2p = w2 + (size_t(pathId) * C + crow) * 128;
        float s = float(b2[pathId * C + crow]);
        for (uint i = 0; i < 128; ++i)
            s += float(w2p[i]) * float(scratch[pathId * hidden + i]);
        scratch[hidden * paths + row] = half(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half* out = y + (size_t)token * C;
    if (w3) {
        for (uint row = tid; row < C; row += tpg) {
            device const half* w3p = w3 + (size_t)row * C;
            float s = float(b3[row]);
            for (uint i = 0; i < C; ++i)
                s += float(w3p[i]) * float(scratch[hidden * paths + i]);
            out[row] = half(s);
        }
    } else {
        for (uint row = tid; row < C; row += tpg)
            out[row] = scratch[hidden * paths + row];
    }
}

// ---------------------------------------------------------------------------
// Window attention over one 8x8 window per threadgroup (64 threads).
// The phase offsets match the reference cycle: (0,0) (-4,-4) (-4,0) (0,-4).
// Out-of-field window tokens are ZERO vectors whose prior still enters the
// softmax denominator (reference semantics — not an approximation).
// ---------------------------------------------------------------------------
kernel void odl_window_attention(
    device const half* qkv    [[buffer(0)]],   // [tokens][3C] published E4M3
    device const half* prior  [[buffer(1)]],   // [heads][64][64]
    device const half* qscale [[buffer(2)]],   // [heads]
    constant uint2&    levelSize [[buffer(3)]],
    constant uint&     C      [[buffer(4)]],
    constant uint&     heads  [[buffer(5)]],
    constant uint&     phase  [[buffer(6)]],
    device half*       headO  [[buffer(7)]],    // [tokens][C] (E4M3 published)
    threadgroup half*  qE     [[threadgroup(0)]],  // heads*64*32
    threadgroup half*  kE     [[threadgroup(1)]],
    threadgroup half*  vE     [[threadgroup(2)]],
    threadgroup float* Oacc  [[threadgroup(3)]],  // heads*32
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    const uint win = 8, HD = 32;
    const int offX[4] = {0, -4, -4, 0};
    const int offY[4] = {0, -4, -4, 0};
    const int shift = (phase == 0) ? 0 : 4;
    const uint gridX = (levelSize.x + shift + win - 1) / win;
    const uint gridY = (levelSize.y + shift + win - 1) / win;
    const uint wgx = tgid % gridX, wgy = tgid / gridX;
    const int baseX = offX[phase] + int(wgx * win);
    const int baseY = offY[phase] + int(wgy * win);
    const uint ti = tid;                        // 0..63 within the window
    const int tx = baseX + int(ti % win);
    const int ty = baseY + int(ti / win);
    const bool inside = tx >= 0 && ty >= 0 && tx < (int)levelSize.x && ty < (int)levelSize.y;
    const uint token = inside ? uint(ty) * levelSize.x + uint(tx) : 0u;

    // 1) cosine-normalize + scale q,k per head; quantize v
    for (uint h = 0; h < heads; ++h) {
        const uint dst = (h * win * win + ti) * HD;
        if (inside) {
            device const half* q = qkv + (size_t)token * 3 * C + h * HD;
            device const half* k = qkv + (size_t)token * 3 * C + C + h * HD;
            device const half* v = qkv + (size_t)token * 3 * C + 2 * C + h * HD;
            float qn = 0, kn = 0;
            for (uint d = 0; d < HD; ++d) { qn += float(q[d])*float(q[d]); kn += float(k[d])*float(k[d]); }
            qn = sqrt(qn); kn = sqrt(kn);
            float invQ = qn > 0 ? 1.0f / qn : 0.0f;
            float invK = kn > 0 ? 1.0f / kn : 0.0f;
            for (uint d = 0; d < HD; ++d) {
                float qq = float(half(float(half(float(q[d]) * invQ)) * qscale[h]));
                float kk = float(half(float(k[d]) * invK));
                qE[dst + d] = half(odl_pub_e4m3(qq));
                kE[dst + d] = half(odl_pub_e4m3(kk));
                vE[dst + d] = half(odl_pub_e4m3(float(v[d])));
            }
        } else {
            for (uint d = 0; d < HD; ++d) { qE[dst+d] = 0; kE[dst+d] = 0; vE[dst+d] = 0; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) per head: S = qk^T + prior ; bit-trick softmax ; O = P V
    for (uint h = 0; h < heads; ++h) {
        const uint qo = (h * win * win) * HD;
        // this thread's row = ti (query)
        float denom = 0.0f;
        float prow[64];
        for (uint b = 0; b < win*win; ++b) {
            float s = prior ? float(prior[h * 64 * 64 + ti * 64 + b]) : 0.0f;
            for (uint d = 0; d < HD; ++d)
                s += float(qE[qo + ti*HD + d]) * float(kE[qo + b*HD + d]);
            float e = float(odl_window_exp(s));
            prow[b] = e;
            denom += e;
        }
        float invD = float(half(1.0f / denom));
        for (uint d = 0; d < HD; ++d) {
            float o = 0.0f;
            for (uint b = 0; b < win*win; ++b)
                o += odl_pub_e4m3(prow[b] * invD) * float(vE[qo + b*HD + d]);
            Oacc[h * HD + d] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (inside) {
            for (uint d = 0; d < HD; ++d)
                headO[(size_t)token * C + h * HD + d] = half(odl_pub_e4m3(Oacc[h * HD + d]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // zero out-of-field tokens' headO so the projection contributes prior-free
    if (!inside) {
        for (uint d = 0; d < C; ++d)
            headO[(size_t)token * C + d] = 0;
    }
}

// ---------------------------------------------------------------------------
// Global (ViT) attention: one threadgroup per head; threads = tokens (padded
// to a multiple of 64). Unnormalized weights with the exp(0) padding
// correction, per the reference.
// ---------------------------------------------------------------------------
kernel void odl_global_attention(
    device const half* qkv    [[buffer(0)]],
    device const half* qscale [[buffer(1)]],
    constant uint&     tokensReal [[buffer(2)]],
    constant uint&     C      [[buffer(3)]],
    constant uint&     heads  [[buffer(4)]],
    device half*       headO  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint HD = 32;
    const uint tokens = (tokensReal + 63u) / 64u * 64u;
    // one thread per real query token; loops heads and all keys
    const uint a = gid;
    if (a >= tokensReal) return;

    const float s32 = sqrt(32.0f);
    const float exp0 = float(odl_global_exp(0.0f));

    for (uint h = 0; h < heads; ++h) {
        device const half* qa = qkv + (size_t)a * 3 * C + h * HD;
        float qn = 0;
        for (uint d = 0; d < HD; ++d) qn += float(qa[d]) * float(qa[d]);
        qn = sqrt(qn);
        const float invQ = qn > 0.0f ? 1.0f / qn : 0.0f;

        float denom = 0.0f;
        float o[HD];
        for (uint d = 0; d < HD; ++d) o[d] = 0.0f;

        for (uint b = 0; b < tokens; ++b) {
            device const half* kb = qkv + (size_t)b * 3 * C + C + h * HD;
            device const half* vb = qkv + (size_t)b * 3 * C + 2 * C + h * HD;
            float kn = 0;
            for (uint d = 0; d < HD; ++d) kn += float(kb[d]) * float(kb[d]);
            kn = sqrt(kn);
            const float invK = kn > 0.0f ? 1.0f / kn : 0.0f;
            float s = 0.0f;
            for (uint d = 0; d < HD; ++d) {
                // three separate half multiplies: q*norm, *sqrt(32), *learned
                float q = float(half(float(half(float(qa[d]) * invQ)) * s32));
                q = float(half(q * float(qscale[h])));
                float k = float(half(float(kb[d]) * invK));
                s += odl_pub_e4m3(q) * odl_pub_e4m3(k);
            }
            float w = odl_pub_e4m3(float(odl_global_exp(s)));
            if (b >= tokensReal) w = exp0;                // padding rows behave like exp(0)
            denom += w;
            for (uint d = 0; d < HD; ++d) o[d] += w * odl_pub_e4m3(float(vb[d]));
        }
        denom -= exp0 * float(tokens - tokensReal);       // reference padding correction
        float invD = denom > 0 ? 1.0f / denom : 0.0f;
        for (uint d = 0; d < HD; ++d)
            headO[(size_t)a * C + h * HD + d] = half(odl_pub_e4m3(o[d] * invD));
    }
}

// ---------------------------------------------------------------------------
// block epilogue: out = pub_e4m3(y * attnScale + projOut)
// dispatch: one thread per (token, channel)
// ---------------------------------------------------------------------------
kernel void odl_block_epilogue(
    device const half* y        [[buffer(0)]],   // [tokens][C] post-skip
    device const half* projOut  [[buffer(1)]],   // [tokens][C]
    device const half* attnScale [[buffer(2)]],
    constant uint&     tokens   [[buffer(3)]],
    constant uint&     C        [[buffer(4)]],
    device half*       out      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= tokens * C) return;
    float v = float(y[gid]) * float(attnScale[gid % C]) + float(projOut[gid]);
    out[gid] = half(odl_pub_e4m3(v));
}

// ---------------------------------------------------------------------------
// block skip: y = pub(x * ffnScale + ffnOut); 32-ch window blocks publish raw
// f16, wider blocks and the ViT publish through E4M3 (reference asymmetry).
// ---------------------------------------------------------------------------
kernel void odl_block_skip(
    device const half* x         [[buffer(0)]],   // block input state [tokens][C]
    device const half* ffn       [[buffer(1)]],   // [tokens][C]
    device const half* ffnScale  [[buffer(2)]],
    constant uint&     tokens    [[buffer(3)]],
    constant uint&     C         [[buffer(4)]],
    constant uint&     rawF16    [[buffer(5)]],   // 1 = raw f16 add, 0 = E4M3
    device half*       y         [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= tokens * C) return;
    float v = float(x[gid]) * float(ffnScale[gid % C]) + float(ffn[gid]);
    y[gid] = half(rawF16 ? float(half(v)) : odl_pub_e4m3(v));
}

// ---------------------------------------------------------------------------
// transitions
// ---------------------------------------------------------------------------
kernel void odl_pool2x2(                       // encoder: field/level -> deeper
    device const half* src   [[buffer(0)]],    // [sw*sh][C]
    constant uint2&    srcSize [[buffer(1)]],
    constant uint2&    dstSize [[buffer(2)]],
    constant uint&     C       [[buffer(3)]],
    device half*       pooled  [[buffer(4)]],   // [dw*dh][C] E4M3
    uint gid [[thread_position_in_grid]])
{
    const uint dw = dstSize.x, dh = dstSize.y;
    if (gid >= dw * dh) return;
    const uint x = gid % dw, y = gid / dw;
    for (uint c = 0; c < C; ++c) {
        auto at = [&](uint sx, uint sy) {
            sx = min(sx, srcSize.x - 1); sy = min(sy, srcSize.y - 1);
            return float(src[((size_t)sy * srcSize.x + sx) * C + c]);
        };
        float a = at(x*2, y*2), b = at(x*2+1, y*2), cc = at(x*2, y*2+1), d = at(x*2+1, y*2+1);
        float top = float(half(a + b));                       // half steps, reference order
        float bot = float(half(cc + d));
        float sum = float(half(top + bot));
        pooled[(size_t)gid * C + c] = half(odl_pub_e4m3(float(half(sum * 0.25f))));
    }
}

kernel void odl_channel_gemm(                  // any [out][in] weight GEMM per token
    device const half* x      [[buffer(0)]],   // [tokens][in]
    device const half* w      [[buffer(1)]],   // [out][in]
    device const half* b      [[buffer(2)]],   // [out] or null
    constant uint&     tokens [[buffer(3)]],
    constant uint&     outCh  [[buffer(4)]],
    constant uint&     inCh   [[buffer(5)]],
    constant uint&     pubMode [[buffer(6)]],  // 0 = E4M3, 1 = f16
    device half*       out    [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= tokens * outCh) return;
    const uint t = gid / outCh, o = gid % outCh;
    device const half* wv = w + (size_t)o * inCh;
    device const half* xv = x + (size_t)t * inCh;
    float s = b ? float(b[o]) : 0.0f;
    for (uint i = 0; i < inCh; ++i) s += float(wv[i]) * float(xv[i]);
    out[gid] = half(pubMode == 0 ? odl_pub_e4m3(s) : float(half(s)));
}

kernel void odl_decoder_skip(                  // upsample + skip * scale, one FMA, E4M3
    device const half* up      [[buffer(0)]],  // [dstW*dstH][C] after upsample
    device const half* skip    [[buffer(1)]],  // [dstW*dstH][C] or empty
    device const half* scale   [[buffer(2)]],  // [C] or null
    constant uint&     tokens [[buffer(3)]],
    constant uint&     C      [[buffer(4)]],
    device half*       out    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= tokens * C) return;
    float v = float(up[gid]);
    if (skip) {
        float s = float(skip[gid]);
        if (scale) s = float(half(s * float(scale[gid % C])));
        v = float(half(v + s));
    }
    out[gid] = half(odl_pub_e4m3(v));
}

kernel void odl_nearest_upsample2x(
    device const half* src    [[buffer(0)]],
    constant uint2&    srcSize [[buffer(1)]],
    constant uint2&    dstSize [[buffer(2)]],
    constant uint&     C      [[buffer(3)]],
    device half*       dst    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dstSize.x * dstSize.y * C) return;
    const uint c = gid % C;
    const uint px = (gid / C) % dstSize.x;
    const uint py = (gid / C) / dstSize.x;
    dst[gid] = src[((size_t)(py / 2) * srcSize.x + (px / 2)) * C + c];
}

// ---------------------------------------------------------------------------
// head: 32 -> 4 f16 GEMM into f32
// ---------------------------------------------------------------------------
kernel void odl_head(
    device const half* state  [[buffer(0)]],   // [tokens][C]
    device const half* w      [[buffer(1)]],   // [4][C]
    device const half* b      [[buffer(2)]],   // [4]
    constant uint&     tokens [[buffer(3)]],
    constant uint&     C      [[buffer(4)]],
    device float*       head  [[buffer(5)]],   // [tokens][4] f32
    uint gid [[thread_position_in_grid]])
{
    if (gid >= tokens) return;
    device const half* x = state + (size_t)gid * C;
    for (uint o = 0; o < 4; ++o) {
        device const half* wv = w + (size_t)o * C;
        float s = b ? float(b[o]) : 0.0f;
        for (uint i = 0; i < C; ++i) s += float(wv[i]) * float(x[i]);
        head[(size_t)gid * 4 + o] = s;
    }
}
