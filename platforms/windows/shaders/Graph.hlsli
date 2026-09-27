// Graph.hlsli — the transformer block kernels: FFN, window attention
// (4-phase shifted windows), global ViT attention, block skip/epilogue and
// the U-net transitions. Byte-exact ports of the CPU reference semantics in
// core/backends/cpu/neural_graph_cpu.cpp (which is the parity golden — where
// the Metal port deviates or races, this file follows the CPU):
//
//   * fp16 storage, E4M3 publications, f32 accumulation
//   * window attention: 8x8 windows, phase cycle (0,0) (-4,-4) (-4,0) (0,-4),
//     cosine-normalized q/k, learned per-head q scale + 64x64 per-head prior,
//     softmax through the half bit-trick exponential, weights published E4M3,
//     the window output accumulated over all 64 query rows in fixed order and
//     scattered to every inside token (exactly as neural_graph_cpu.cpp does)
//   * ViT: global attention over all level tokens, unnormalized weights with
//     the exp(0) padding correction (padded qkv rows are zero, host-cleared)
//   * transitions: 2x2 box pool with per-step half rounding, nearest upsample,
//     decoder skip with the reference's in_scale/scale FMA chain
//
// Root signature "rsBuffers" (shared with Preprocess.hlsli):
//   t0..t7  ByteAddressBuffer inputs
//   u0..u3  RWByteAddressBuffer outputs
//   b0      root constants P[16]
//
// Shader model: SM 6.x compatible, also compiles under d3dcompiler cs_5_0.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "Common.hlsli"

ByteAddressBuffer   In0 : register(t0);
ByteAddressBuffer   In1 : register(t1);
ByteAddressBuffer   In2 : register(t2);
ByteAddressBuffer   In3 : register(t3);
ByteAddressBuffer   In4 : register(t4);
ByteAddressBuffer   In5 : register(t5);
ByteAddressBuffer   In6 : register(t6);
ByteAddressBuffer   In7 : register(t7);

RWByteAddressBuffer Out0 : register(u0);
RWByteAddressBuffer Out1 : register(u1);
RWByteAddressBuffer Out2 : register(u2);
RWByteAddressBuffer Out3 : register(u3);

cbuffer OdlParams : register(b0)
{
    uint4 P[16];
};

// ---------------------------------------------------------------------------
// odl_ffn — one threadgroup per token, 128 threads.
//   dense C=32 : hidden=128, paths=1   (w1 [128][32], w2 [32][128])
//   wide C     : hidden=128, paths=C/32(w1 [paths*128][C], w2 [paths*32][128],
//                concat -> pub_e4m3, then w3 [C][C])
//   ViT        : hidden=4096, paths=1  (w1 [4096][C], w2 [C][4096])
// P[0] = (tokens, C, hidden, paths)
// P[1] = (hasB1, hasB2, hasW3, -)
//   t0 x [tokens][C], t1 w1, t2 b1(opt), t3 w2, t4 b2(opt), t5 w3(opt), t6 b3(opt)
//   u0 y [tokens][C]  (pre-skip FFN result, f16 storage)
// ---------------------------------------------------------------------------
groupshared float gsFfn[6144];          // hidden*paths + C (concat) + C (out)

[numthreads(128, 1, 1)]
void odl_ffn(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    const uint tokens = P[0].x;
    const uint C      = P[0].y;
    const uint hidden = P[0].z;
    const uint paths  = P[0].w;
    const uint hasB1  = P[1].x;
    const uint hasB2  = P[1].y;
    const uint hasW3  = P[1].z;
    const uint token  = gid.x;
    if (token >= tokens) return;
    const uint hidOff = hidden * paths;   // concat region start
    const uint conOff = hidOff + C;       // post-w3 output region start

    // stage 1: per row h = pub_e4m3(silu(w1 x + b1)); every path reads the
    // full x [C] (reference semantics).
    for (uint row = tid.x; row < hidOff; row += 128u)
    {
        float s = hasB1 != 0u ? OdlLoadHalf(In2, row) : 0.0f;
        for (uint i = 0; i < C; ++i)
            s += OdlLoadHalf(In1, row * C + i) * OdlLoadHalf(In0, token * C + i);
        gsFfn[row] = OdlPubE4m3(OdlSilu(s));
    }
    GroupMemoryBarrierWithGroupSync();

    // stage 2: per path C_out = w2 * hidden_path (+b2). Dense blocks keep the
    // raw f32 result; wide blocks store the f16-rounded concat (published
    // through E4M3 at read time, as the reference does).
    const uint rows2   = (paths == 1u) ? C : (paths * 32u);
    const uint stride2 = (paths == 1u) ? hidden : 128u;
    const uint pathRow = (paths == 1u) ? C : 32u;
    for (uint row2 = tid.x; row2 < rows2; row2 += 128u)
    {
        float s2 = hasB2 != 0u ? OdlLoadHalf(In4, row2) : 0.0f;
        const uint pathId = row2 / pathRow;
        for (uint i = 0; i < stride2; ++i)
            s2 += OdlLoadHalf(In3, row2 * stride2 + i) * gsFfn[pathId * hidden + i];
        gsFfn[hidOff + row2] = (paths == 1u) ? s2 : OdlHalf(s2);
    }
    GroupMemoryBarrierWithGroupSync();

    if (hasW3 != 0u)
    {
        // stage 3: one C -> C contraction over the E4M3-published concat
        for (uint row3 = tid.x; row3 < C; row3 += 128u)
        {
            float s3 = OdlLoadHalf(In6, row3);
            for (uint i = 0; i < C; ++i)
                s3 += OdlLoadHalf(In5, row3 * C + i) * OdlPubE4m3(gsFfn[hidOff + i]);
            gsFfn[conOff + row3] = s3;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // publish: pairs of halfs, each dword written exactly once
    const uint base = token * C;
    for (uint pw = tid.x; pw < C / 2u; pw += 128u)
    {
        float a, b;
        if (hasW3 != 0u)
        {
            a = gsFfn[conOff + 2u * pw];
            b = gsFfn[conOff + 2u * pw + 1u];
        }
        else
        {
            a = gsFfn[hidOff + 2u * pw];
            b = gsFfn[hidOff + 2u * pw + 1u];
        }
        OdlStoreHalf2(Out0, base / 2u + pw, a, b);
    }
}

// ---------------------------------------------------------------------------
// odl_window_attention — one threadgroup per 8x8 window, 64 threads
// (one window token each). Out-of-field window slots are zero vectors whose
// prior still enters the softmax denominator (reference semantics).
// P[0] = (levelW, levelH, C, heads)
// P[1] = (phase, hasPrior, hasQscale, -)
//   t0 qkv [tokens][3C] (E4M3 published), t1 prior [heads][64][64] (opt),
//   t2 qscale [heads] (opt)
//   u0 headO [tokens][C] (E4M3 published)
// ---------------------------------------------------------------------------
groupshared float gsK[2048];            // 64 window tokens x 32 dims
groupshared float gsV[2048];
groupshared float gsO[2048];            // per-row attention outputs

[numthreads(64, 1, 1)]
void odl_window_attention(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    const uint W       = P[0].x;
    const uint H       = P[0].y;
    const uint C       = P[0].z;
    const uint heads   = P[0].w;
    const uint phase   = P[1].x;
    const uint hasPrior  = P[1].y;
    const uint hasQscale = P[1].z;

    const int  offX[4] = {0, -4, -4, 0};
    const int  offY[4] = {0, -4, -4, 0};
    const uint win = 8u, HD = 32u;
    const uint shift  = (phase == 0u) ? 0u : 4u;
    const uint gridX  = (W + shift + win - 1u) / win;
    const uint gridY  = (H + shift + win - 1u) / win;
    const uint wgx    = gid.x % gridX;
    const uint wgy    = gid.x / gridX;
    const int  baseX  = offX[phase] + int(wgx * win);
    const int  baseY  = offY[phase] + int(wgy * win);
    const uint ti     = tid.x;                       // 0..63 within the window
    const int  tx     = baseX + int(ti % win);
    const int  ty     = baseY + int(ti / win);
    const bool inside = tx >= 0 && ty >= 0 && tx < int(W) && ty < int(H);
    const uint token  = inside ? uint(ty) * W + uint(tx) : 0u;

    for (uint h = 0; h < heads; ++h)
    {
        // 1) publish k/v for this head (cosine-normalized, E4M3 grid)
        if (inside)
        {
            float kReg[32], vReg[32];
            float kn = 0.0f;
            [unroll]
            for (uint d = 0; d < 32u; ++d)
            {
                kReg[d] = OdlLoadHalf(In0, token * 3u * C + C + h * HD + d);
                vReg[d] = OdlLoadHalf(In0, token * 3u * C + 2u * C + h * HD + d);
                kn += kReg[d] * kReg[d];
            }
            float ks   = sqrt(kn);
            float invK = ks > 0.0f ? 1.0f / ks : 0.0f;
            [unroll]
            for (uint d2 = 0; d2 < 32u; ++d2)
            {
                gsK[ti * HD + d2] = OdlPubE4m3(OdlHalf(kReg[d2] * invK));
                gsV[ti * HD + d2] = OdlPubE4m3(vReg[d2]);
            }
        }
        else
        {
            [unroll]
            for (uint dz = 0; dz < 32u; ++dz)
            {
                gsK[ti * HD + dz] = 0.0f;
                gsV[ti * HD + dz] = 0.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 2) this thread's query row: normalize + learned scale, published
        float qPub[32];
        [unroll]
        for (uint qz = 0; qz < 32u; ++qz) qPub[qz] = 0.0f;
        if (inside)
        {
            float qReg[32];
            float qn = 0.0f;
            [unroll]
            for (uint d3 = 0; d3 < 32u; ++d3)
            {
                qReg[d3] = OdlLoadHalf(In0, token * 3u * C + h * HD + d3);
                qn += qReg[d3] * qReg[d3];
            }
            float qs   = sqrt(qn);
            float invQ = qs > 0.0f ? 1.0f / qs : 0.0f;
            float qsc  = hasQscale != 0u ? OdlLoadHalf(In2, h) : 1.0f;
            [unroll]
            for (uint d4 = 0; d4 < 32u; ++d4)
                qPub[d4] = OdlPubE4m3(OdlHalf(qReg[d4] * invQ) * qsc);
        }

        // 3) S = qk^T + prior ; softmax through the bit-trick exponential
        float denom = 0.0f;
        float prow[64];
        for (uint b = 0; b < win * win; ++b)
        {
            float s = hasPrior != 0u ? OdlLoadHalf(In1, h * 64u * 64u + ti * 64u + b) : 0.0f;
            [unroll]
            for (uint d5 = 0; d5 < 32u; ++d5)
                s += qPub[d5] * gsK[b * HD + d5];
            float e = OdlWindowExp(s);
            prow[b] = e;
            denom += e;
        }
        const float invD = OdlHalf(1.0f / denom);
        [unroll]
        for (uint d6 = 0; d6 < 32u; ++d6)
        {
            float o = 0.0f;
            for (uint b2 = 0; b2 < win * win; ++b2)
                o += OdlPubE4m3(prow[b2] * invD) * gsV[b2 * HD + d6];
            gsO[ti * HD + d6] = o;
        }
        GroupMemoryBarrierWithGroupSync();

        // 4) the window output: sum over all 64 query rows in fixed order
        //    (exactly the reference accumulation), then scatter to every
        //    inside token of the window.
        if (tid.x < 32u)
        {
            float acc = 0.0f;
            [loop] for (uint a = 0; a < 64u; ++a)
                acc += gsO[a * HD + tid.x];
            gsK[tid.x] = acc;                        // reuse gsK (consumed above)
        }
        GroupMemoryBarrierWithGroupSync();

        if (inside)
        {
            [unroll]
            for (uint dp = 0; dp < 16u; ++dp)
                OdlStoreHalf2(Out0, (token * C + h * HD + 2u * dp) / 2u,
                              gsK[2u * dp], gsK[2u * dp + 1u]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// odl_global_attention — global ViT attention, one thread per real token.
// Padded key rows (tokensReal..tokens) read zeros (the host clears the qkv
// buffer first) and behave exactly like the reference's exp(0) padding rows.
// P[0] = (tokensReal, C, heads, -)
// P[1] = (hasQscale, -, -, -)
//   t0 qkv [tokensPadded][3C], t1 qscale [heads] (opt)
//   u0 headO [tokensReal][C] (E4M3 published)
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_global_attention(uint3 dtid : SV_DispatchThreadID)
{
    const uint tokensReal = P[0].x;
    const uint C          = P[0].y;
    const uint heads      = P[0].z;
    const uint hasQscale  = P[1].x;
    const uint a = dtid.x;
    if (a >= tokensReal) return;

    const uint HD    = 32u;
    const uint tokens = (tokensReal + 63u) / 64u * 64u;
    const float s32  = sqrt(32.0f);
    const float exp0 = OdlGlobalExp(0.0f);

    for (uint h = 0; h < heads; ++h)
    {
        float qReg[32];
        float qn = 0.0f;
        [unroll]
        for (uint d = 0; d < 32u; ++d)
        {
            qReg[d] = OdlLoadHalf(In0, a * 3u * C + h * HD + d);
            qn += qReg[d] * qReg[d];
        }
        float qs   = sqrt(qn);
        float invQ = qs > 0.0f ? 1.0f / qs : 0.0f;
        float qsc  = hasQscale != 0u ? OdlLoadHalf(In1, h) : 1.0f;
        float qP[32];
        [unroll]
        for (uint d2 = 0; d2 < 32u; ++d2)
        {
            // three separate half multiplies: q*norm, *sqrt(32), *learned
            float q = OdlHalf(OdlHalf(qReg[d2] * invQ) * s32);
            q = OdlHalf(q * qsc);
            qP[d2] = OdlPubE4m3(q);
        }

        float denom = 0.0f;
        float o[32];
        [unroll]
        for (uint oz = 0; oz < 32u; ++oz) o[oz] = 0.0f;

        for (uint b = 0; b < tokens; ++b)
        {
            float kReg[32], vReg[32];
            float kn = 0.0f;
            [loop]
            for (uint d3 = 0; d3 < 32u; ++d3)
            {
                kReg[d3] = OdlLoadHalf(In0, b * 3u * C + C + h * HD + d3);
                vReg[d3] = OdlLoadHalf(In0, b * 3u * C + 2u * C + h * HD + d3);
                kn += kReg[d3] * kReg[d3];
            }
            float ks   = sqrt(kn);
            float invK = ks > 0.0f ? 1.0f / ks : 0.0f;
            float s = 0.0f;
            [loop]
            for (uint d4 = 0; d4 < 32u; ++d4)
            {
                float k = OdlHalf(kReg[d4] * invK);
                s += qP[d4] * OdlPubE4m3(k);
            }
            float w = OdlPubE4m3(OdlGlobalExp(s));
            if (b >= tokensReal) w = exp0;           // padding rows behave like exp(0)
            denom += w;
            [loop]
            for (uint d5 = 0; d5 < 32u; ++d5)
                o[d5] += w * OdlPubE4m3(vReg[d5]);
        }
        denom -= exp0 * float(tokens - tokensReal);  // reference padding correction
        const float invD = denom > 0.0f ? 1.0f / denom : 0.0f;
        [unroll]
        for (uint dp = 0; dp < 16u; ++dp)
            OdlStoreHalf2(Out0, (a * C + h * HD + 2u * dp) / 2u,
                          OdlPubE4m3(o[2u * dp] * invD),
                          OdlPubE4m3(o[2u * dp + 1u] * invD));
    }
}

// ---------------------------------------------------------------------------
// odl_block_skip — y = pub(x * ffnScale + ffn); 32-ch window blocks publish
// raw f16, wider blocks and the ViT publish through E4M3 (reference asymmetry).
// P[0] = (tokens, C, rawF16, -)   P[1] = (hasScale, -, -, -)
//   t0 x [tokens][C], t1 ffn [tokens][C], t2 ffnScale [C] (opt), u0 y
// Dispatch: tokens * C/2 threads (one dword = one channel pair each).
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_block_skip(uint3 dtid : SV_DispatchThreadID)
{
    const uint tokens   = P[0].x;
    const uint C        = P[0].y;
    const uint rawF16   = P[0].z;
    const uint hasScale = P[1].x;
    const uint pair = dtid.x;
    if (pair >= tokens * (C / 2u)) return;
    const uint token = pair / (C / 2u);
    const uint i0    = token * C + 2u * (pair % (C / 2u));

    float2 x  = OdlLoadHalf2(In0, i0);
    float2 f  = OdlLoadHalf2(In1, i0);
    float  s0 = hasScale != 0u ? OdlLoadHalf(In2, (i0 + 0u) % C)     : 1.0f;
    float  s1 = hasScale != 0u ? OdlLoadHalf(In2, (i0 + 1u) % C)     : 1.0f;
    float  v0 = x.x * s0 + f.x;
    float  v1 = x.y * s1 + f.y;
    if (rawF16 != 0u)
        OdlStoreHalf2(Out0, pair, OdlHalf(v0), OdlHalf(v1));
    else
        OdlStoreHalf2(Out0, pair, OdlPubE4m3(v0), OdlPubE4m3(v1));
}

// ---------------------------------------------------------------------------
// odl_block_epilogue — out = pub_e4m3(y * attnScale + projOut)
// P[0] = (tokens, C, hasScale, -)
//   t0 y [tokens][C], t1 projOut [tokens][C], t2 attnScale [C] (opt), u0 out
// Dispatch: tokens * C/2 threads.
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_block_epilogue(uint3 dtid : SV_DispatchThreadID)
{
    const uint tokens   = P[0].x;
    const uint C        = P[0].y;
    const uint hasScale = P[0].z;
    const uint pair = dtid.x;
    if (pair >= tokens * (C / 2u)) return;
    const uint token = pair / (C / 2u);
    const uint i0    = token * C + 2u * (pair % (C / 2u));

    float2 y  = OdlLoadHalf2(In0, i0);
    float2 pj = OdlLoadHalf2(In1, i0);
    float  a0 = hasScale != 0u ? OdlLoadHalf(In2, (i0 + 0u) % C) : 1.0f;
    float  a1 = hasScale != 0u ? OdlLoadHalf(In2, (i0 + 1u) % C) : 1.0f;
    OdlStoreHalf2(Out0, pair,
                  OdlPubE4m3(y.x * a0 + pj.x),
                  OdlPubE4m3(y.y * a1 + pj.y));
}

// ---------------------------------------------------------------------------
// odl_pool2x2 — encoder 2x2 box pool: ((a+b)+(c+d)) * 0.25, every step a half
// (reference order), published E4M3.
// P[0] = (srcW, srcH, dstW, dstH)   P[1] = (C, -, -, -)
//   t0 src [sw*sh][C], u0 pooled [dw*dh][C]
// Dispatch: dw*dh threads (each owns its whole row -> safe pair writes).
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_pool2x2(uint3 dtid : SV_DispatchThreadID)
{
    const uint srcW = P[0].x, srcH = P[0].y;
    const uint dstW = P[0].z, dstH = P[0].w;
    const uint C    = P[1].x;
    const uint tok  = dtid.x;
    if (tok >= dstW * dstH) return;
    const uint x = tok % dstW, y = tok / dstW;

    for (uint c2 = 0; c2 < C / 2u; ++c2)
    {
        float out2[2];
        [unroll]
        for (uint hc = 0; hc < 2u; ++hc)
        {
            uint c = 2u * c2 + hc;
            float s00 = OdlLoadHalf(In0, ((min(y * 2u,     srcH - 1u)) * srcW + min(x * 2u,     srcW - 1u)) * C + c);
            float s10 = OdlLoadHalf(In0, ((min(y * 2u,     srcH - 1u)) * srcW + min(x * 2u + 1u, srcW - 1u)) * C + c);
            float s01 = OdlLoadHalf(In0, ((min(y * 2u + 1u, srcH - 1u)) * srcW + min(x * 2u,     srcW - 1u)) * C + c);
            float s11 = OdlLoadHalf(In0, ((min(y * 2u + 1u, srcH - 1u)) * srcW + min(x * 2u + 1u, srcW - 1u)) * C + c);
            float top = OdlHalf(s00 + s10);
            float bot = OdlHalf(s01 + s11);
            float sum = OdlHalf(top + bot);
            out2[hc] = OdlPubE4m3(OdlHalf(sum * 0.25f));
        }
        OdlStoreHalf2(Out0, tok * (C / 2u) + c2, out2[0], out2[1]);
    }
}

// ---------------------------------------------------------------------------
// odl_channel_gemm — any [out][in] weight GEMM per token.
// P[0] = (tokens, outCh, inCh, pubMode)  P[1] = (hasBias, mode, -, -)
//   mode 0: GEMM      (t0 x [tokens][inCh], t1 w [outCh][inCh], t2 b [outCh] opt)
//   mode 2: publish   (element-wise E4M3 publication of the DML GEMM output;
//                      reads and writes u0 in place)
//   pubMode: 0 = publish E4M3, 1 = raw f16
//   u0 out [tokens][outCh]
// Dispatch: tokens * outCh/2 threads (outCh is always even).
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_channel_gemm(uint3 dtid : SV_DispatchThreadID)
{
    const uint tokens  = P[0].x;
    const uint outCh   = P[0].y;
    const uint inCh    = P[0].z;
    const uint pubMode = P[0].w;
    const uint hasBias = P[1].x;
    const uint mode    = P[1].y;
    const uint pairs   = tokens * (outCh / 2u);
    const uint pair    = dtid.x;
    if (pair >= pairs) return;

    if (mode == 2u)
    {
        // DML post-pass: quantize the raw f16 GEMM output onto the E4M3 grid
        float2 v = OdlLoadHalf2(Out0, 2u * pair);
        OdlStoreHalf2(Out0, pair, OdlPubE4m3(v.x), OdlPubE4m3(v.y));
        return;
    }

    const uint token = pair / (outCh / 2u);
    const uint o0    = 2u * (pair % (outCh / 2u));
    float s0 = hasBias != 0u ? OdlLoadHalf(In2, o0)     : 0.0f;
    float s1 = hasBias != 0u ? OdlLoadHalf(In2, o0 + 1u) : 0.0f;
    const uint xBase = token * inCh;
    for (uint i = 0; i < inCh; ++i)
    {
        float xv = OdlLoadHalf(In0, xBase + i);
        s0 += OdlLoadHalf(In1, o0 * inCh + i)       * xv;
        s1 += OdlLoadHalf(In1, (o0 + 1u) * inCh + i) * xv;
    }
    if (pubMode == 0u)
        OdlStoreHalf2(Out0, pair, OdlPubE4m3(s0), OdlPubE4m3(s1));
    else
        OdlStoreHalf2(Out0, pair, OdlHalf(s0), OdlHalf(s1));
}

// ---------------------------------------------------------------------------
// odl_decoder_skip — upsample + skip * scale, the reference's exact FMA chain:
//   v = f16(up * inScale) ; s = f16(skip * scale) ; v = f16(v + s) ; pub_e4m3
// P[0] = (tokens, C, -, -)
// P[1] = (hasSkip, hasScale, hasInScale, -)
//   t0 up [tokens][C], t1 skip (opt), t2 scale [C] (opt), t3 inScale [C] (opt)
//   u0 out [tokens][C]
// Dispatch: tokens * C/2 threads.
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_decoder_skip(uint3 dtid : SV_DispatchThreadID)
{
    const uint tokens     = P[0].x;
    const uint C          = P[0].y;
    const uint hasSkip    = P[1].x;
    const uint hasScale   = P[1].y;
    const uint hasInScale = P[1].z;
    const uint pair = dtid.x;
    if (pair >= tokens * (C / 2u)) return;
    const uint token = pair / (C / 2u);
    const uint i0    = token * C + 2u * (pair % (C / 2u));

    float2 v = OdlLoadHalf2(In0, i0);
    if (hasInScale != 0u)
    {
        v.x = OdlHalf(v.x * OdlLoadHalf(In3, (i0 + 0u) % C));
        v.y = OdlHalf(v.y * OdlLoadHalf(In3, (i0 + 1u) % C));
    }
    if (hasSkip != 0u)
    {
        float2 s = OdlLoadHalf2(In1, i0);
        if (hasScale != 0u)
        {
            s.x = OdlHalf(s.x * OdlLoadHalf(In2, (i0 + 0u) % C));
            s.y = OdlHalf(s.y * OdlLoadHalf(In2, (i0 + 1u) % C));
        }
        v.x = OdlHalf(v.x + s.x);
        v.y = OdlHalf(v.y + s.y);
    }
    OdlStoreHalf2(Out0, pair, OdlPubE4m3(v.x), OdlPubE4m3(v.y));
}

// ---------------------------------------------------------------------------
// odl_nearest_upsample2x — pure 2x nearest copy, whole dwords moved.
// P[0] = (srcW, srcH, dstW, dstH)   P[1] = (C, -, -, -)
//   t0 src [sw*sh][C], u0 dst [dw*dh][C]
// Dispatch: dstW*dstH*C/2 threads.
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_nearest_upsample2x(uint3 dtid : SV_DispatchThreadID)
{
    const uint srcW = P[0].x, srcH = P[0].y;
    const uint dstW = P[0].z, dstH = P[0].w;
    const uint C    = P[1].x;
    const uint pair = dtid.x;
    if (pair >= dstW * dstH * (C / 2u)) return;
    const uint c2  = pair % (C / 2u);
    const uint px  = (pair / (C / 2u)) % dstW;
    const uint py  = pair / (C / 2u) / dstW;
    const uint srcToken = (py / 2u) * srcW + (px / 2u);
    uint dword = In0.Load(((srcToken * C) / 2u + c2) << 2);
    Out0.Store(pair << 2, dword);
}

// ---------------------------------------------------------------------------
// odl_head — final C -> 4 f16 GEMM into f32.
// P[0] = (tokens, C, hasB, -)
//   t0 state [tokens][C], t1 w [4][C], t2 b [4] (opt), u0 head [tokens][4] f32
// Dispatch: tokens threads.
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void odl_head(uint3 dtid : SV_DispatchThreadID)
{
    const uint tokens = P[0].x;
    const uint C      = P[0].y;
    const uint hasB   = P[0].z;
    const uint t = dtid.x;
    if (t >= tokens) return;
    for (uint o = 0; o < 4u; ++o)
    {
        float s = hasB != 0u ? OdlLoadHalf(In2, o) : 0.0f;
        for (uint i = 0; i < C; ++i)
            s += OdlLoadHalf(In1, o * C + i) * OdlLoadHalf(In0, t * C + i);
        Out0.Store((t * 4u + o) << 2, asuint(s));
    }
}
