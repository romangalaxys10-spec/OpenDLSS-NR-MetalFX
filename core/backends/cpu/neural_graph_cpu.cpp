// The neural rendering graph on CPU.
//
// Port of the reference block semantics (original repo docs/network.md; our
// docs/NETWORK.md):
//   block = x*ffnScale + FFN(x) -> y ; out = y*attnScale + Proj(Attn(QKV(y)))
//   fp16 storage, E4M3 publications, f32 accumulation (the reference
//   accumulates in f16 on NVIDIA tensor cores; we accumulate in f32 for
//   hardware portability — the publication grid and half bit-tricks match)
//   window attention: 8x8 windows, phase cycle (0,0) (-4,-4) (-4,0) (0,-4),
//   cosine-normalized q/k, learned per-head q scale + 64x64 per-head prior,
//   softmax through the half bit-trick exponential, weights published E4M3
//   ViT: global attention, unnormalized weights, exp(0) padding correction
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/backend.h"
#include "opendlss/fp16.h"
#include "opendlss/e4m3.h"
#include "opendlss/logging.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

namespace opendlss {

namespace {

using F16Vec = std::vector<uint16_t>;

inline uint16_t pub_e4m3(float v) { return f32_to_f16(e4m3_publish(v)); }
inline uint16_t pub_f16(float v)  { return f32_to_f16(v); }
inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

// out[o] = sum_i W[o*in+i]*x[i] + b[o]; f16 weights/bias, f32 accumulate.
void gemm_f16_acc(const uint16_t* W, const uint16_t* b, const uint16_t* x,
                  uint32_t out, uint32_t in, std::vector<float>& acc) {
    acc.assign(out, 0.f);
    for (uint32_t o = 0; o < out; ++o) {
        float s = b ? f16_to_f32(b[o]) : 0.f;
        const uint16_t* w = W + size_t(o) * in;
        for (uint32_t i = 0; i < in; ++i) s += f16_to_f32(w[i]) * f16_to_f32(x[i]);
        acc[o] = s;
    }
}

const TensorRecord* T(const Model& m, uint32_t b, uint32_t l, const char* p) {
    return m.tensor(b, l, p);
}

// ---------------------------------------------------------------------------
// FFN. Returns f32 out (caller publishes).
// ---------------------------------------------------------------------------
void ffn(const Model& m, uint32_t blockIdx, uint32_t C, bool isViT,
         const uint16_t* x, std::vector<float>& outF) {
    if (C == 32 || isViT) {
        uint32_t hidden = isViT ? 4096 : 128;
        const TensorRecord* w1 = T(m, blockIdx, 0, "w1");
        const TensorRecord* b1 = T(m, blockIdx, 0, "b1");
        const TensorRecord* w2 = T(m, blockIdx, 0, "w2");
        const TensorRecord* b2 = T(m, blockIdx, 0, "b2");
        if (!w1 || !w2) { log_error("graph: block %u missing ffn weights", blockIdx); outF.assign(C, 0.f); return; }
        std::vector<float> h;
        gemm_f16_acc(w1->f16.data(), b1 ? b1->f16.data() : nullptr, x, hidden, C, h);
        F16Vec hP(hidden);
        for (uint32_t i = 0; i < hidden; ++i) hP[i] = pub_e4m3(silu(h[i]));   // E4M3 boundary
        gemm_f16_acc(w2->f16.data(), b2 ? b2->f16.data() : nullptr, hP.data(), C, hidden, outF);
        return;
    }
    if (C == 512) {
        // 512 -> 8 branches of 64: each 64->256->64 with SiLU after the 256 middle only,
        // concat to 512, then one 512->512 contraction (w3/b3).
        const TensorRecord* w1 = T(m, blockIdx, 0, "w1");   // [512][512]: split 8x[256][64] pre, stored as [512][512]
        const TensorRecord* b1 = T(m, blockIdx, 0, "b1");
        const TensorRecord* w2 = T(m, blockIdx, 0, "w2");   // 8x[64][256]
        const TensorRecord* b2 = T(m, blockIdx, 0, "b2");
        const TensorRecord* w3 = T(m, blockIdx, 0, "w3");
        const TensorRecord* b3 = T(m, blockIdx, 0, "b3");
        if (!w1 || !w2 || !w3) { log_error("graph: block %u missing 512-ffn weights", blockIdx); outF.assign(C, 0.f); return; }
        F16Vec concat(C);
        for (uint32_t br = 0; br < 8; ++br) {
            const uint16_t* xBr = x + size_t(br) * 64;
            // pre: rows [br*32 .. br*32+31] of w1 act on inputs [br*64 .. br*64+63]
            std::vector<float> mid(256);
            for (uint32_t o = 0; o < 256; ++o) {
                const uint16_t* w = w1->f16.data() + (size_t(br) * 256 + o) * 64;
                float s = b1 ? f16_to_f32(b1->f16[br * 256 + o]) : 0.f;
                for (uint32_t i = 0; i < 64; ++i) s += f16_to_f32(w[i]) * f16_to_f32(xBr[i]);
                mid[o] = s;                                   // SiLU only after the middle
            }
            F16Vec midP(256);
            for (uint32_t i = 0; i < 256; ++i) midP[i] = pub_e4m3(mid[i]);
            std::vector<float> branchOut;
            gemm_f16_acc(w2->f16.data() + size_t(br) * 64 * 256,
                         b2 ? b2->f16.data() + br * 64 : nullptr,
                         midP.data(), 64, 256, branchOut);
            for (uint32_t i = 0; i < 64; ++i)
                concat[br * 64 + i] = f32_to_f16(branchOut[i]);
        }
        F16Vec concatP(C);
        for (uint32_t i = 0; i < C; ++i) concatP[i] = pub_e4m3(f16_to_f32(concat[i]));
        gemm_f16_acc(w3->f16.data(), b3 ? b3->f16.data() : nullptr, concatP.data(), C, C, outF);
        return;
    }
    // wide blocks 64/128/256: C/32 paths, each C -> 128 -> 32 (SiLU after first),
    // concatenated to C, then one C -> C contraction (w3/b3).
    const uint32_t paths = C / 32;
    const TensorRecord* w1 = T(m, blockIdx, 0, "w1");   // [paths*128][C]
    const TensorRecord* b1 = T(m, blockIdx, 0, "b1");
    const TensorRecord* w2 = T(m, blockIdx, 0, "w2");   // [paths*32][128]
    const TensorRecord* b2 = T(m, blockIdx, 0, "b2");
    const TensorRecord* w3 = T(m, blockIdx, 0, "w3");
    const TensorRecord* b3 = T(m, blockIdx, 0, "b3");
    if (!w1 || !w2 || !w3) { log_error("graph: block %u missing wide-ffn weights", blockIdx); outF.assign(C, 0.f); return; }
    F16Vec concat(C);
    for (uint32_t p = 0; p < paths; ++p) {
        std::vector<float> h;
        gemm_f16_acc(w1->f16.data() + size_t(p) * 128 * C, b1 ? b1->f16.data() + p * 128 : nullptr,
                     x, 128, C, h);
        F16Vec hP(128);
        for (uint32_t i = 0; i < 128; ++i) hP[i] = pub_e4m3(silu(h[i]));
        std::vector<float> o32;
        gemm_f16_acc(w2->f16.data() + size_t(p) * 32 * 128, b2 ? b2->f16.data() + p * 32 : nullptr,
                     hP.data(), 32, 128, o32);
        for (uint32_t i = 0; i < 32; ++i) concat[p * 32 + i] = f32_to_f16(o32[i]);
    }
    F16Vec concatP(C);
    for (uint32_t i = 0; i < C; ++i) concatP[i] = pub_e4m3(f16_to_f32(concat[i]));
    gemm_f16_acc(w3->f16.data(), b3 ? b3->f16.data() : nullptr, concatP.data(), C, C, outF);
}

// ---------------------------------------------------------------------------
// Window attention. y: [tokens][C] f16; writes projection input O (E4M3 f16 bits)
// into oTokens [tokens][C].
// ---------------------------------------------------------------------------
void window_attention(const Model& m, uint32_t blockIdx, uint32_t C, uint32_t heads,
                      uint32_t phase, uint32_t lw, uint32_t lh,
                      const F16Vec& y, F16Vec& oTokens) {
    const uint32_t HD = 32;
    const uint32_t W = lw, H = lh;
    const uint32_t tokens = W * H;
    const uint32_t win = 8;
    static const int offX[4] = {0, -4, -4, 0};
    static const int offY[4] = {0, -4, -4, 0};

    const TensorRecord* qkvT  = T(m, blockIdx, 1, "qkv");    // [3C][C]
    const TensorRecord* projT = T(m, blockIdx, 1, "proj");   // [C][C]
    const TensorRecord* priorT= T(m, blockIdx, 1, "prior");  // [heads][64][64]
    const TensorRecord* qsT   = T(m, blockIdx, 1, "qscale"); // [heads]
    if (!qkvT || !projT) {
        log_error("graph: block %u missing attention weights", blockIdx);
        oTokens.assign(size_t(tokens) * C, 0);
        return;
    }

    oTokens.assign(size_t(tokens) * C, 0);

    // QKV over all tokens; input published E4M3 (the reference's QKV input grid)
    F16Vec qkvP(size_t(tokens) * 3 * C);
    {
        std::vector<float> acc;
        F16Vec yP(size_t(tokens) * C);
        for (uint32_t t = 0; t < tokens; ++t)
            for (uint32_t i = 0; i < C; ++i)
                yP[size_t(t) * C + i] = pub_e4m3(f16_to_f32(y[size_t(t) * C + i]));
        for (uint32_t t = 0; t < tokens; ++t) {
            gemm_f16_acc(qkvT->f16.data(), nullptr, &yP[size_t(t) * C], 3 * C, C, acc);
            for (uint32_t i = 0; i < 3 * C; ++i) qkvP[size_t(t) * 3 * C + i] = f32_to_f16(acc[i]);
        }
    }

    std::vector<float> qscale(heads, 1.0f);
    if (qsT && qsT->f16.size() >= heads)
        for (uint32_t hI = 0; hI < heads; ++hI) qscale[hI] = f16_to_f32(qsT->f16[hI]);

    const int shiftX = -offX[phase], shiftY = -offY[phase];
    const uint32_t gridX = (W + uint32_t(shiftX) + win - 1) / win;
    const uint32_t gridY = (H + uint32_t(shiftY) + win - 1) / win;

    // per-token head outputs (E4M3 published), gathered from windows
    F16Vec headO(size_t(tokens) * C, 0);

    F16Vec qE(size_t(heads) * win * win * HD), kE(size_t(heads) * win * win * HD),
           vE(size_t(heads) * win * win * HD);
    std::vector<uint8_t> inside(win * win);
    std::vector<uint32_t> tokIdx(win * win);

    for (uint32_t gy = 0; gy < gridY; ++gy) {
        for (uint32_t gx = 0; gx < gridX; ++gx) {
            const int baseX = offX[phase] + int(gx * win);
            const int baseY = offY[phase] + int(gy * win);
            for (uint32_t ti = 0; ti < win * win; ++ti) {
                int tx = baseX + int(ti % win);
                int ty = baseY + int(ti / win);
                inside[ti] = (tx >= 0 && ty >= 0 && tx < int(W) && ty < int(H)) ? 1 : 0;
                tokIdx[ti] = inside[ti] ? uint32_t(ty) * W + uint32_t(tx) : 0;
                if (!inside[ti]) {
                    for (uint32_t hI = 0; hI < heads; ++hI) {
                        const uint32_t qo = (hI * win * win + ti) * HD;
                        std::memset(&qE[qo], 0, HD * 2);
                        std::memset(&kE[qo], 0, HD * 2);
                        std::memset(&vE[qo], 0, HD * 2);
                    }
                    continue;
                }
                const uint32_t t = tokIdx[ti];
                for (uint32_t hI = 0; hI < heads; ++hI) {
                    const uint32_t qo = (hI * win * win + ti) * HD;
                    const uint16_t* qSrc = &qkvP[size_t(t) * 3 * C + hI * HD];
                    const uint16_t* kSrc = &qkvP[size_t(t) * 3 * C + C + hI * HD];
                    const uint16_t* vSrc = &qkvP[size_t(t) * 3 * C + 2 * C + hI * HD];
                    float qn = 0, kn = 0;
                    for (int d = 0; d < int(HD); ++d) {
                        float q = f16_to_f32(qSrc[d]), k = f16_to_f32(kSrc[d]);
                        qn += q * q; kn += k * k;
                    }
                    qn = std::sqrt(qn); kn = std::sqrt(kn);
                    const float invQ = qn > 0 ? 1.0f / qn : 0.0f;
                    const float invK = kn > 0 ? 1.0f / kn : 0.0f;
                    for (int d = 0; d < int(HD); ++d) {
                        float q = f16_to_f32(f32_to_f16(f16_to_f32(qSrc[d]) * invQ)) * qscale[hI];
                        float k = f16_to_f32(f32_to_f16(f16_to_f32(kSrc[d]) * invK));
                        qE[qo + d] = pub_e4m3(q);
                        kE[qo + d] = pub_e4m3(k);
                        vE[qo + d] = pub_e4m3(f16_to_f32(vSrc[d]));
                    }
                }
            }
            // per head: S = q k^T + prior, exp trick, softmax, O = P V
            std::vector<float> Oaccum(HD, 0.f);
            for (uint32_t hI = 0; hI < heads; ++hI) {
                const uint32_t qo = hI * win * win * HD;
                const uint16_t* prior = (priorT && priorT->f16.size() >= size_t(heads) * 64 * 64)
                                      ? priorT->f16.data() + size_t(hI) * 64 * 64 : nullptr;
                std::fill(Oaccum.begin(), Oaccum.end(), 0.f);
                for (uint32_t a = 0; a < win * win; ++a) {       // query row
                    // S row
                    float sRow[64];
                    float denom = 0.f;
                    for (uint32_t bI = 0; bI < win * win; ++bI) {
                        float s = prior ? f16_to_f32(prior[a * 64 + bI]) : 0.f;
                        for (int d = 0; d < int(HD); ++d)
                            s += f16_to_f32(qE[qo + size_t(a) * HD + d]) * f16_to_f32(kE[qo + size_t(bI) * HD + d]);
                        sRow[bI] = s;
                    }
                    // softmax through the bit-trick exponential (fixed order)
                    float pRow[64];
                    for (uint32_t bI = 0; bI < win * win; ++bI) {
                        float e = window_exp_trick(sRow[bI]);
                        pRow[bI] = e;
                        denom += e;
                    }
                    const float invD = f16_to_f32(f32_to_f16(1.0f / denom));
                    for (uint32_t bI = 0; bI < win * win; ++bI) {
                        float wgt = e4m3_publish(pRow[bI] * invD);   // weights published E4M3
                        for (int d = 0; d < int(HD); ++d)
                            Oaccum[d] += wgt * f16_to_f32(vE[qo + size_t(bI) * HD + d]);
                    }
                }
                // scatter O for every inside token of this window
                for (uint32_t ti = 0; ti < win * win; ++ti) {
                    if (!inside[ti]) continue;
                    for (int d = 0; d < int(HD); ++d)
                        headO[size_t(tokIdx[ti]) * C + hI * HD + d] = pub_e4m3(Oaccum[d]);
                }
            }
        }
    }

    // projection C -> C over all tokens
    std::vector<float> acc;
    for (uint32_t t = 0; t < tokens; ++t) {
        gemm_f16_acc(projT->f16.data(), nullptr, &headO[size_t(t) * C], C, C, acc);
        for (uint32_t i = 0; i < C; ++i)
            oTokens[size_t(t) * C + i] = f32_to_f16(acc[i]);   // raw f16 projection out
    }
}

// ---------------------------------------------------------------------------
// Global (ViT) attention: all level tokens, unnormalized weights.
// ---------------------------------------------------------------------------
void global_attention(const Model& m, uint32_t blockIdx, uint32_t C, uint32_t heads,
                      uint32_t tokensReal,
                      const F16Vec& y, F16Vec& oTokens) {
    const uint32_t HD = 32;
    const uint32_t tokens = (tokensReal + 63) / 64 * 64;   // pad to multiple of 64
    const TensorRecord* qkvT  = T(m, blockIdx, 1, "qkv");
    const TensorRecord* projT = T(m, blockIdx, 1, "proj");
    const TensorRecord* qsT   = T(m, blockIdx, 1, "qscale");
    if (!qkvT || !projT) { oTokens.assign(size_t(tokens) * C, 0); return; }

    F16Vec yP(size_t(tokens) * C, 0);
    for (uint32_t t = 0; t < tokensReal; ++t)
        for (uint32_t i = 0; i < C; ++i)
            yP[size_t(t) * C + i] = pub_e4m3(f16_to_f32(y[size_t(t) * C + i]));

    F16Vec qkvP(size_t(tokens) * 3 * C, 0);
    {
        std::vector<float> acc;
        for (uint32_t t = 0; t < tokens; ++t) {
            gemm_f16_acc(qkvT->f16.data(), nullptr, &yP[size_t(t) * C], 3 * C, C, acc);
            for (uint32_t i = 0; i < 3 * C; ++i) qkvP[size_t(t) * 3 * C + i] = f32_to_f16(acc[i]);
        }
    }
    std::vector<float> qscale(heads, 1.0f);
    if (qsT && qsT->f16.size() >= heads)
        for (uint32_t hI = 0; hI < heads; ++hI) qscale[hI] = f16_to_f32(qsT->f16[hI]);

    F16Vec headO(size_t(tokens) * C, 0);
    const float exp0 = window_exp_trick(0.0f);   // ViT uses its own constants in the
    const float exp0g = global_exp_trick(0.0f);  // reference; our port keeps one trick
    (void)exp0;                                   // family — see docs/NETWORK.md
    std::vector<float> norm(C);
    (void)norm;

    for (uint32_t hI = 0; hI < heads; ++hI) {
        // normalize + scale q,k
        F16Vec qP(size_t(tokens) * HD), kP(size_t(tokens) * HD), vP(size_t(tokens) * HD);
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint16_t* qSrc = &qkvP[size_t(t) * 3 * C + hI * HD];
            const uint16_t* kSrc = &qkvP[size_t(t) * 3 * C + C + hI * HD];
            const uint16_t* vSrc = &qkvP[size_t(t) * 3 * C + 2 * C + hI * HD];
            float qn = 0, kn = 0;
            for (int d = 0; d < int(HD); ++d) {
                qn += f16_to_f32(qSrc[d]) * f16_to_f32(qSrc[d]);
                kn += f16_to_f32(kSrc[d]) * f16_to_f32(kSrc[d]);
            }
            qn = std::sqrt(qn); kn = std::sqrt(kn);
            const float invQ = qn > 0 ? 1.0f / qn : 0.0f;
            const float invK = kn > 0 ? 1.0f / kn : 0.0f;
            const float s32 = std::sqrt(32.0f);
            for (int d = 0; d < int(HD); ++d) {
                // three separate half multiplies: q*norm, *sqrt(32), *learned
                float q = f16_to_f32(f32_to_f16(f16_to_f32(qSrc[d]) * invQ));
                q = f16_to_f32(f32_to_f16(q * s32));
                q = f16_to_f32(f32_to_f16(q * qscale[hI]));
                float k = f16_to_f32(f32_to_f16(f16_to_f32(kSrc[d]) * invK));
                qP[size_t(t) * HD + d] = pub_e4m3(q);
                kP[size_t(t) * HD + d] = pub_e4m3(k);
                vP[size_t(t) * HD + d] = pub_e4m3(f16_to_f32(vSrc[d]));
            }
        }
        // unnormalized softmax weights; denominator corrected for padding rows
        for (uint32_t a = 0; a < tokensReal; ++a) {
            float denom = 0.f;
            std::vector<float> wRow(tokens);
            for (uint32_t bI = 0; bI < tokens; ++bI) {
                float s = 0;
                for (int d = 0; d < int(HD); ++d)
                    s += f16_to_f32(qP[size_t(a) * HD + d]) * f16_to_f32(kP[size_t(bI) * HD + d]);
                float e = global_exp_trick(s);
                wRow[bI] = e;
                denom += e;
            }
            denom -= exp0g * float(tokens - tokensReal);   // padding correction
            const float invD = denom > 0 ? 1.0f / denom : 0.0f;
            std::vector<float> O(HD, 0.f);
            for (uint32_t bI = 0; bI < tokens; ++bI) {
                float wgt = e4m3_publish(wRow[bI] * invD);
                if (wgt == 0.0f && wRow[bI] > 0.0f) { /* saturated tiny weights drop out, as in reference */ }
                for (int d = 0; d < int(HD); ++d)
                    O[d] += wgt * f16_to_f32(vP[size_t(bI) * HD + d]);
            }
            for (int d = 0; d < int(HD); ++d)
                headO[size_t(a) * C + hI * HD + d] = pub_e4m3(O[d]);
        }
    }

    oTokens.assign(size_t(tokens) * C, 0);
    std::vector<float> acc;
    for (uint32_t t = 0; t < tokensReal; ++t) {
        gemm_f16_acc(projT->f16.data(), nullptr, &headO[size_t(t) * C], C, C, acc);
        for (uint32_t i = 0; i < C; ++i)
            oTokens[size_t(t) * C + i] = f32_to_f16(acc[i]);
    }
}

// ---------------------------------------------------------------------------
// one block
// ---------------------------------------------------------------------------
void block_execute(const Model& m, const BlockScheduleEntry& entry,
                   uint32_t tokensReal, const F16Vec& xIn, F16Vec& xOut) {
    const uint32_t C = entry.channels;
    const uint32_t entryTokens = entry.levelWidth * entry.levelHeight;
    if (entryTokens != tokensReal) {
        std::fprintf(stderr, "[graph] block %u token mismatch: entry %u vs state %u\n",
                     entry.index, entryTokens, tokensReal);
    }
    // FFN
    std::vector<float> fOut;
    ffn(m, entry.index, C, entry.isViT, xIn.data(), fOut);
    // ffnScale (per-channel f16)
    F16Vec y(size_t(tokensReal) * C);
    {
        const TensorRecord* fs = T(m, entry.index, 0, "scale_ffn");
        for (uint32_t t = 0; t < tokensReal; ++t) {
            for (uint32_t i = 0; i < C; ++i) {
                float xs = f16_to_f32(xIn[size_t(t) * C + i]);
                float scale = fs && fs->f16.size() > i ? f16_to_f32(fs->f16[i]) : 1.0f;
                float f = fOut[i];
                if (C == 32 && !entry.isViT) {
                    y[size_t(t) * C + i] = pub_f16(xs * scale + f);           // raw f16 add
                } else {
                    y[size_t(t) * C + i] = pub_e4m3(xs * scale + f);          // E4M3 publication
                }
            }
        }
    }
    // attention
    F16Vec oTokens;
    if (entry.isViT) {
        global_attention(m, entry.index, C, entry.heads, tokensReal, y, oTokens);
    } else {
        window_attention(m, entry.index, C, entry.heads, entry.windowPhase,
                         entry.levelWidth, entry.levelHeight, y, oTokens);
    }
    // out = y * attnScale + projOut, published E4M3
    const TensorRecord* as = T(m, entry.index, 0, "scale_attn");
    xOut.assign(size_t(tokensReal) * C, 0);
    for (uint32_t t = 0; t < tokensReal; ++t)
        for (uint32_t i = 0; i < C; ++i) {
            float ys = f16_to_f32(y[size_t(t) * C + i]);
            float scale = as && as->f16.size() > i ? f16_to_f32(as->f16[i]) : 1.0f;
            xOut[size_t(t) * C + i] = pub_e4m3(ys * scale + f16_to_f32(oTokens[size_t(t) * C + i]));
        }
}

// ---------------------------------------------------------------------------
// transitions between stages
// ---------------------------------------------------------------------------

// 2x2 box pool: ((a+b)+(c+d)) * 0.25, every step a half (reference semantics).
void pool2x2(const F16Vec& src, uint32_t sw, uint32_t sh, uint32_t C,
             F16Vec& dst, uint32_t dw, uint32_t dh) {
    dst.assign(size_t(dw) * dh * C, 0);
    for (uint32_t y = 0; y < dh; ++y) {
        for (uint32_t x = 0; x < dw; ++x) {
            for (uint32_t c = 0; c < C; ++c) {
                auto at = [&](uint32_t sx, uint32_t sy) -> float {
                    sx = std::min(sx, sw - 1); sy = std::min(sy, sh - 1);
                    return f16_to_f32(src[(size_t(sy) * sw + sx) * C + c]);
                };
                float a = at(x*2, y*2), b = at(x*2+1, y*2), cc = at(x*2, y*2+1), d = at(x*2+1, y*2+1);
                float top = f16_to_f32(f32_to_f16(a + b));
                float bot = f16_to_f32(f32_to_f16(cc + d));
                float sum = f16_to_f32(f32_to_f16(top + bot));
                dst[(size_t(y) * dw + x) * C + c] = pub_e4m3(f16_to_f32(f32_to_f16(sum * 0.25f)));
            }
        }
    }
}

void upsample2x_nearest(const F16Vec& src, uint32_t sw, uint32_t /*sh*/, uint32_t C,
                        F16Vec& dst, uint32_t dw, uint32_t dh) {
    dst.assign(size_t(dw) * dh * C, 0);
    for (uint32_t y = 0; y < dh; ++y)
        for (uint32_t x = 0; x < dw; ++x)
            for (uint32_t c = 0; c < C; ++c)
                dst[(size_t(y) * dw + x) * C + c] = src[(size_t(y / 2) * sw + (x / 2)) * C + c];
}

} // namespace

// ---------------------------------------------------------------------------
// top-level graph execution
// ---------------------------------------------------------------------------

struct CpuGraphWorkspace {
    // reserved for cross-frame caching (KV reuse etc.); stateless for now
    uint64_t lastSeed = 0;
};

bool neural_graph_execute(const Model& model, const Geometry& geom,
                          const uint16_t* features, float* head, uint64_t frameSeed,
                          CpuGraphWorkspace** persistent) {
    (void)persistent; (void)frameSeed;
    const auto& sched = model.schedule();
    if (sched.empty()) { log_error("graph: empty schedule"); return false; }
    if (!model.tensor("input_proj")) { log_error("graph: missing input_proj"); return false; }
    if (!model.tensor("head.w"))     { log_error("graph: missing head.w"); return false; }

    const uint32_t nLevels = model.config().levels;
    const uint32_t FW = geom.fieldWidth, FH = geom.fieldHeight;

    // resolve per-block geometry
    std::vector<BlockScheduleEntry> entries = sched;
    for (auto& e : entries) {
        if (e.onField) { e.levelWidth = FW; e.levelHeight = FH; }
        else {
            const Level& lv = geom.levels[std::min(e.level, nLevels - 1)];
            e.levelWidth = lv.width; e.levelHeight = lv.height;
        }
    }

    // state
    F16Vec state;                       // [tokens][C] f16 bits
    uint32_t stateW = 0, stateH = 0, stateC = 0;
    std::vector<F16Vec> skipAtLevel(nLevels + 1);   // +1 = field-level skip (block 0)
    F16Vec block0Out;

    for (size_t bi = 0; bi < entries.size(); ++bi) {
        BlockScheduleEntry& e = entries[bi];
        const bool first = (bi == 0);
        // Effective depth ordering: the padded field is SHALLOWER than level 0
        // (blocks 0/70 run on it), levels deepen upward.
        const int32_t effE = e.onField ? -1 : int32_t(e.level);

        if (first) {
            // input embedding: 16 -> C0 on the field
            const TensorRecord* ip = model.tensor("input_proj");
            const uint32_t C0 = entries[0].channels;
            stateC = C0; stateW = FW; stateH = FH;
            state.assign(size_t(FW) * FH * C0, 0);
            std::vector<float> acc;
            for (uint64_t t = 0; t < uint64_t(FW) * FH; ++t) {
                gemm_f16_acc(ip->f16.data(), nullptr, features + t * 16, C0, 16, acc);
                for (uint32_t i = 0; i < C0; ++i)
                    state[size_t(t) * C0 + i] = pub_e4m3(acc[i]);
            }
        } else {
            const BlockScheduleEntry& p = entries[bi - 1];
            const int32_t effP = p.onField ? -1 : int32_t(p.level);
            if (effP == effE && p.onField == e.onField && p.channels == e.channels) {
                // no transition
            } else if (effP == effE && p.channels == e.channels * 2) {
                // channel contraction on the same level (ViT exit)
                const TensorRecord* down = model.tensor("vitout.down");
                if (!down) { log_error("graph: missing vitout.down"); return false; }
                F16Vec proj(size_t(stateW) * stateH * e.channels);
                std::vector<float> acc;
                for (uint32_t t = 0; t < stateW * stateH; ++t) {
                    gemm_f16_acc(down->f16.data(), nullptr, &state[size_t(t) * p.channels], e.channels, p.channels, acc);
                    for (uint32_t i = 0; i < e.channels; ++i)
                        proj[size_t(t) * e.channels + i] = pub_f16(acc[i]);   // f16 out
                }
                state = std::move(proj);
                stateC = e.channels;
            } else if (effP == effE && p.channels * 2 == e.channels) {
                // channel expansion on the same level (ViT entry)
                const TensorRecord* up = model.tensor("vitin.up");
                if (!up) { log_error("graph: missing vitin.up"); return false; }
                F16Vec proj(size_t(stateW) * stateH * e.channels);
                std::vector<float> acc;
                for (uint32_t t = 0; t < stateW * stateH; ++t) {
                    gemm_f16_acc(up->f16.data(), nullptr, &state[size_t(t) * p.channels], e.channels, p.channels, acc);
                    for (uint32_t i = 0; i < e.channels; ++i)
                        proj[size_t(t) * e.channels + i] = pub_e4m3(acc[i]);
                }
                state = std::move(proj);
                stateC = e.channels;
            } else if (effE > effP) {
                // encoder transition: 2x2 box pool of the raw output, E4M3, then C -> 2C GEMM
                const std::string lvlName = p.onField ? "trans_f" : "trans" + std::to_string(p.level);
                const TensorRecord* up = model.tensor(lvlName + ".up");
                if (!up) { log_error("graph: missing %s.up", lvlName.c_str()); return false; }

                const uint32_t srcW = p.onField ? FW : geom.levels[std::min(p.level, nLevels - 1)].width;
                const uint32_t srcH = p.onField ? FH : geom.levels[std::min(p.level, nLevels - 1)].height;
                const uint32_t dstW = e.onField ? FW : geom.levels[std::min(e.level, nLevels - 1)].width;
                const uint32_t dstH = e.onField ? FH : geom.levels[std::min(e.level, nLevels - 1)].height;

                F16Vec pooled;
                pool2x2(state, srcW, srcH, p.channels, pooled, dstW, dstH);
                state.assign(size_t(dstW) * dstH * e.channels, 0);
                std::vector<float> acc;
                for (uint32_t t = 0; t < dstW * dstH; ++t) {
                    gemm_f16_acc(up->f16.data(), nullptr, &pooled[size_t(t) * p.channels], e.channels, p.channels, acc);
                    for (uint32_t i = 0; i < e.channels; ++i)
                        state[size_t(t) * e.channels + i] = pub_e4m3(acc[i]);
                }
                stateW = dstW; stateH = dstH; stateC = e.channels;
            } else {
                // decoder transition: down-GEMM at the deep level, 2x upsample,
                // + skip * transitionScale (one f16 FMA), E4M3
                const std::string lvlName = p.onField ? "trans_f" : "trans" + std::to_string(p.level);
                const TensorRecord* down = model.tensor(lvlName + ".down");
                const TensorRecord* scale = model.tensor(lvlName + ".scale");
                const TensorRecord* inScale = model.tensor(lvlName + ".in_scale");
                if (!down) { log_error("graph: missing %s.down", lvlName.c_str()); return false; }
                const uint32_t srcW = p.onField ? FW : geom.levels[std::min(p.level, nLevels - 1)].width;
                const uint32_t srcH = p.onField ? FH : geom.levels[std::min(p.level, nLevels - 1)].height;
                const uint32_t dstW = e.onField ? FW : geom.levels[std::min(e.level, nLevels - 1)].width;
                const uint32_t dstH = e.onField ? FH : geom.levels[std::min(e.level, nLevels - 1)].height;

                F16Vec proj(size_t(srcW) * srcH * e.channels);
                std::vector<float> acc;
                for (uint32_t t = 0; t < srcW * srcH; ++t) {
                    gemm_f16_acc(down->f16.data(), nullptr, &state[size_t(t) * p.channels], e.channels, p.channels, acc);
                    for (uint32_t i = 0; i < e.channels; ++i)
                        proj[size_t(t) * e.channels + i] = pub_f16(acc[i]);   // f16 out
                }
                F16Vec up;
                upsample2x_nearest(proj, srcW, srcH, e.channels, up, dstW, dstH);
                // skip add
                const F16Vec& skip = e.onField ? block0Out : skipAtLevel[e.level];
                state.assign(size_t(dstW) * dstH * e.channels, 0);
                for (uint32_t t = 0; t < dstW * dstH; ++t) {
                    for (uint32_t i = 0; i < e.channels; ++i) {
                        float v = f16_to_f32(up[size_t(t) * e.channels + i]);
                        if (inScale && inScale->f16.size() > i) v = f16_to_f32(f32_to_f16(v * f16_to_f32(inScale->f16[i])));
                        if (!skip.empty()) {
                            float s = f16_to_f32(skip[size_t(t) * e.channels + i]);
                            if (scale && scale->f16.size() > i) s = f16_to_f32(f32_to_f16(s * f16_to_f32(scale->f16[i])));
                            v = f16_to_f32(f32_to_f16(v + s));   // one f16 FMA
                        }
                        state[size_t(t) * e.channels + i] = pub_e4m3(v);
                    }
                }
                stateW = dstW; stateH = dstH; stateC = e.channels;
            }
        }

        // execute the block
        F16Vec out;
        std::fprintf(stderr, "[dbg] block %u lvl %u field %d C %u tokens %u state %zu y exp %zu\n",
                     e.index, e.level, int(e.onField), e.channels, stateW * stateH,
                     state.size(), size_t(stateW * stateH) * e.channels);
        block_execute(model, e, stateW * stateH, state, out);
        state = std::move(out);

        // save skips: block 0's field output, and each encoder stage's output
        if (bi == 0) block0Out = state;
        if (bi + 1 < entries.size()) {
            const BlockScheduleEntry& next = entries[bi + 1];
            const bool goingDeeper = next.level > e.level || (!e.onField && next.onField == false && e.onField);
            if (e.onField) skipAtLevel[nLevels] = state;    // field-level skip
            else if (goingDeeper && e.level < nLevels) skipAtLevel[e.level] = state;
        } else {
            if (e.onField) skipAtLevel[nLevels] = state;
        }
    }

    // head: C -> 4 f16 GEMM into f32
    const TensorRecord* hw = model.tensor("head.w");
    const TensorRecord* hb = model.tensor("head.b");
    const uint32_t C = stateC;
    const uint64_t tokens = uint64_t(stateW) * stateH;
    std::vector<float> acc;
    for (uint64_t t = 0; t < tokens; ++t) {
        gemm_f16_acc(hw->f16.data(), hb ? hb->f16.data() : nullptr, &state[size_t(t) * C], 4, C, acc);
        head[t * 4 + 0] = acc[0];
        head[t * 4 + 1] = acc[1];
        head[t * 4 + 2] = acc[2];
        head[t * 4 + 3] = acc[3];
    }
    return true;
}

} // namespace opendlss
