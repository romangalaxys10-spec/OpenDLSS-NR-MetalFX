// Common.glsl — shared helpers for every OpenDLSS-NR Vulkan compute kernel.
//
// The bit-tricks here MUST stay identical to core/include/opendlss/fp16.h and
// e4m3.h (the CPU reference) — they are checked by the parity harness.
// Common.metal is the Metal port of the same helpers; where Metal used a
// log2()-based normal path in odl_e4m3_encode, this port uses the CPU's
// bit-exact f32-field arithmetic instead (a hardware log2 can be off by one
// ulp near powers of two, which would desynchronize the E4M3 grid). All
// other numerics are line-for-line ports of Common.metal.
//
// GLSL 450 core only — no extensions, no float16_t type. Half values are
// stored one-per-uint32 SSBO word (low 16 bits) so that concurrent writes by
// different threads can never alias the same storage word; the 2x memory
// overhead buys race-free portability. Conversion between f32 and the half
// grid goes through the software codec below (round-to-nearest-even), not
// through packHalf2x16, so rounding is defined on every driver.
//
// Push-constant convention (one 16-word block shared by every kernel; the
// host always pushes 64 bytes; unused words are zero):
//   odl_preprocess         [0..4]  cond(style,tone,structure,skin,autoMask) f32
//                          [5..6]  validSize.xy   [7] frameSeed  [8] hasHistory
//   odl_input_embed        [0..1]  fieldSize.xy   [2] channels
//   odl_ffn                [0] tokens [1] C [2] hidden [3] paths
//                          [4] perPathOut [5] inSlice (0 = full C)
//                          [6] applySilu [7] hasW3
//   odl_window_attention   [0..1] levelSize.xy [2] C [3] heads [4] phase
//                          [5] hasPrior
//   odl_global_attention   [0] tokensReal [1] C [2] heads
//   odl_block_skip         [0] tokens [1] C [2] rawF16 [3] hasScale
//   odl_block_epilogue     [0] tokens [1] C [2] hasScale
//   odl_pool2x2            [0..1] srcSize.xy [2..3] dstSize.xy [4] C
//   odl_channel_gemm       [0] tokens [1] outCh [2] inCh [3] pubMode
//                          [4] hasBias
//   odl_decoder_skip       [0] tokens [1] C [2] hasSkip [3] hasScale
//                          [4] hasInScale  (binding 3 = in_scale)
//   odl_nearest_upsample2x [0..1] srcSize.xy [2..3] dstSize.xy [4] C
//   odl_head               [0] tokens [1] C [2] hasBias
//   odl_denoise            [0] strength (f32)
//   odl_upscale_h          [0] dstW
//   odl_upscale_v          [0] dstH
//   odl_sharpen            [0] amount (f32)
//   odl_motion             (none)
//   odl_reproject          [0] hasHistory
//   odl_temporal_blend     [0] maxBlend (f32)
//   odl_head_composite     [0] blendScale (f32)  [1] hasHistory
//
// This file is #included by every kernel .comp file; the INCLUDING file owns
// the `#version 450` directive and the GL_GOOGLE_include_directive extension
// (required for #include processing by glslangValidator).
//
// Optional Metal buffers/parameters (null prior / bias / skip / history)
// cannot be null-checked in GLSL, so the host binds a 1-element dummy
// resource and passes a hasX flag; semantics are identical.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
// (No #version here — included files must not re-declare the version.)

// ---------------------------------------------------------------------------
// push constants (see the slot table in the file header)
// ---------------------------------------------------------------------------
layout(push_constant) uniform Pcb {
    uint pc[16];
} pcb;

uint pcu(uint i)     { return pcb.pc[i]; }
float pcf(uint i)    { return uintBitsToFloat(pcb.pc[i]); }
uvec2 pcuv2(uint i)  { return uvec2(pcb.pc[i], pcb.pc[i + 1u]); }

// ---------------------------------------------------------------------------
// minimal 64-bit unsigned arithmetic (uvec2 = (lo, hi)); used by the pixel
// hash so we don't require the shaderInt64 feature on the device.
// ---------------------------------------------------------------------------

// full 32x32 -> 64 product, returned as (lo, hi)
uvec2 odl_mul32x32(uint a, uint b) {
    uint a0 = a & 0xFFFFu, a1 = a >> 16;
    uint b0 = b & 0xFFFFu, b1 = b >> 16;
    uint p00 = a0 * b0;
    uint p01 = a0 * b1;
    uint p10 = a1 * b0;
    uint p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    uint lo  = (p00 & 0xFFFFu) | (mid << 16);
    uint hi  = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
    return uvec2(lo, hi);
}

// 64x64 -> 64 (wrapping) product
uvec2 odl_mul64(uvec2 a, uvec2 b) {
    uvec2 ll = odl_mul32x32(a.x, b.x);   // a.lo * b.lo
    uvec2 lh = odl_mul32x32(a.x, b.y);   // a.lo * b.hi
    uvec2 hl = odl_mul32x32(a.y, b.x);   // a.hi * b.lo
    // result = ll + (lh.lo << 32) + (hl.lo << 32) mod 2^64; the hi words of
    // lh/hl only feed bits >= 64 and wrap away.
    uint hi = ll.y + lh.x + hl.x;        // mod 2^32, exactly right
    return uvec2(ll.x, hi);
}

uvec2 odl_shr64(uvec2 a, uint s) {
    if (s == 0u) return a;
    if (s >= 64u) return uvec2(0u, 0u);
    if (s >= 32u) return uvec2(a.y >> (s - 32u), 0u);
    return uvec2((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}

uvec2 odl_xor64(uvec2 a, uvec2 b) { return a ^ b; }

// ---------- pixel hash (matches preprocess_pixel_hash / Common.metal) ----------
uint odl_hash(uint x, uint y, uint frameIndex, uint seed) {
    uvec2 k1 = uvec2(0x7F4A7C15u, 0x9E3779B9u);   // 0x9E3779B97F4A7C15
    uvec2 k2 = uvec2(0x27D4EB4Fu, 0xC2B2AE3Du);   // 0xC2B2AE3D27D4EB4F
    uvec2 k3 = uvec2(0x9E3779F9u, 0x165667B1u);   // 0x165667B19E3779F9
    uvec2 m1 = uvec2(0xED558CCDu, 0xFF51AFD7u);   // 0xFF51AFD7ED558CCD
    uvec2 m2 = uvec2(0x1A85EC53u, 0xC4CEB9FEu);   // 0xC4CEB9FE1A85EC53
    uvec2 h = odl_xor64(uvec2(seed, 0u), uvec2(0u, 0u));
    h = odl_xor64(h, odl_mul64(uvec2(x, 0u), k1));
    h = odl_xor64(h, odl_mul64(uvec2(y, 0u), k2));
    h = odl_xor64(h, odl_mul64(uvec2(frameIndex, 0u), k3));
    h = odl_xor64(h, odl_shr64(h, 33u)); h = odl_mul64(h, m1);
    h = odl_xor64(h, odl_shr64(h, 33u)); h = odl_mul64(h, m2);
    h = odl_xor64(h, odl_shr64(h, 33u));
    uvec2 top = odl_xor64(h, odl_shr64(h, 32u));
    return top.x ^ top.y;
}

// ---------------------------------------------------------------------------
// software IEEE-754 binary16 codec — line-for-line port of fp16.h
// (round-to-nearest-even; identical semantics to hardware F16C)
// ---------------------------------------------------------------------------
uint odl_f32_to_f16(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = (bits >> 16) & 0x8000u;
    int  ex   = int((bits >> 23) & 0xFFu) - 127;
    uint man  = bits & 0x007FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu) {                 // Inf / NaN
        if (man != 0u) {
            uint m16 = man >> 13;
            if (m16 == 0u) m16 = 1u;
            return sign | 0x7C00u | m16;
        }
        return sign | 0x7C00u;
    }
    if (ex > 15) return sign | 0x7C00u;                    // overflow -> Inf
    if (ex >= -14) {                                       // normal
        uint man16 = man >> 13;
        uint rnd = (man >> 12) & 1u;
        man16 += rnd;
        if (man16 >> 10 != 0u) { man16 = 0u; ++ex; }       // mantissa overflow
        if (ex > 15) return sign | 0x7C00u;
        return sign | (uint(ex + 15) << 10) | man16;
    }
    if (ex < -25) return sign;                             // underflow to zero
    // subnormal: shift with rounding (ex in [-25, -15] -> shift in [1, 11])
    uint shift   = uint(-14 - ex);
    uint manFull = man | 0x00800000u;
    uint man16   = manFull >> (13 + shift);
    uint rem     = manFull << (19 - shift);
    uint halfUlp = 1u << 31;
    if (rem > halfUlp || (rem == halfUlp && (man16 & 1u) != 0u)) ++man16;
    return sign | man16;
}

float odl_f16_to_f32(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint ex16 = (h >> 10) & 0x1Fu;
    uint man  = h & 0x03FFu;
    uint bits;
    if (ex16 == 0u) {
        if (man == 0u) { bits = sign; }
        else {                                           // subnormal -> normalize
            int e = -1;
            uint m = man;
            do { ++e; m <<= 1; } while ((m & 0x0400u) == 0u);
            m &= 0x03FFu;
            bits = sign | (uint(127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (ex16 == 31u) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | (uint(ex16 - 15 + 127) << 23) | (man << 13);
    }
    return uintBitsToFloat(bits);
}

// SSBO half access: one half per uint32 word (low 16 bits, high bits zero).
#define ODL_LD(buf, i)  odl_f16_to_f32((buf)[i] & 0xFFFFu)
#define ODL_ST(buf, i, v) (buf)[i] = odl_f32_to_f16(v)

// ---------------------------------------------------------------------------
// E4M3 (OCP FP8) codec — bit-exact port of core/include/opendlss/e4m3.h.
//   1 | 4 (bias 7) | 3 ; max normal 448 ; 0xFF/0x7F = NaN ; saturating RTNE.
// ---------------------------------------------------------------------------
float odl_e4m3_decode(uint code) {
    uint sign = (code & 0x80u) << 24;
    uint e = (code >> 3) & 0x0Fu;
    uint m = code & 0x07u;
    if (e == 15u && m == 7u) return uintBitsToFloat(0x7FC00000u);   // NaN
    float v;
    if (e == 0u) {
        if (m == 0u) v = 0.0;                                       // zero
        else v = ldexp(float(m), -9);                               // m * 2^-9
    } else {
        v = ldexp(1.0 + float(m) / 8.0, int(e) - 7);
    }
    return sign != 0u ? -v : v;
}

uint odl_e4m3_encode(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = ((bits >> 31) & 1u) << 7;                           // 0x80
    uint expf = (bits >> 23) & 0xFFu;
    if (expf == 0xFFu) {                                            // Inf / NaN
        if ((bits & 0x007FFFFFu) != 0u) return 0x7Fu;               // NaN
        return sign | 0x7Eu;                                        // inf saturates
    }
    int  ex = int(expf) - 127;
    uint man = bits & 0x007FFFFFu;

    if (expf == 0u) {                                               // f32 zero/subnormal
        if (man == 0u) return sign;
        if (abs(value) < 0.5 * ldexp(1.0, -9)) return sign;         // below min subnormal
    }

    if (ex < -6) {                                                  // subnormal in E4M3
        if (ex < -30) return sign;
        float av = abs(value);
        float m_f = av * 512.0;
        uint m = uint(m_f);
        float frac = m_f - float(m);
        if (frac > 0.5 || (frac == 0.5 && (m & 1u) != 0u)) ++m;
        if (m >= 8u) return sign | (1u << 3);                       // min normal 2^-6
        return sign | m;
    }
    if (ex > 8 || (ex == 8 && man > 0x00400000u)) return sign | 0x7Eu;  // > 448 saturate
    uint man3 = man >> 20;
    uint rem  = man & 0x000FFFFFu;
    uint hulp = 1u << 19;
    if (rem > hulp || (rem == hulp && (man3 & 1u) != 0u)) {
        ++man3;
        if (man3 == 8u) { man3 = 0u; ++ex; if (ex > 8) return sign | 0x7Eu; }
    }
    return sign | (uint(ex + 7) << 3) | man3;
}

// publication = quantize onto the E4M3 grid, keep f32
float odl_pub_e4m3(float v) { return odl_e4m3_decode(odl_e4m3_encode(v)); }

// ---------------------------------------------------------------------------
// attention exponent bit-tricks (ports of fp16.h / Common.metal) — every
// intermediate goes through the half grid exactly as the CPU reference does.
// ---------------------------------------------------------------------------

// window blocks:
//   x = clamp(half(0.044921875 s + 1.30078125), 1.03125, 1.5693359375)
//   exp = reinterpret_half((bits(x) << 5) ^ 0x8000) == 2^(1.4375 s - 5.375)
float odl_window_exp(float s) {
    uint xb = odl_f32_to_f16(0.044921875 * s + 1.30078125);
    float xf = odl_f16_to_f32(xb);
    xf = clamp(xf, 1.03125, 1.5693359375);
    uint b = (odl_f32_to_f16(xf) << 5) & 0xFFFFu;
    b ^= 0x8000u;
    return odl_f16_to_f32(b);
}
// global ViT blocks:
//   x = clamp(half(0.08953947 s + 1.70936143), 1.439453125, 1.9775390625)
//   exp = reinterpret_half((bits(x) << 4) + 0x4000) == 2^(1.4326 s - 3.6502)
float odl_global_exp(float s) {
    uint xb = odl_f32_to_f16(0.08953947 * s + 1.70936143);
    float xf = odl_f16_to_f32(xb);
    xf = clamp(xf, 1.439453125, 1.9775390625);
    uint b = (odl_f32_to_f16(xf) << 4) & 0xFFFFu;
    b = b + 0x4000u;
    return odl_f16_to_f32(b);
}

float odl_silu(float v) { return v / (1.0 + exp(-v)); }

// mirrored field sampling: 2*valid - x - 2 off the valid rect (no edge repeat)
int odl_mirror(int v, uint valid) {
    if (v >= 0 && v < int(valid)) return v;
    return 2 * int(valid) - v - 2;
}

// Lanczos3 resampling kernel (separable passes)
float odl_lanczos3(float x) {
    x = abs(x);
    if (x < 1e-6) return 1.0;
    if (x >= 3.0) return 0.0;
    const float pi = 3.14159265358979;
    return 3.0 * sin(pi * x) * sin(pi * x / 3.0) / (pi * pi * x * x);
}
