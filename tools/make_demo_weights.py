#!/usr/bin/env python3
# make_demo_weights.py — generate a complete, structurally valid model directory
# for OpenDLSS-NR MetalFX.
#
# The network is REAL (same block semantics as the full graph: fp16 storage,
# E4M3 publications, cosine window attention, one global ViT block, U-net
# transitions) but the weights are UNTRAINED: they are initialized small and
# near-identity so the network behaves as a mild, stable residual filter.
# This lets every demo, test and CI job exercise the full machinery without
# any proprietary weight data. Replace with trained weights of the same
# manifest layout (docs/WEIGHTS.md) to get the real thing.
#
# Usage: python3 tools/make_demo_weights.py --out models/demo-nr [--seed 1234]
#
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
import argparse
import hashlib
import json
import math
import os
import struct

# ---------------------------------------------------------------------------
# E4M3 (OCP FP8) encoding — matches core/include/opendlss/e4m3.h bit for bit.
# ---------------------------------------------------------------------------
E4M3_EXP_BIAS = 7
E4M3_MAX = 448.0


def e4m3_encode(v: float) -> int:
    if math.isnan(v):
        return 0x7F
    sign = 0x80 if (math.copysign(1.0, v) < 0 and v != 0.0) else 0x00
    v = abs(v)
    if math.isinf(v) or v > E4M3_MAX:
        return sign | 0x7E
    if v == 0.0:
        return sign
    # decompose
    exp = math.floor(math.log2(v))
    if v < 1.0:
        # floor(log2) for subnormal boundary handling
        exp = math.floor(math.log2(v))
    # try normal representation
    mant = v / (2.0 ** exp)
    if mant >= 2.0:
        mant /= 2.0
        exp += 1
    # E4M3 normal: 1.mmm * 2^(e-7), e in 1..15 (e=15,m=7 is NaN)
    # subnormal: 0.mmm * 2^-6  (v = m/8 * 2^-6, m=1..7)
    if exp < -6:
        # subnormal: v / 2^-9 = m/8 -> m = v * 2^12
        m_f = v * (2.0 ** 12) * 8.0 / 8.0  # v * 2^12 * (8/8) = m
        m_f = v * 4096.0
        m = int(round(m_f))
        # round to nearest even on ties
        frac = m_f - m
        if frac > 0.5 or (frac == 0.5 and (m & 1)):
            m += 1
        if m >= 8:
            return sign | (1 << 3)  # min normal 2^-6
        return sign | m
    if exp > 8 or (exp == 8 and mant > 1.75):
        return sign | 0x7E
    # normal with 3 mantissa bits: quantize mant in [1,2) to steps of 1/8
    q = mant * 8.0  # 8..16
    qi = int(round(q - 8.0 + 0.0))  # mantissa code 0..7 (value 1.mmm)
    frac = q - 8.0 - qi
    if frac > 0.5 or (frac == 0.5 and (qi & 1)):
        qi += 1
    if qi >= 8:
        qi = 0
        exp += 1
        if exp > 8 or (exp == 8 and qi > 6):
            return sign | 0x7E
        if exp == 8:
            # check max normal 448 = 1.75 * 2^8
            return sign | (15 << 3) | 6 if True else 0
    if exp == 8 and qi > 6:
        return sign | 0x7E
    return sign | ((exp + 7) << 3) | qi


def pack_e4m3(values) -> bytes:
    return bytes(e4m3_encode(float(v)) for v in values)


def f16_encode(v: float) -> int:
    """IEEE binary16 round-to-nearest-even (for scalar tensors stored as f16
    inside the E4M3 stage files we instead publish through E4M3; this helper is
    used when a scalar must be stored as half, e.g. qscale/blend_scale)."""
    if math.isnan(v):
        return 0x7E00
    sign = 0x8000 if (math.copysign(1.0, v) < 0 and v != 0.0) else 0
    v = abs(v)
    if v == 0.0:
        return sign
    if math.isinf(v) or v > 65504.0:
        return sign | 0x7C00
    exp = math.floor(math.log2(v))
    mant = v / (2.0 ** exp)
    if mant >= 2.0:
        mant /= 2
        exp += 1
    if exp >= -14:
        q = round((mant - 1.0) * 1024.0)
        if q >= 1024:
            q = 0
            exp += 1
            if exp > 15:
                return sign | 0x7C00
        return sign | ((exp + 15) << 10) | q
    # subnormal
    q = round(v * (2.0 ** 24))
    if q >= 1024:
        return sign | ((1) << 10)
    return sign | q


def f16_bytes(v: float) -> bytes:
    return struct.pack("<H", f16_encode(v))


# ---------------------------------------------------------------------------
# tiny tensor bookkeeping
# ---------------------------------------------------------------------------
class StageWriter:
    def __init__(self):
        self.buf = bytearray()
        self.tensors = []

    def add(self, name, block, layer, parameter, values, shape=None):
        offset = len(self.buf)
        data = pack_e4m3(values)
        self.buf += data
        rec = {
            "name": name,
            "block": block,
            "layer": layer,
            "parameter": parameter,
            "stage": "weights",
            "stageOffset": offset,
            "byteLength": len(data),
        }
        if shape:
            rec["shape"] = list(shape)
        self.tensors.append(rec)
        return data


def rng_stream(seed):
    # deterministic PRNG (numpy-free): splitmix64-ish
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return z

    return nxt


def randn(rng, n, scale):
    # Box-Muller from the integer stream
    out = []
    while len(out) < n:
        u1 = ((rng() >> 11) + 1) / 9007199254740993.0
        u2 = (rng() >> 11) / 9007199254740992.0
        g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        out.append(g * scale)
    return out[:n]


# ---------------------------------------------------------------------------
# the demo schedule
# ---------------------------------------------------------------------------
# field -> L0 -> L1 -> (ViT) -> L1 -> L0 -> field, 32 ch, one ViT block at 64 ch
# 7 blocks total; phases cycle per level like the reference.
SCHEDULE = [
    {"index": 0, "level": 0, "channels": 32, "heads": 1, "vit": False, "phase": 0, "onField": True},
    {"index": 1, "level": 0, "channels": 32, "heads": 1, "vit": False, "phase": 1, "onField": False},
    {"index": 2, "level": 1, "channels": 32, "heads": 1, "vit": False, "phase": 2, "onField": False},
    {"index": 3, "level": 1, "channels": 64, "heads": 2, "vit": True,  "phase": 0, "onField": False},
    {"index": 4, "level": 1, "channels": 32, "heads": 1, "vit": False, "phase": 3, "onField": False},
    {"index": 5, "level": 0, "channels": 32, "heads": 1, "vit": False, "phase": 0, "onField": False},
    {"index": 6, "level": 0, "channels": 32, "heads": 1, "vit": False, "phase": 1, "onField": True},
]


def build_model(seed: int):
    st = StageWriter()
    rng = rng_stream(seed)

    def mat(out_dim, in_dim, scale):
        return randn(rng, out_dim * in_dim, scale)

    # input embedding 16 -> 32: pass the 16 lanes through, scaled 0.5
    ip = [0.0] * (32 * 16)
    for i in range(16):
        ip[i * 16 + i] = 0.5
    st.add("input_proj", 0, 0, "input_proj", ip, [32, 16])

    for entry in SCHEDULE:
        b = entry["index"]
        C = entry["channels"]
        if entry["vit"]:
            hidden = 4096
            st.add(f"block{b}.layer0.w1", b, 0, "w1", mat(hidden, C, 0.02), [hidden, C])
            st.add(f"block{b}.layer0.b1", b, 0, "b1", [0.0] * hidden, [hidden])
            st.add(f"block{b}.layer0.w2", b, 0, "w2", mat(C, hidden, 0.02), [C, hidden])
            st.add(f"block{b}.layer0.b2", b, 0, "b2", [0.0] * C, [C])
        else:
            st.add(f"block{b}.layer0.w1", b, 0, "w1", mat(128, C, 0.05), [128, C])
            st.add(f"block{b}.layer0.b1", b, 0, "b1", [0.0] * 128, [128])
            st.add(f"block{b}.layer0.w2", b, 0, "w2", mat(C, 128, 0.05), [C, 128])
            st.add(f"block{b}.layer0.b2", b, 0, "b2", [0.0] * C, [C])
        # ffnScale / attnScale: neutral (identity skip)
        st.add(f"block{b}.layer0.scale_ffn", b, 0, "scale_ffn", [1.0] * C, [C])
        st.add(f"block{b}.layer0.scale_attn", b, 0, "scale_attn", [1.0] * C, [C])
        # attention
        heads = entry["heads"]
        st.add(f"block{b}.layer1.qkv", b, 1, "qkv", mat(3 * C, C, 0.05), [3 * C, C])
        st.add(f"block{b}.layer1.proj", b, 1, "proj", mat(C, C, 0.05), [C, C])
        if not entry["vit"]:
            st.add(f"block{b}.layer1.prior", b, 1, "prior",
                   randn(rng, heads * 64 * 64, 0.02), [heads, 64, 64])
        st.add(f"block{b}.layer1.qscale", b, 1, "qscale", [1.0] * heads, [heads])

    # transitions (flat demo ladder: square channel matrices)
    st.add("trans_f.up",   0, 0, "trans_up",   [0.5] * (32 * 32), [32, 32])   # field -> L0
    st.add("trans0.up",    1, 0, "trans_up",   [0.5] * (32 * 32), [32, 32])   # L0 -> L1
    st.add("vitin.up",     3, 0, "trans_up",   mat(64, 32, 0.05), [64, 32])   # into ViT
    st.add("vitout.down",  4, 0, "trans_down", mat(32, 64, 0.05), [32, 64])   # out of ViT
    st.add("trans1.down",  4, 0, "trans_down", [0.5] * (32 * 32), [32, 32])   # L1 -> L0
    st.add("trans1.scale", 4, 0, "trans_scale", [1.0] * 32, [32])
    st.add("trans0.down",  5, 0, "trans_down", [0.5] * (32 * 32), [32, 32])   # L0 -> field
    st.add("trans0.scale", 5, 0, "trans_scale", [1.0] * 32, [32])
    st.add("trans0.in_scale", 5, 0, "trans_in_scale", [1.0] * 32, [32])

    # head: 32 -> 4 (RGB residual + blend logit)
    head_w = randn(rng, 4 * 32, 0.01)
    st.add("head.w", 6, 0, "head_w", head_w, [4, 32])
    st.add("head.b", 6, 0, "head_b", [0.0, 0.0, 0.0, -1.5], [4])
    # blend scale: how much of the history the head may blend (video/game)
    st.add("head.blend_scale", 6, 0, "blend_scale", [0.9], [1])

    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/demo-nr")
    ap.add_argument("--seed", type=int, default=0xA11CE)
    args = ap.parse_args()

    st = build_model(args.seed)
    os.makedirs(args.out, exist_ok=True)
    blob = bytes(st.buf)
    with open(os.path.join(args.out, "weights.bin"), "wb") as f:
        f.write(blob)
    sha = hashlib.sha256(blob).hexdigest()

    manifest = {
        "format": "opendlss-nr-metalfx/1",
        "config": {"levels": 2, "minField": 64, "windowSize": 8},
        "schedule": SCHEDULE,
        "stages": [{
            "id": "weights",
            "file": "weights.bin",
            "packedByteLength": len(blob),
            "sha256": sha,
        }],
        "tensors": st.tensors,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"model written to {args.out} ({len(blob)} bytes of E4M3 weights, "
          f"{len(st.tensors)} tensors, sha256 {sha[:12]}...)")


if __name__ == "__main__":
    main()
