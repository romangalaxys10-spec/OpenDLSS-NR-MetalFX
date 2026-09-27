#!/usr/bin/env python3
# export_weights_pytorch.py — export a trained PyTorch checkpoint into the
# OpenDLSS-NR MetalFX manifest format (docs/WEIGHTS.md).
#
# The training side defines the same block family (window attention + FFN +
# transitions, fp16 storage, E4M3-quantized publications) in PyTorch; after
# training, this script re-lays the weights into the flat E4M3 stage file the
# runtimes consume. The network architecture itself (depth, channels, ViT
# blocks) must match the schedule written into the manifest — this script
# derives both from one `--schedule` description so they cannot drift.
#
# Usage (schema example):
#   python3 tools/export_weights_pytorch.py --checkpoint ckpt.pt \
#       --schedule tools/reference_schedule.json --out models/my-nr
#
# The checkpoint is expected to map parameter names like
#   blocks.<i>.ffn.w1 / blocks.<i>.ffn.w2 / blocks.<i>.ffn.w3
#   blocks.<i>.attn.qkv / blocks.<i>.attn.proj / blocks.<i>.attn.prior
#   blocks.<i>.ffn_scale / blocks.<i>.attn_scale / blocks.<i>.qscale
#   trans.<L>.up / trans.<L>.down / trans.<L>.scale / head.w / head.b
# to torch tensors (float32/float16).
#
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
import argparse
import hashlib
import json
import os
import sys


def f32_to_f16_bits(v):
    """IEEE binary16 RTNE — identical to core/include/opendlss/fp16.h."""
    import struct
    (bits,) = struct.unpack("<I", struct.pack("<f", v))
    sign = (bits >> 16) & 0x8000
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    if ((bits >> 23) & 0xFF) == 0xFF:
        return sign | 0x7C00 | (1 if man else 0) if man else sign | 0x7C00
    if exp > 15:
        return sign | 0x7C00
    if exp >= -14:
        man16 = man >> 13
        man16 += (man >> 12) & 1
        if man16 >> 10:
            man16 = 0
            exp += 1
        if exp > 15:
            return sign | 0x7C00
        return sign | (exp + 15) << 10 | man16
    if exp < -25:
        return sign
    man_full = man | 0x800000
    shift = -14 - exp
    man16 = man_full >> (13 + shift)
    rem = man_full << (19 - shift)
    if rem > (1 << 31) or (rem == (1 << 31) and (man16 & 1)):
        man16 += 1
    return sign | man16


def e4m3_encode(v):
    """OCP E4M3, RTNE + saturation — identical to core/include/opendlss/e4m3.h."""
    import math
    if math.isnan(v):
        return 0x7F
    sign = 0x80 if v < 0 else 0
    a = abs(v)
    if math.isinf(a) or a >= 448.0:
        return sign | 0x7E
    if a == 0.0:
        return sign
    m_f = a * 512.0
    if m_f < 8.0:
        m = int(m_f)
        frac = m_f - m
        if frac > 0.5 or (frac == 0.5 and (m & 1)):
            m += 1
        if m >= 8:
            return sign | 0x08
        return sign | m
    exp = int(math.floor(math.log2(a)))
    mant = a / (2.0 ** exp)
    q = mant * 8.0
    mi = int(q)
    frac = q - mi
    if frac > 0.5 or (frac == 0.5 and (mi & 1)):
        mi += 1
    if mi >= 16:
        mi >>= 1
        exp += 1
    mi -= 8
    if exp > 8 or (exp == 8 and mi > 6):
        return sign | 0x7E
    return sign | ((exp + 7) << 3) | mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help=".pt/.pth file (torch.load-able)")
    ap.add_argument("--schedule", required=True, help="schedule JSON (see docs/WEIGHTS.md)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default='{"levels":6,"minField":320,"windowSize":8}')
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch required for export: pip install torch")
    try:
        import numpy as np
    except ImportError:
        sys.exit("numpy required for export")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    sched = json.load(open(args.schedule))
    config = json.loads(args.config)

    os.makedirs(args.out, exist_ok=True)
    blob = bytearray()
    tensors = []

    def add(name, block, layer, param, values, shape):
        nonlocal blob
        offset = len(blob)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        data = bytes(e4m3_encode(float(v)) for v in arr)
        blob += data
        tensors.append({
            "name": name, "block": block, "layer": layer, "parameter": param,
            "stage": "weights", "stageOffset": offset, "byteLength": len(data),
            "shape": list(shape),
        })

    def get(name):
        key = name if name in state else name.replace(".", "_")
        if key not in state:
            raise KeyError(f"checkpoint missing '{name}'")
        return state[key].detach().to(torch.float32).numpy()

    for entry in sched:
        b, i = entry["index"], entry["index"]
        C, heads = entry["channels"], entry["heads"]
        vit = entry.get("vit", False)
        hidden = 4096 if vit else 128
        add(f"block{b}.layer0.w1", b, 0, "w1", get(f"blocks.{i}.ffn.w1"), [hidden, C])
        add(f"block{b}.layer0.b1", b, 0, "b1", get(f"blocks.{i}.ffn.b1"), [hidden])
        add(f"block{b}.layer0.w2", b, 0, "w2", get(f"blocks.{i}.ffn.w2"), [C, hidden])
        add(f"block{b}.layer0.b2", b, 0, "b2", get(f"blocks.{i}.ffn.b2"), [C])
        if not vit and C not in (32,):
            add(f"block{b}.layer0.w3", b, 0, "w3", get(f"blocks.{i}.ffn.w3"), [C, C])
            add(f"block{b}.layer0.b3", b, 0, "b3", get(f"blocks.{i}.ffn.b3"), [C])
        add(f"block{b}.layer0.scale_ffn", b, 0, "scale_ffn", get(f"blocks.{i}.ffn_scale"), [C])
        add(f"block{b}.layer0.scale_attn", b, 0, "scale_attn", get(f"blocks.{i}.attn_scale"), [C])
        add(f"block{b}.layer1.qkv", b, 1, "qkv", get(f"blocks.{i}.attn.qkv"), [3 * C, C])
        add(f"block{b}.layer1.proj", b, 1, "proj", get(f"blocks.{i}.attn.proj"), [C, C])
        if not vit:
            add(f"block{b}.layer1.prior", b, 1, "prior", get(f"blocks.{i}.attn.prior"), [heads, 64, 64])
        add(f"block{b}.layer1.qscale", b, 1, "qscale", get(f"blocks.{i}.qscale"), [heads])

    if "input_proj" in state or "input_proj" in {k.replace("_", ".") for k in state}:
        add("input_proj", 0, 0, "input_proj", get("input_proj"),
            [sched[0]["channels"], 16])
    for key, param in (("head.w", "head_w"), ("head.b", "head_b")):
        add(key, sched[-1]["index"], 0, param, get(key),
            [4, sched[-1]["channels"]] if key == "head.w" else [4])
    if "head.blend_scale" in state:
        add("head.blend_scale", sched[-1]["index"], 0, "blend_scale",
            get("head.blend_scale"), [1])

    with open(os.path.join(args.out, "weights.bin"), "wb") as f:
        f.write(bytes(blob))
    sha = hashlib.sha256(bytes(blob)).hexdigest()

    manifest = {
        "format": "opendlss-nr-metalfx/1",
        "config": config,
        "schedule": sched,
        "stages": [{"id": "weights", "file": "weights.bin",
                    "packedByteLength": len(blob), "sha256": sha}],
        "tensors": tensors,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"exported {len(tensors)} tensors ({len(blob)} bytes) to {args.out}")


if __name__ == "__main__":
    main()
