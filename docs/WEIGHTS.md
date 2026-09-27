# Weights and the manifest

A model directory is self-describing: `manifest.json` plus stage files holding
E4M3-packed weights. The layout is byte-compatible with the reference
repository's model directories, so full-size original manifests load
unmodified when you have one.

## manifest.json

```json
{
  "format": "opendlss-nr-metalfx/1",
  "config": { "levels": 6, "minField": 320, "windowSize": 8 },
  "schedule": [
    { "index": 0, "level": 0, "channels": 32, "heads": 1,
      "vit": false, "phase": 0, "onField": true },
    ...
  ],
  "stages": [
    { "id": "weights", "file": "weights.bin",
      "packedByteLength": 653549, "sha256": "…" }
  ],
  "tensors": [
    { "name": "block0.layer0.w1", "block": 0, "layer": 0, "parameter": "w1",
      "stage": "weights", "stageOffset": 0, "byteLength": 4096, "shape": [128, 32] },
    ...
  ]
}
```

- `config` — geometry parameters (reference values: 6 levels, floor 320, window 8).
- `schedule` — the block ladder, in execution order. `level` is the resolution
  level; `onField: true` marks the field-level blocks (the reference's block
  0/70 class); `vit: true` marks global-attention blocks; `phase` is the
  shifted-window origin cycle 0..3.
- `stages` — packed E4M3 byte blobs with length and sha256.
- `tensors` — records locating each tensor in its stage (1 byte per E4M3
  element). `shape` is `[out, in]` for matrices (row-major) and natural for
  vectors/priors.

## Tensor naming

| Name | Shape | Role |
| --- | --- | --- |
| `input_proj` | [C0, 16] | input embedding from the 16 lanes |
| `block<b>.layer0.w1/b1/w2/b2` | per FFN width | FFN (dense for C=32 and ViT) |
| `block<b>.layer0.w3/b3` | [C, C] | wide-block contraction (C=64..512) |
| `block<b>.layer0.scale_ffn` | [C] | per-channel ffnScale |
| `block<b>.layer0.scale_attn` | [C] | per-channel attnScale |
| `block<b>.layer1.qkv` | [3C, C] | QKV projection |
| `block<b>.layer1.proj` | [C, C] | attention output projection |
| `block<b>.layer1.prior` | [heads, 64, 64] | learned attention prior (window blocks) |
| `block<b>.layer1.qscale` | [heads] | learned per-head q temperature |
| `trans<L>.up` | [2C, C] | encoder pool+expand GEMM from level L |
| `trans<L>.down` | [C, 2C] | decoder contract GEMM from level L |
| `trans<L>.scale` | [C] | decoder skip scale |
| `trans<L>.in_scale` | [C] | decoder upsample scale (adapter path) |
| `trans_f.up` / `trans_f.down` / … | — | field↔L0 transitions |
| `vitin.up` / `vitout.down` | — | ViT channel doublings |
| `head.w` / `head.b` | [4, C] / [4] | the head (RGB residual + blend logit) |
| `head.blend_scale` (parameter `blend_scale`) | [1] | head temporal blend scale |

## Generating weights

```bash
# untrained demo model — stable, runs everywhere, exercises the full machinery
python3 tools/make_demo_weights.py --out models/demo-nr --seed 1234

# trained model export (from your own training run)
python3 tools/export_weights_pytorch.py --checkpoint ckpt.pt --out models/my-nr
```

The demo generator writes the complete manifest + one E4M3 stage with a
deterministic PRNG (documented splitmix64 stream), near-identity scales, and a
head biased to a small residual with a bounded blend logit — a structurally
real, numerically safe network.

## Quantization rules (what "publication" means)

- weights are stored as E4M3 bytes; the host decodes to f16 once at load;
- activations flow as f16 but are published through the E4M3 grid at every
  inter-layer boundary, on the QKV input, on attention weights and outputs,
  and on block outputs — matching the reference's FP8 publication points;
- accumulation is f32 in our ports (the reference uses tensor-core f16);
  see docs/NETWORK.md for the exact trade and rationale.
