# The network

The graph this project executes is the OpenDLSS-NR network family — the
71-block Swin/ViT generative rendering U-net of the reference repository —
with block semantics, lane packing, padding rules and head composition ported
rule for rule. This file is the implementation-level summary; the reference's
`docs/network.md` remains the specification ground truth.

What we preserve exactly:
- the 16-lane input packing and mirrored padding,
- the padded-field geometry (see `docs/ARCHITECTURE.md` — verified against the
  reference's published table),
- block structure: `y = x·ffnScale + FFN(x)`, `out = y·attnScale + Proj(Attn(QKV(y)))`,
- fp16 storage, E4M3 publications, the bit-trick attention exponential,
  cosine attention with a learned per-head scale and 64×64 prior,
- the 4-phase shifted-window cycle (0,0) (-4,-4) (-4,0) (0,-4) advancing per
  block per level, decoder continuing the encoder's count,
- transitions: half-step 2×2 box pool → E4M3 → channel GEMM (encoder);
  down-GEMM → nearest 2× upsample → one-f16-FMA skip (decoder);
  the 69→70 adapter (block-0 output · adapterScale),
- the head: `neural = clamp(proxy + rgb/4)`, `weight = clamp(sigmoid(logit)·blendScale)`,
  `display = lerp(neural, history, weight)`.

What we adapt, deliberately and openly:
- **Accumulation**: the reference accumulates FP8 GEMMs in f16 on NVIDIA
  tensor cores; we accumulate in f32 (portable across Metal/Vulkan/DirectML)
  while keeping every publication on the E4M3 grid and every storage on the
  f16 grid. Bit-exactness against NVIDIA captures is neither claimed nor
  possible off that hardware; self-consistency across our four backends is
  the contract (CPU parity golden, shared bit math everywhere).
- **Schedule as data**: the block ladder comes from the manifest
  (`schedule[]`), so the executor runs any depth/channel layout — the 7-block
  demo today, the full 71-block network when a manifest of that shape is
  supplied.
- **MetalFX owns scaling**: the reference network is resolution-preserving;
  the resolution change (the DLSS-SR role) is MetalFX's job on Apple Silicon.

## The 16 input lanes

| Lane | Contents |
| --- | --- |
| 0–2 | three Gaussian lanes, Box-Muller from a hash of the *padded* pixel coordinate + per-frame seed |
| 3 | constant 1 |
| 4–6 | the display proxy, centred: `f16((f16(c) − 0.5) · 0.125)` |
| 7–9 | the same for the reprojected previous output; copy of 4–6 with no history |
| 10 | style id / 128 |
| 11 | local tone |
| 12–14 | structure / skin / auto-mask conditioning |
| 15 | 0 |

Outside the valid rectangle the image sample mirrors (`2·valid − x − 2`) while
the noise hash keeps using the padded coordinate — the padding is a reflected
copy with its own noise, per the reference.

## Window attention (one head = 32 channels)

1. cosine normalization: `q /= |q|`, `k /= |k|` (f16 grid), then `q *= scale`
   with a learned per-head scalar; `v` is only quantized. All three published E4M3.
2. `S = q·kᵀ + prior` — the learned 64×64 per-head bias.
3. softmax through the bit-trick exponential `2^(1.4375 s − 5.375)` with the
   clamp window (ViT variant: `2^(1.4326 s − 3.6502)`, window ±3), reciprocal
   in f16, weights published E4M3.
4. `O = P·V` published E4M3, then the C→C projection.

Out-of-field window tokens are zero vectors whose prior still enters the
softmax denominator — masking them would change the result (reference semantics).

## The ViT blocks

Global attention over all tokens of the bottom level: no prior, unnormalized
softmax weights (the reciprocal multiplies the value accumulator), q scaled by
three separate half multiplies (`norm · √32 · learned`), token counts padded to
a multiple of 64 with the `exp(0)·(padded − real)` denominator correction.

## Transitions

| Direction | Operation |
| --- | --- |
| encoder (deeper) | 2×2 box pool of the raw output (`((a+b)+(c+d))·0.25` in half steps) → E4M3 → C→2C GEMM |
| decoder (shallower) | 2C→C GEMM at the deep level (f16 out) → nearest 2× upsample → `+ skip · scale` as one f16 FMA → E4M3 |
| into / out of ViT | `vitin.up` (C→2C) / `vitout.down` (2C→C, f16 out) |
| L0 → field | down-GEMM → upsample → `+ block0 · adapterScale` (the 69→70 adapter) |

## What the network is for

It re-renders the frame the engine/video pipeline already produced: generating
detail from injected noise under the conditioning scalars, adjusting tone and
structure, and emitting the temporal blend logit that decides how much
reprojected history to keep. It is a *rendering* network, not a classifier —
the demo model's near-identity weights produce a stable mild residual, which
is exactly what makes it safe to run everywhere before trained weights exist.
