# The network

`src/nr_graph.cpp` records the whole thing into one command buffer. This file is what that code means.

## Inputs and outputs

The input features are f32 `[fullHeight * fullWidth][16]`, listed below. The output (the "head") is f32
`[fullHeight * fullWidth][4]`: an RGB residual and a temporal-blend logit.

The 16 input lanes, in order (`shaders/preprocess.comp`, `demo/shaders/nr_preprocess.comp`):

| lane | contents |
| --- | --- |
| 0-2 | three Gaussian lanes, Box-Muller from a hash of the *padded* pixel coordinate and a per-frame seed |
| 3 | constant 1 |
| 4-6 | the display proxy, centred: `f16((f16(c) - 0.5) * 0.125)` per channel, `c` an sRGB-encoded code value in 0..1 |
| 7-9 | the same for the reprojected previous output; a copy of lanes 4-6 when there is no history |
| 10 | style id / 128 |
| 11 | local tone |
| 12-14 | the structure / skin / auto-mask conditioning triple |
| 15 | 0 |

Only the valid rectangle carries image data. Outside it the sampling coordinate is **mirrored**
(`2 * valid - x - 2`, no edge repeat) while the noise hash still uses the padded coordinate, so the padding is a
reflected copy of the image with its own noise.

The network is resolution-preserving. It re-renders a tone-mapped proxy of the frame, generating detail from the
three injected Gaussian lanes under the tone, structure, skin and style conditioning; it is given noise and does
not remove any. The upsampling in the graph is internal (the decoder), not a change of output resolution.

## Geometry: levels and the padded field

`Geometry::fromValid` computes, from the requested (valid) size:

```
level[k] = alignUp(ceil(level[k-1] / 2), 4),  level[-1] = the padded field,  k = 0..5
```

Block 0 and block 70 run on the field itself; level 0 is half of it; the ViT works on level 5.

### The padded field

Every halving must be exact for the decoder to land back on the encoder's sizes, and the decoder's level-0
upsample emits **whole 8-pixel windows** and is then cropped. The field is therefore aligned to a power of two
that covers all the size reductions the graph makes on that axis:

```
reductions(v):  size = v
                for level in 0..5:
                    half = alignUp(ceil(size / 2), 4)
                    if half < size:                 reductions += 1
                    if level == 0 and half % 8:     reductions += 1     # the level-0 upsample + crop
                    size = half

align = 1 << reductions(valid)
field = max(320, alignUp(valid, align))
if fieldW % (4 * alignW) == 0 and fieldH % (4 * alignH) == 0:  fieldW += alignW
```

The last line has no reason that is visible anywhere, and it still has to be reproduced: the field size decides
the window grid, and therefore the result inside the valid rectangle too. A dimension that stops shrinking (a level of 4 stays 4)
makes the count smaller than six, which is why small inputs get fields that are not multiples of 64.

Two invariants follow and the rest of the graph relies on them:

* **level 0 is a whole number of 8-pixel windows**, so the decoder's whole-window upsample never has to crop and
  the code can upsample straight to `level[0]`. The extra reduction is exactly what buys this. The one exception
  is the `max(320, ...)` floor, which can raise the field past the alignment it was computed for: at widths 1-16
  and 25-32 level 0 comes out at 164, and there the network runs its level-0 decoder stage on `alignUp(164, 8)`
  and crops in the last block. This port does not implement that crop and
  I made `Geometry::fromValid` refuse those sizes instead of differing silently. The practical limit is 33 pixels on
  each axis;
* every level is a multiple of 4, so the 4x4 tiles the native layout uses (and the 2x2 pools) never straddle an
  edge, and out-of-field regions are always whole tiles.

Examples (all verified against native captures):

| valid | field | level 0..5 |
| --- | --- | --- |
| 512x512 | 576x512 | 288x256, 144x128, 72x64, 36x32, 20x16, 12x8 |
| 768x768 | 832x768 | 416x384 ... 16x12 |
| 644x768 | 768x768 | 384x384 ... 12x12 |
| 1920x1080 | 1920x1152 | 960x576 ... 32x20 |
| 3840x2160 | 3840x2176 | 1920x1088 ... 60x36 |

## A block

Blocks 0-4, 66-70 have 32 channels; the others widen towards the bottleneck. One block, in publication order:

```
    state x  (E4M3 [tokens][C])
      |
      +-- FFN(x) ------------------------------+
      |                                        v
      +--> x * ffnScale ---------------------> (+) --> y   (f16; E4M3 for the QKV input)
                                                       |
      +-- Proj(WindowAttention(QKV(y))) -------+       |
      |                                        v       |
      +--> y * attnScale --------------------> (+) --> out (E4M3 [tokens][C])
```

`ffnScale` and `attnScale` are learned per-channel f16 vectors. The skip is *seeded into the accumulator* (it is
the C operand of the first MMA of the chain), not added afterwards, which is why the FP8 GEMM has a residual
prologue rather than an epilogue, because the rounding differs.

One asymmetry matters: the 32-channel blocks add the **raw f16** FFN result as the attention skip, the wider
blocks add its **E4M3 publication**. Both are forced; they are visible in the boundary fixtures.

### FFN by width

| C | FFN |
| --- | --- |
| 32 | dense `32 -> 128 -> 32`, SiLU after the first layer |
| 64 / 128 / 256 | `C/32` parallel paths, each `C -> 128 -> 32` with SiLU after the first layer, concatenated to C, then one `C -> C` layer (W3) that also carries the skip. They are called experts in the code after the weight naming, but there is no router: every path runs on every token, so this is a grouped bottleneck FFN, not a mixture of experts |
| 512 | one `512 -> 512` layer split into 8 branches of 64, each `64 -> 256 -> 64` with SiLU after the 256-wide middle **only**, concatenated to 512, then one `512 -> 512` contraction carrying the skip |
| 1024 (ViT) | dense `1024 -> 4096 -> 1024`, SiLU after the first layer |

Every inter-layer boundary is an E4M3 publication.

### Attention

Per head (32 channels each; `heads = C / 32`, 32 heads in the ViT):

1. **cosine normalization**: `q /= |q|`, `k /= |k|` in f16 with the network's reduction order, then
   `q *= scale` with a learned per-head f32; `v` is only quantized. The result is E4M3. This is scaled cosine
   attention: the logit becomes a bounded cosine similarity times a learned temperature instead of a dot product
   that grows with the norms, which is what makes an f16 accumulator and a 3-bit-mantissa weight grid viable at
   all - and it is also why there is no max subtraction in the softmax below.
2. **S = q kᵀ + prior**, the learned 64x64 per-head bias as the MMA's C operand.
3. **exponential**: a bit trick on the half, not a transcendental:
   `x = clamp(f16(0.044921875 s + 1.30078125), 1.03125, 1.5693359375)`, then reinterpret
   `(bits(x) << 5) ^ 0x8000` as a half. That is `2^(1.4375 s - 5.375)` with a linear mantissa, i.e. `exp(s)` up
   to a constant factor that the softmax divides out, with `s` effectively clamped to about `[-6, +6]`.
4. **softmax** in a fixed pairwise order (`numerics.md`), reciprocal in f16, weights published as E4M3.
5. **O = P V**, published as E4M3, then the `C -> C` projection with the scaled skip.

The window is 8x8 tokens. The ViT attends over *all* level-5 tokens at once, and differs in four ways that are
easy to miss and impossible to guess:

| | window blocks | ViT blocks |
| --- | --- | --- |
| prior | learned 64x64 per head | none |
| Q scale | `q * norm * learned` | `q * norm * sqrt(32) * learned`, three separate half multiplies |
| square sum | `fma(v, v, f16(high^2))` in half | low square in f32, high square rounded to half first, sum rounded once |
| softmax | weights normalized before `P V` | weights *unnormalized*; the reciprocal multiplies the value accumulator afterwards, and the denominator subtracts `expWeight(0) * paddedTokens - tokens` for the padding rows |

The exponential is the same bit trick with different constants:
`x = clamp(f16(0.08953947 s + 1.70936143), 1.439453125, 1.9775390625)`, then `(bits(x) << 4) + 0x4000`, which is
`2^(1.4326 s - 3.6502)`, with `s` effectively clamped to about `[-3, +3]` rather than `[-6, +6]`.

Tokens are padded to a multiple of 64; the padding rows have zero K/V and get the same `exp(0)` weight, which
the correction above removes from the denominator.

### Window phases

Shifted windows: each block places the window grid at an origin offset, cycling

```
phase:   0        1        2        3
origin: (0,0)   (-4,-4)  (-4,0)   (0,-4)
```

Four views, not the two that shifted-window attention usually alternates: the diagonal shift is split into its
two axes, so a window boundary in x and one in y are each crossed by a different block.

**The phase advances once per block at a resolution level, and a decoder stage continues the count its encoder
stage left.** Level 2 has six encoder blocks (phases 0,1,2,3,0,1) so its six decoder blocks start at 2; the
full-resolution level has block 0 (phase 0) and block 70 (phase 1). `Graph::takeWindowPhase` is that counter, and
it replaces what would otherwise be a table of unexplained offsets.

Windows whose origin is negative, or that run off the right or bottom edge, are not clipped: the grid is
`ceil((size + shift) / 8)` windows and the tokens outside the field are **zero vectors**. They contribute a zero
key and a zero value, but their score is still `prior + 0` and their exponential still enters the softmax
denominator. That is not an approximation. Masking them out instead changes the result.

## Transitions

| | |
| --- | --- |
| encoder | 2x2 box pool of the last block's raw f16 output (`((a+b)+(c+d)) * 0.25`, every step a half), E4M3, then a `C -> 2C` GEMM |
| decoder | `2C -> C` GEMM at the low level (f16 out), nearest 2x upsample, `+ skip * transitionScale` as one f16 FMA, E4M3 |
| 30 -> 31 | pool, then `512 -> 1024` into the ViT |
| 38 -> 39 | `1024 -> 512` (f16), upsample, `+ skip512 * scale` |
| 69 -> 70 | 2x upsample of block 69 `* inputScale`, `+ block-0 output * adapterScale`, one f16 multiply and one FMA |

The pooled tensor is never a network output on its own: the fixtures' `transition-N-M` captures are the *next*
stage's input, after the channel-doubling GEMM.

## The head and what it means

Block 70's raw f16 output goes through a `32 -> 4` f16 GEMM into f32. Channels 0-2 are an RGB residual on the
proxy and channel 3 is a temporal blend logit:

```
neural  = clamp(proxy + rgb / 4, 0, 1)              # in the proxy's sRGB code space
weight  = clamp(sigmoid(logit) * blendScale, 0, 1)  # blendScale is a learned f16 (block70.layer0.blend_scale)
display = lerp(neural, reprojected_history, weight)
```

`parity` checks the first line against the native RGB output; the demo does the rest (see [frame.md](frame.md)).

## Weight tensors consumed

153 tensor records; the graph reads 145 of them. The eight unread ones are the ViT's 2-byte
`blockNN.layer3.layer` scalars (values from 5e-5 to -558.5, so not no-op scales). Every ViT block boundary is
exact without them. `block70.layer0.blend_scale` is read by the demo, not by the graph.
