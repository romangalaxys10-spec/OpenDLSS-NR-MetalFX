# The NR network on Vulkan: design notes

These notes are the mental model behind `src/`, `shaders/`, `scripts/ptx/` and `demo/`: what the network
computes, why each mechanism has to be the way it is, and which parts are forced by the network, by Vulkan, or
by this implementation. The top-level `README.md` covers building and running.

Two words are used consistently: **native** is the implementation this one is matched against, and **the CPU
reference** is `src/reference.cpp`, this repository's own port of the arithmetic. Where a comment in the code
says a choice is *specified*, it means that choice is fixed by the former and is not free.

The sections below the index answer the questions that span files: what the representations and coordinate
spaces are, which invariants the code relies on, and which parts of the design are forced versus chosen.

* [network.md](network.md) is the graph: inputs and outputs, the resolution pyramid, what one block is, the
  residual structure and the data flow through it, the padded field and the window phases.
* [numerics.md](numerics.md) is the exactness contract: where values are published, the tensor-core
  arithmetic, the half roundings, the softmax, and what "bit-exact" is measured to mean.
* [weights.md](weights.md) is the model directory, the packed fragment layouts, and the re-layouts the host
  performs.
* [execution.md](execution.md) is Vulkan resources, the kernel set and the fusions, **synchronization and
  launch order** (barrier-free chaining), initialization and teardown, ownership and lifetimes, failure and
  fallback behaviour, the performance-critical paths.
* [frame.md](frame.md) is the demo's frame: rendered image to input features, the head to pixels, motion
  vectors, and the temporal loop (what carries between frames).

## One page

The network is a **U-net of shifted-window transformer blocks** with a global ViT at the bottom: 71 blocks over
six pooling levels, everything in FP8 (E4M3) activations with FP16 accumulation, 141 MiB of weights.

It takes one rendered frame (a low dynamic range *proxy* of it, three lanes of Gaussian noise, the previous
frame's output reprojected, and five conditioning scalars) and produces four f32 channels per pixel: an RGB
residual and one temporal-blend logit. The displayed image is `clamp(proxy + rgb / 4, 0, 1)` blended with the
reprojected history by `sigmoid(logit)`. It is a generative neural renderer, not an upscaler: input and
output are the same resolution.

```
 proxy + noise + history (16 f32)                                              RGBA f32
              |                                                                    ^
        [block 0, 32ch, full res] ------------------------ skip ----------> [block 70, 32ch] -> head
              | 2x2 pool                                                            ^ 2x upsample
        [blocks 1-4, 32ch, L0] --------------- skip ---------------> [blocks 66-69, 32ch, L0]
              | pool + 32->64                                                       ^ 64->32 + upsample
        [blocks 5-8, 64ch, L1] --------------- skip ---------------> [blocks 62-65, 64ch, L1]
              | pool + 64->128                                                      ^
        [blocks 9-14, 128ch, L2] -------------- skip --------------> [blocks 56-61, 128ch, L2]
              | pool + 128->256                                                     ^
        [blocks 15-22, 256ch, L3] ------------- skip --------------> [blocks 48-55, 256ch, L3]
              | pool + 256->512                                                     ^
        [blocks 23-30, 512ch, L4] ------------- skip --------------> [blocks 39-47, 512ch, L4]
              | pool + 512->1024                                                    ^ 1024->512 + upsample
        [blocks 31-38, 1024ch, L5: global attention over all tokens]  ---------------
```

Three facts explain most of the code:

1. **Every value that crosses a kernel boundary is an E4M3 byte or an IEEE half**, and the network's identity is
   the *sequence of those publications*, not the real-valued function. Matching NVIDIA's output therefore means
   matching an ordering of roundings, not an error bound. See [numerics.md](numerics.md).
2. **The unit of work is an 8x8 window at one resolution level**, and a block is FFN -> QKV -> window attention
   -> projection with two scaled skips. Everything about the data layout follows from wanting that whole chain to
   stay in registers. See [network.md](network.md) and [execution.md](execution.md).
3. **The image is padded to a "field" whose size the network's own shape rules dictate**, so that six exact
   halvings and the decoder's whole-window upsample all fit. The padding is part of the arithmetic: it decides
   which tokens exist. See [network.md](network.md#the-padded-field).

## Representations, units and coordinate spaces

| thing | representation |
| --- | --- |
| activation tensor | `[rows][channels]` row-major, rows padded to a multiple of 64; E4M3 bytes, or IEEE halves for the raw/residual tensors (native stores tokens in 4x4 pixel tiles of 16 instead) |
| token index | `y * width + x` over the **padded field**, y down, the valid rectangle at its top-left |
| token order in a window | queries in natural order (row-major in the 8x8 window), keys and values in **physical** order (4x4-tiled) - which is also the softmax's reduction order |
| ViT tokens | one dimension, row-major over level 5, padded up to a multiple of 64 |
| window origin | `(-shiftX, -shiftY)` from a multiple of 8, with shifts of 0 or 4 |
| weights after re-layout | `[K/batchK][batchK/32][N][32]` E4M3 (native stores MMA fragments) |
| per-channel skip scales | one f16 each; per-head attention scale: one f32 |
| proxy and head RGB | sRGB **code values** in 0..1; the network sees `(code - 0.5) / 8` |
| head channels | 0-2 an RGB residual added at 1/4 in code space, 3 a temporal-blend logit |
| motion vector | uv of the render target, current -> previous, y down, plus a flag: whether the previous position is on screen (whether there is a history); the background moves with the camera's rotation |

## Invariants

| invariant | why it holds | what breaks without it |
| --- | --- | --- |
| activation rows are padded to a multiple of 64 and zero-filled at creation | `Activation::allocRows` | a 16-row cooperative-matrix load near the end of a tensor reads outside the allocation, or reads garbage |
| level 0 is a whole number of 8-pixel windows | the field rule's extra reduction; sizes where the 320 floor breaks it are refused | the decoder's level-0 stage would have to run on whole windows and crop, which is not implemented |
| every level dimension is a multiple of 4 | `alignUp(ceil(x / 2), 4)` | 4x4 tiles and 2x2 pools would straddle the field edge; out-of-field regions would not be whole tiles |
| window origins are even | multiples of 8 minus a shift of 0 or 4 | the fused 2x2 pool's output index `row / 2` stops being exact and half-resolution pixels are written twice or not at all |
| out-of-field tokens are zero vectors whose exponential still enters the softmax denominator | the tensor load clamps to zero; the publication masks them | edge windows differ from native - masking them out is a different function |
| every value crossing a kernel boundary is E4M3 or f16, rounded f32 -> f16 -> E4M3 in that order | `numerics.md` | the output diverges from native even though the real-valued function is unchanged |
| NaN publishes as +0 and the sign of a zero survives | `common.glsl` | the E4M3 NaN code poisons the MMA that reads it; `-0` differences break byte equality with the captures |
| the chained activation permutation stays inside each 16-product group | it rotates bits 1-3 of the index, leaving bit 4 alone | folding it into the weight rows would change the F13 grouping, i.e. the arithmetic |
| a chained launch is always a PTX launch, and only ever waits on a launch recorded earlier | `Graph::Routes::chain`, `check()`s in the GLSL kernels, `Kernels::checkChainOrder` at every recording | a GLSL kernel neither waits nor signals: the GPU hangs, or a producer with no barrier races its consumer |
| the GPU issues a launch's workgroups before any of a later launch on the same queue | NVIDIA's behaviour, not a Vulkan guarantee (`execution.md`); the chained waits' watchdog turns a violation into a reported, failed frame | a consumer could hold the SMs its producer needs; without the watchdog, a hang |
| a counter index never exceeds 512 per sync region | checked against the field height; chaining falls back to barriers above it | indices run into the neighbouring region |
| the sync counters are zero at the top of every recording | `Kernels::resetSync` | a consumer sees a stale count and starts early |
| re-recording the graph allocates nothing and a label collides with nothing | `Graph::allocate` keys by label and shape, `usedThisRecord_` | two logical tensors would silently alias |
| the split-K scratch is sized once | its device address is baked into already-recorded launches | recorded launches would write through a dangling address |
| history images alternate by frame parity | frame N reads `history[N & 1]`, writes `history[1 - (N & 1)]` | the temporal loop reads the frame it is writing |
| a pixel without a history gets the no-history input and blend weight 0 | the motion unpack's flag, read by the preprocess and the composite | the previous frame's colour at the same pixel - another surface's - stands in for a history that does not exist |

## What has no derivation

The rest of this documentation explains mechanisms. These have none: they are values and orderings the
specification fixes, and the only account of them I can give is that they are what they are. Some have a derived mechanism
and an opaque value (the exponential is `2^(1.4375 s - 5.375)` by construction; `0.044921875` is not derivable);
others are architecture.

**Constants.** The cubic SiLU's coefficients
(`-0.055908203125`, `0.447265625`, `0.89453125`) and its clamp to +/-4. Both exponential approximations: the
window's affine `0.044921875 s + 1.30078125` clamped to `[1.03125, 1.5693359375]` with a 5-bit shift and
`^ 0x8000`, and the ViT's `0.08953946828842163 s + 1.7093614339828491` clamped to `[1.439453125, 1.9775390625]`
with a 4-bit shift and `+ 0x4000`. The extra `sqrt(32)` on the ViT's queries. The display proxy's shoulder
(knee at 0.75, `exp(-5.770780 * (v - 0.75))`) and the head's quarter-scale composition
`clamp(proxy + rgb / 4, 0, 1)`. The style id divided by 128.

**Orderings** - free in exact arithmetic, fixed here, and in f16 they disagree. The softmax reduction tree
(`b[g][j] = e[g+16j] + e[g+16j+8]`, `t[g] = ((b0+b1)+b2)+b3`, then the even and odd halves summed separately and
added). The cosine norm's `fma(v, v, f16(high^2))` followed by a stride 8/4/2/1 tree. The 2x2 pool's
`((a+b)+(c+d)) * 0.25`. The K order of every GEMM chain, and the partition splits within it
(ViT contract 4096/1024, qkv 1024/512, projection 1024/256, expand unsplit) whose partial sums are then added in
partition order.

**Layouts.** The within-32 activation permutation (a rotation of index bits 1-3). The 4x4-tiled
"physical" token order that keys, values and the attention prior's key axis live in. Which tensor slot of a
block holds which matrix, the two 16-byte zero pads inside each block record, and the ViT's qkv tensor putting
its per-head scales *before* the weights.

**Structure.** The block count per stage (4, 4, 6, 8, 8 encoder, 8 ViT, mirrored back) and the channel widths.
Which stages use a dense FFN, which use `C/32` experts, and that the 512 stage instead uses eight 64-wide
branches with SiLU on the 256-wide middle only. The four window views and their order, and that a decoder stage
continues its encoder stage's phase count. That the ViT has no attention prior. That the 32-channel blocks take
the raw f16 FFN output as the attention skip while the wider ones take its E4M3 publication. That the history is
stored **truncated** to the half grid rather than rounded.

**Semantics.** The field rule's final `+= alignWidth` step and its floor of 320. The
sixteen input feature lanes, their order, and the mirrored (not clamped, not repeated) sampling outside the
valid rectangle while the noise still hashes the padded coordinate. That out-of-field tokens behave as zero
vectors whose `exp(prior)` still enters the softmax denominator instead of being masked out.

## What is fundamental and what is ours

| kind | examples |
| --- | --- |
| **semantic requirement** (changing it changes the output) | everything in the section above, plus every publication point and its rounding |
| **Vulkan / API requirement** | rows padded to 64 for cooperative-matrix loads; `requiredSubgroupSize = 32`; buffer device addresses for the PTX launches; a compute-to-compute barrier between unchained dispatches; one descriptor layout because the kernels share a pipeline layout |
| **implementation choice** | activations row-major instead of 4x4-tiled; which kernel computes a block; weights re-laid out on the host at load time; two descriptor pools; the graph recorded into one command buffer |
| **optimization** | the fusions (`F_PRE`, `F_POOL`, `F_UPRES`, `F_POST`, `F_HEAD`, the deferred projection); barrier-free chaining; persistent grids; operands staged in shared memory; split-K; the SiLU lookup table |
| **inherited, no visible reason** | the `+ alignWidth` step in the field rule; the two 16-byte zero pads inside each block tensor; the eight unread 2-byte ViT tensors; the NaN-weight guard that never fires |
