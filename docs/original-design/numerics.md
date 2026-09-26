# Numerics: the exactness contract

The network's identity is a sequence of roundings. Two implementations that agree on the real-valued function
but disagree on where values are published produce visibly different frames, so every publication point below
is a specification, not a detail.

## The two grids

**E4M3** (`e4m3fn`: 4-bit exponent, 3-bit mantissa, max 448, no infinities) carries every activation that
crosses a kernel boundary and every GEMM operand. **IEEE binary16** carries everything else: intermediates
inside a kernel, accumulators, and the residual/skip tensors of the 32-channel blocks. The conversions live in
`common.glsl` (`e4m3CodeFromF16Bits` / `e4m3Hw`, `roundF16`) and `numeric.h` (`e4m3FromF16Bits`, `f16Bits`).

Publication rules, all three implementations (GLSL, PTX, CPU reference) agreeing:

* a value is rounded to half first, then to E4M3 from the half, never f32 straight to E4M3;
* RNE at both steps; saturation at ±448 (code 0x7e), no infinity code;
* **NaN publishes as +0**, not as the E4M3 NaN code 0x7f. This is load-bearing: a zero row's cosine
  normalization is `0 * inf`, which happens for every out-of-field token, and 0x7f would poison the MMA that
  reads it. The attention result is identical to native's, which the edge windows in the fixtures prove;
* **the sign of a zero survives** (`-0` is code 0x80). The hardware conversion keeps it and so do the native
  kernels; canonicalizing it to `+0` costs nothing numerically but breaks byte equality with the captures.

`roundF16` is written on the bit pattern on purpose. The NVIDIA GLSL compiler treats `float(float16_t(x))` as a
no-op and silently removes the intermediate publication; the first parity bug of this port was exactly that,
inside the SiLU polynomial.

## Tensor-core arithmetic

Every FP8 GEMM is a chain of `16x16x32` E4M3 x E4M3 -> f16 cooperative-matrix multiplies in K order, which is
the hardware's `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`, the same instruction the native kernels
use. A k32 step is internally **two groups of 16 products**, each computed as a fixed-point dot product:

```
E = max(exp(accumulator), max over the 16 pairs of exp(a) + exp(b))     # per group
sum = trunc(acc * 2^(13 - E)) + sum_i trunc(a_i b_i * 2^(13 - E))       # F13: 13 fractional bits, truncated
result = roundF16(sum * 2^E)                                             # the group's accumulator
```

`ref::adaFp8Fdpa16` is that, and `ref::adaF16Fdpa8` is its f16 counterpart (8 products, F24, exact integer
accumulation) for the two f16 GEMMs. Consequences that the code depends on:

* the **K order is part of the result**: reassociating the chain changes bits;
* the **residual seeds the accumulator** (the C operand of the first step), it is not added at the end;
* a `partition` (a split-K mode) breaks the chain: each partition accumulates from zero and the
  partial sums are added in f16, in partition order. The values are not a tuning choice. The native ViT
  launches carry them as the `z` dimension of the grid: FFN contract 4096/1024, qkv 1024/512, projection
  1024/256, FFN expand unsplit. Split-K reproduces exactly that structure in separate workgroups.

## Where the half roundings are

| step | rounding |
| --- | --- |
| SiLU | `clamp(x, -4, 4)`, `p = fma(-0.055908203125, |x|, 0.447265625)`, `p = fma(x, p, 0.89453125)`, `x * p`. Five half operations, each rounded once. A 65536-entry f16->f16 table is equivalent and is what the GLSL kernels use |
| cosine norm (window) | `r[c] = fma(v[c], v[c], f16(v[c+16]^2))` for 16 pairs, then a stride 8/4/2/1 tree, every level a half; `inversesqrt` in f32 then to half |
| cosine norm (ViT) | the same tree, but `r[c] = f16(f32(v[c])^2 + f32(f16(v[c+16]^2)))`: the low square stays in f32, the high one is rounded first |
| scaled skip | the product of two halves is exact in f32 and rounds once, so `f16(a) * f16(s)` is the same publication as the f32 spelling |
| 2x2 pool | `((a + b) + (c + d)) * 0.25`, four half operations |
| post blend | `f16(up * sA)` then `fma(adapter, sB, that)`: the second product is contracted into the add |
| upsample merge | `fma(skip, scale, projection)`, one FMA |

## The softmax

Scores come out of the MMA in the accumulator's element order; the reduction follows the native one exactly:

```
e[k]                  = expWeight(S[k])                    # 64 keys, physical (tiled) order
b[g][j]               = e[g + 16j] + e[g + 16j + 8]        # g = 0..15, j = 0..3
t[g]                  = ((b[g][0] + b[g][1]) + b[g][2]) + b[g][3]
total                 = (((t0 + t2) + t4) + t6) + (((t1 + t3) + t5) + t7)
weight[k]             = E4M3( e[k] * f16(1 / total) )
```

Every `+` is a half add. The pairings `(k, k+8)` and the even/odd split are lane-local in the accumulator
layout, which is why the kernels can do this in registers with two shuffles.

`expWeight` is the bit trick described in [network.md](network.md#attention). Note what it is *not*: there is no
max subtraction. The clamp on the affine argument is the only thing keeping the exponentials in range, which is
also why the network can get away with an f16 denominator.

The ViT's softmax has the same reduction tree but normalizes at the other end: its E4M3 weights are the raw
exponentials, the running total is accumulated per 64-key block (`total = total + blockSum`, in block order),
the padding rows' `expWeight(0) * padding` is subtracted once at the end, and the reciprocal multiplies the
value accumulator instead of the weights. Its exponential constants and Q scaling differ too
([network.md](network.md#attention)).

## What "bit-exact" is measured to mean

`dlss5vk parity` (and its browser twin) runs the checks a fixture declares and names the equality each one proved:

| verdict | meaning | gate |
| --- | --- | --- |
| bit-exact | every byte (E4M3 boundaries) or every bit pattern (the f32 head, the composed RGB halves) is identical: `+0` is not `-0`, a NaN equals only the same NaN | pass |
| equal only up to the sign of zero | numerically equivalent, not the same bytes | **fail**, reported apart |
| within one code | the composed image against an old 8-bit capture, whose own rounding is not known exactly | pass, reported apart |
| mismatch | anything else | fail |

A fixture declares its checks (`boundaries`, `head`, `output`), and the checker refuses it before anything runs if
a declared check has no reference, a reference is missing, short or names nothing in the graph, or a reference is
carried that no declared check uses. A boundary fixture must account for all 75 comparable boundaries (blocks 0-69
and the five encoder transitions; block 70 feeds the head on chip): each has a reference or is declared omitted
with a reason. A fixture cut short therefore fails; it cannot pass as a smaller suite.

The head and the composed image are compared on the **production schedule** (no captures, counter chaining as
configured), resubmitted `--repeat` times on the same buffers; the same graph with a barrier after every launch
must give the same head. Boundaries come from a separate **instrumented schedule** that adds a copy and two
barriers at every boundary and materializes the deferred projections; its head must equal production's too. The
captures localize a difference; they do not stand in for the schedule that ships.

At 512x512 the boundary fixture is 75 boundaries, 57,704,448 E4M3 bytes, all bit-exact, signed zeros included,
and the old 8-bit capture of native's image is matched within one code (786,180 channels exact, 252 one code off).
The default PTX route, the GLSL reference route (`DLSS5VK_UNFUSED=1`) and every other route switch give the same
bytes. Eleven further fixtures from 644x768 to 3840x2160 are bit-exact on the composed RGB halves, and the four at
1024x768, 1920x1080, 2560x1440 and 3840x2160 on all 75 boundaries as well.

Three independent implementations back that up, but not everywhere equally. `src/reference.cpp` is a CPU port of
the arithmetic, and `dlss5vk verify` runs block 0 through it kernel by kernel, feeding each check the GPU's own
inputs so a mismatch names one kernel. It covers the dense 32-channel block and the f16 GEMM. The expert FFN,
the 512 split block and **the ViT have no CPU reference**: they are checked only by their two GPU routes agreeing
with the native captures at every block boundary. That is still ground truth rather than a shared guess, but a
ViT failure cannot be bisected the way a block-0 failure can.

`verify` compares values, not bits (a NaN equals a NaN, `+0` equals `-0`): it exists to name the kernel where a
difference starts. `parity` is the gate.

## Two approximations that are matched behaviourally, not instruction for instruction

* **The reciprocal square root and the reciprocal.** Three spellings are in play: the cosine norm is
  `inversesqrt` in GLSL, `rsqrt.approx.f32` in the PTX, `rsqrt.approx.ftz.f32` in native; the softmax reciprocal
  is `1.0 / x` in GLSL, `rcp.rn.f32` (correctly rounded) in the PTX, `rcp.approx.ftz.f32` in native. The `.ftz`
  cannot matter here: both inputs are f16 values widened to f32, so they are never subnormal, and both outputs
  (a norm of a small sum, a reciprocal of a total that is at least ~1) are far from the subnormal range. What
  remains is `approx` (about one f32 ulp) versus correctly rounded, and that is absorbed: everything these feed
  is published through the E4M3 grid, whose steps are ~2^-3 relative. The fixtures confirm it at every
  resolution, but this is the one arithmetic step I match by outcome rather than by instruction.
* The E4M3 NaN code. Native's bare `cvt.rn.satfinite.e4m3x2` emits it for the `0 * inf` of a zero row; this port
  substitutes `+0` at the same point. The intermediate byte is not captured anywhere, and the attention output
  it feeds is identical.

These are the only two places where this implementation matches an *effect* rather than an instruction. If a
future driver or GPU breaks parity, check them first.

## Weights that never occur

No weight byte in the shipped model is the E4M3 NaN code. `dlss5vk parity` prints
`NaN weight codes replaced: 0` at every resolution. So the substitution in `Model::fp8MatrixBytes` guards a case
that has never happened, and the value it picks (0) follows the reference decoder rather than hardware
behaviour. That is dead code, and it stays in.
