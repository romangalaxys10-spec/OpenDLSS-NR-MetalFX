# The NR network in the browser: a WebGPU port

A port of the Vulkan implementation in this repository to WebGPU, written to be read. The point is not that
the network runs in a tab; it is that everything the Vulkan version leans on (a tensor-core MMA, an FP8 type,
half precision, inline PTX, device-scoped atomics) is absent here, and every one of those substitutions had
to be made without changing a single published byte.

It does not change a single published byte. On an RTX 4070 SUPER:

```
=== self-test ===     SELFTEST PASS
=== parity nr512 ===  PARITY PASS: 75 bit-exact, 1 within tolerance, 0 failed
=== parity nr768 ===  PARITY PASS: 2 bit-exact, 0 within tolerance, 0 failed
```

Every stored block output (blocks 0-69; block 70 feeds the head directly) and every stage transition, compared
byte for byte against what native produced from the same input; then the head and the composed image, compared
bit for bit against what native produced from a recorded display proxy. The one tolerance is named as one: the
older 8-bit capture of native's image in `nr512`, matched within one code. Same fixtures, same fixture contract
and same verdicts as `dlss5vk parity`.

## What the browser does not have

| the Vulkan version uses | WebGPU offers | what this port does |
| --- | --- | --- |
| `VK_EXT_shader_float8`, E4M3 as a type | nothing | bytes in `u32` words, decoded arithmetically |
| `VK_KHR_cooperative_matrix`, a tensor-core MMA | nothing | the k32 step written out as fixed point |
| `float16_t` throughout | `shader-f16`, an optional feature | every rounding placed by hand on the f32 bit pattern |
| PTX, inline assembly, `VK_NV_cuda_kernel_launch` | nothing | WGSL only |
| barrier-free chaining through device atomics | one implicit barrier between dispatches | a dispatch per stage |
| 48–96 KB of shared memory | 16 KB of workgroup storage by default | 32 KB, requested from the adapter |
| 64-bit integers | none | the f16 step's fixed-point sum bounded to fit `i32` |
| `exp2` as an instruction you can trust | `exp2` specified to a few ULP | powers of two built from the exponent field |

The last row is the one that surprises people. WGSL gives its transcendentals an accuracy allowance, so a
result compared byte for byte cannot go through them; `pow2` in `shaders/numerics.wgsl` writes the exponent
field directly instead. For the same reason nothing here calls `fma`: WGSL does not say whether a multiply and
an add will be contracted, and the answer changes the last bit.

## Why exactness, and not "close enough"

The network's identity is a *sequence of publications*, not a real-valued function: every value that crosses a
kernel boundary is an E4M3 byte or an IEEE half, and matching native means matching that sequence of
roundings rather than an error bound. `docs/numerics.md` in the repository root makes the case in full.

It has a practical consequence for a port. An implementation can be visually perfect and still be a different
network, and the only way to know is to compare bytes against something trustworthy. So this port was built
from the bottom: nothing was written on top of the arithmetic until the arithmetic was shown to be right.

## Running it

The port itself has no build step: its sources are ES modules and WGSL text, served as they are, so what
runs in the browser is what is in the repository. The demo is the exception, and is bundled, because the
viewer it uses ships as a script that publishes its API on `window`.

```
npm install
node serve.mjs                      # then open the printed links
node demo/build.mjs                 # the demo's two bundles; the port itself needs no build
node tools/check.mjs                # every gate: the self-test and both parity fixtures
node tools/check_numerics.mjs       # the arithmetic, on the CPU, no GPU needed
node tools/headless.mjs selftest    # the arithmetic, both sides, in headless Chrome
node tools/headless.mjs demo        # the demo, reported back as a PNG
```

`NR_HEADED=1` opens a real window instead of running headless, for watching a run rather than gating on it.

Three directories live outside the repository because of their size, and are mounted by environment variable:

| variable | default | what it is |
| --- | --- | --- |
| `NR_WEIGHTS` | `../../models/nr` | the model directory: `manifest.json` and `model/stages/*` (141 MiB) |
| `NR_FIXTURES` | none | the parity fixtures; `tools/check.mjs` asks for `nr512` and `nr768` |
| `NR_SCENES` | none | the demo scenes: `view.json`, a `.glb` and `lighting.hdr` per scene |

## The three pages

**`/web/selftest.html`**: every scalar function the arithmetic is made of, run through the JavaScript oracle
(`src/numerics.js`) and through the WGSL (`shaders/numerics.wgsl`), both compared against
`web/fixtures/numerics.bin`: what the Vulkan implementation's own CPU reference produced for the same inputs.
Exhaustive over all 65 536 half bit patterns wherever the domain allows it. Neither side of the port is ever
validated only against the other.

**`/web/parity.html`**: the whole network on a recorded frame, with the resulting image beside the table. The
browser twin of `dlss5vk parity`. It takes `?fixture=`; a fixture declares its checks (`boundaries`, `head`,
`output`) and the page runs exactly those, or refuses the fixture before running anything if a reference is
missing, short, unknown or unused (the contract is in the repository's README). Two fixtures matter, because
neither is sufficient alone:

* `nr512` carries the input features and one file per stored block output. It gates the network block by
  block and stops at block 69, so of the full-resolution post block, the head and the composition it only sees
  an 8-bit image, which can be compared within one code and no better.
* `nr768` carries the display proxy the network was given, the head native produced from it, and the composed
  image. It gates exactly the part the other cannot reach, bit for bit, and nothing about where inside the
  network a difference began.

`tools/check.mjs` runs both, which is the only way either is worth quoting.

**`/demo/index.html`**: the demo. The viewer renders the scene with WebGL and the port runs the network on it
with WebGPU, so each frame crosses between the two APIs: the colour target and the velocity buffer are read back
into typed arrays, uploaded, run through the graph, and the composition kernel writes both the displayed image
and the next frame's history. A 1904×929 frame costs about 15 ms of readback and 465 ms of network.

## What the port cannot reproduce, and why

**The three noise lanes, in principle.** The input features carry three Box-Muller Gaussians, and native
evaluates them with the hardware's *approximate* `log2`, `cos` and `sin` rather than correctly rounded ones.
Those are not the same functions from one vendor to the next, so the last bits of those three lanes are a
property of the machine. This is why the boundary fixture carries recorded features rather than generating
them. On the card above they do reproduce — `nr768` generates its features from the recorded proxy and still
matches native's head bit for bit — but that is a measurement on one vendor, not a guarantee.

**The demo's proxy.** The network sees a display proxy of the rendered frame rather than the frame itself, and
the tone response that proxy is built from belongs to the viewer rendering the scene. It is a faithful proxy
of what is on screen, which is what the network reads, but it is not the one native was measured against.

## Five things the verification caught

Worth recording, because all five are the kind of thing a port gets silently wrong.

**The ViT's exponential is half arithmetic with half constants.** It is usually written
`fma(score, 0.08953946828842163, 1.7093614339828491)`, but those f32 spellings are not the numbers the
arithmetic sees: the operation is a *half* fused multiply-add, so the constants are `f16` of those, namely
0.08953857421875 and 1.708984375 (`0x2DBB`, `0x3ED6`). Using the f32 literals gives a different function.
Evaluating it in f32 rather than half turns out to be safe, and `shaders/numerics.wgsl` carries the argument
for why, but that had to be established rather than assumed.

**The E4M3 NaN code has two decodes in the Vulkan tree.** `shaders/common.glsl` reads `0x7f`/`0xff` as a
signed zero; `src/numeric.h` reads it as NaN. Nothing distinguishes them, because no activation publishes that
code and no weight holds it, which is itself the evidence, since the CPU reference and the GPU agree byte for
byte on real data. The port follows the shaders.

**`layout: 'auto'` drops bindings a kernel does not read**, so five self-test cases produced pure zeros with
no exception raised. The port uses explicit layouts everywhere and installs an `uncapturederror` handler: a
WebGPU validation failure is a no-op, not a throw, and a no-op reads exactly like a kernel that computes zeros.

**A rejected command buffer is silent in the same way.** Binding one shared placeholder buffer to several
*writable* bindings is writable aliasing, which WebGPU refuses at `encoder.finish()`, before anything runs,
with no dispatch to blame. The whole frame was dropped and every output stayed at zero. `Network.run` now
validates the first frame inside an error scope, and each writable binding has its own placeholder.

**A name is not a value, and an ungated stage is an unchecked one.** The post block's skip comes from block
0's *output*; in the Vulkan graph the variable holding it is called `adapter`, though it is allocated as
`"retained full block0"` and captured as the `block-0` boundary. This port followed the name and skipped from
the input adapter instead. Every block boundary stayed exact, because the mistake is downstream of the last
one: the boundaries end at block 69, and the post block, the head and the composition had no gate at all.
The `nr768` fixture was sitting in the fixture directory the whole time with the head and the composed image
in it. The lesson is the cheaper half of the story — the part written rather than ported was the part left
untested, and its only check was a second hand-written copy of the same arithmetic agreeing with it.

## Layout

```
src/numerics.js         the publication grid on the CPU: the port's oracle
src/numerics_cases.js   what the fixture answers, shared by the Node check and the browser self-test
src/gpu.js              the thin WebGPU layer: device, buffers, compile, dispatch
src/passes.js           pipelines, tensors, and the recorder the graph builds into
src/geometry.js         the padded field, the pooling levels, the window phases, the block layouts
src/model.js            weight loading: stage buffers, and the tensors no kernel addresses in place
src/graph.js            the 71 blocks
src/network.js          device, weights, kernels, a recorded graph
src/matmul/             every matrix multiply but two: one plain kernel and the transforms that specialize it
src/window/             8x8 shifted-window attention, the same way
src/parity.js           the fixture comparison
src/server.js           static serving, the extra mounts, and the /report endpoint
demo/                   the interactive demo: viewer setup, the scene, the UI and the frame passes
shaders/numerics.wgsl   the publication grid in WGSL
shaders/gemm_f16.wgsl   the input adapter and the head
shaders/vit.wgsl        the global attention at the bottom of the U-net
shaders/ops.wgsl        the pool, the two skip merges, the format conversions
shaders/frame.wgsl      input features, the temporal blend, and the display transform
shaders/preprocess.wgsl input features from a recorded proxy, for the parity harness
tools/dump_numerics.cpp generates web/fixtures/numerics.bin from the Vulkan reference
tools/check.mjs         every gate, in one command
tools/check_numerics.mjs the JS half of the arithmetic check, no GPU
tools/headless.mjs      runs a page in headless Chrome and exits with its verdict
```

The two generated kernels are the reason the port is worth reading twice. Each is one plain, scalar WGSL
kernel plus a directory of source-to-source transforms, applied in an order their preconditions fix. Each
transform is a separate file because each is a separate claim about what may be changed without changing the
result — the thread mapping, the exponent reduction, the operand format, the publication — and the parity
harness is what settles the claim. The plain kernel stays in the repository, readable, and four times slower
than the network can afford.

## Where the speed went

A warm frame at 512×512 is 73 ms across 451 dispatches, against about 2.7 ms for the Vulkan implementation on
the same card:

```
gemm_fp8            44.5 ms
window_attend       16.8 ms
vit_attend           3.0 ms
gemm_f16             1.1 ms
everything else      0.8 ms
```

Almost all of the remaining gap is the three routes WebGPU cannot take, and it is worth being specific:

* **No tensor core.** Every 16-product group is emulated: a pass to find the shared exponent, a truncation and
  an integer add per product, one rounding at the end. Call it six operations where the hardware does one.
  This is inherent and it is most of the 44 ms above.
* **No fusion between blocks.** The Vulkan version runs a whole 32-channel block as a single kernel with the
  FFN, the QKV projection, the attention and the output projection all staying in registers. Here each is a
  dispatch and every intermediate makes a round trip through memory. The cosine normalization *is* fused into
  the attention, which is what a fused block would extend.
* **No chaining.** The fused route also drops the barriers between launches and lets consumers spin on
  producer counters. WebGPU has one barrier between dispatches and no way to opt out.

Two things that are not in the list, because they were paid for rather than lost. The weights upload as the
model file stores them, in MMA fragment order, and the GEMM addresses that order directly — so there is no
host pass over 141 MiB at load. And every shape, stride and byte offset is a pipeline-creation override
rather than a uniform read, which folds the partial-tile guards away and costs around two hundred pipelines,
compiled off the queue while the weights download.

A fourth cost belongs to the demo rather than to the port. The viewer renders with WebGL and the network runs
on WebGPU, and the browser offers no way to share a texture between them, so every frame is read back into
typed arrays and uploaded again. At 1904×930 that is about 15 ms.
