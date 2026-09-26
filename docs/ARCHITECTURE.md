# Architecture

```
include/opendlss/opendlss.h      C ABI (stable)
src/sdk/opendlss.cpp             C ABI -> frame pipeline
src/common/                      nr_frame* (product pipeline), nr_image, nr_video,
                                 nr_numeric (exact f16/E4M3), nr_cpu_reference, nr_geometry shim
        │
        ├── Apple ──► src/metal/  nr_metal_context / nr_metal_model / nr_metal_kernels /
        │             nr_metal_graph / nr_metalfx + shaders/*.metal   (MetalFX on top)
        └── Win/Linux ► src/vulkan/ vk_context / nr_model / kernels / nr_graph / verify (+ main.cpp = dlss5vk)
                      shaders/vulkan-glsl/*.comp -> glslang -> .spv; ptxgen/ -> PTX fast path
```

The two backends expose the same API shape (`Context/Model/Kernels/Graph/Activation`), so the
product pipeline exists once (`src/common/nr_frame_impl.inc`, instantiated per backend). The
Vulkan tree is the upstream code carried over; the Metal tree is new. `dlss5vk` remains the
bit-exactness workbench on NVIDIA hardware; `opendlss` is the product surface everywhere.

Graph recording (both backends): pre adapter + block 0 -> encoder 32 (1-4) -> 64 (5-8) -> 128
(9-14) -> 256 (15-22) -> split-512 (23-30) -> global ViT 1024ch (31-38) -> decoder 512 (39-47) ->
256 (48-55) -> 128 (56-61) -> 64 (62-65) -> 32 (66-70) -> post blend + f16 head -> f32 RGBA.
Window phases cycle {0,0} {4,4} {4,0} {0,4} per level; every inter-kernel tensor is E4M3/F16
published with the exact bit rules of docs/original-design/numerics.md.
