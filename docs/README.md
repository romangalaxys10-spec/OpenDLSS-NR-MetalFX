# Documentation index

| Doc | Contents |
| --- | --- |
| [../README.md](../README.md) | project overview, quick start |
| [INSTALLATION.md](INSTALLATION.md) | per-platform install, viewer, CI |
| [USAGE.md](USAGE.md) | CLI reference, options, conditioning scalars |
| [ARCHITECTURE.md](ARCHITECTURE.md) | repo layout, backend contract, data flow |
| [METALFX.md](METALFX.md) | the MetalFX spatial/temporal integration in depth |
| [NETWORK.md](NETWORK.md) | the graph: lanes, blocks, attention, transitions, head |
| [WEIGHTS.md](WEIGHTS.md) | manifest format, tensor naming, quantization rules |
| [GAMES.md](GAMES.md) | engine integration via the C SDK, engine checklist |
| [VIDEO.md](VIDEO.md) | video pipeline, Y4M/ffmpeg workflow |
| [IMAGES.md](IMAGES.md) | image pipeline and stages |
| [../CHANGELOG.md](../CHANGELOG.md) | release history |

## The one-paragraph version

OpenDLSS-NR MetalFX ports the OpenDLSS-NR neural rendering network family
(Swin/ViT transformer U-net, FP8(E4M3) publications on FP16 storage, bit-trick
attention, padded-field geometry) to Apple Silicon with Metal compute + MetalFX
scalers, and to Windows (D3D12/DirectML) and Linux (Vulkan) with the same
numerics contract. The CPU backend is the parity golden. Three content modes —
image, video, game — drive the same ops through one backend interface, exposed
by a CLI, a C SDK for engines, demos, tests and CI.
