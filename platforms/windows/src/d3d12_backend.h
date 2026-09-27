// d3d12_backend.h — the Windows D3D12 + DirectML backend.
//
// Host structure mirrors platforms/macos/src/metal_backend.mm one-for-one
// (same method list, same fallbacks); the DirectML GEMM path plays the role
// MetalFX plays on macOS — an accelerator layered over our own HLSL compute
// kernels, never a requirement:
//
//   * device/queue creation with debug-layer opt-in and WARP fallback
//   * every kernel from platforms/windows/shaders compiled at runtime with
//     d3dcompiler (D3DCompileFromFile, cs_5_0 profile; the sources are
//     SM 6.x compatible and can be precompiled with DXC — see CMakeLists)
//   * shader directory resolution: $OPENDLSS_D3D_SHADER_DIR first, then the
//     build-dir copy (CMake copies the shaders next to the binary), then the
//     source-tree path baked in at configure time
//   * token tensors live in byte-address buffers as f16 pairs (two halfs per
//     dword, little-endian) — one consistent layout for every kernel
//   * DirectML accelerates the big channel GEMMs (>= 4096 weight elements)
//     when directml.dll loads; every GEMM falls back to the odl_channel_gemm
//     HLSL kernel otherwise, and E4M3 publications run in our kernels either
//     way so the quantization grid never depends on DML
//   * game mode uses our Lanczos + reprojection + temporal-blend fallback
//     (DirectML has no temporal scaler — the MTLFXTemporalScaler role is
//     filled by our own kernels)
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include "opendlss/backend.h"

#include <memory>

namespace opendlss {

struct D3D12BackendOptions {
    bool enableDebugLayer = false;  // also settable via OPENDLSS_D3D12_DEBUG=1
    bool preferWarp       = false;  // also settable via OPENDLSS_D3D12_WARP=1
    bool enableDirectML   = true;   // also settable via OPENDLSS_D3D12_DML=0
};

// Defined in d3d12_backend.cpp. Returns a backend whose info().kind is
// BackendKind::D3D12; the instance is inert (all ops report failure) when no
// D3D12 device could be created.
std::unique_ptr<IBackend> create_d3d12_backend();
std::unique_ptr<IBackend> create_d3d12_backend(const D3D12BackendOptions& options);

} // namespace opendlss
