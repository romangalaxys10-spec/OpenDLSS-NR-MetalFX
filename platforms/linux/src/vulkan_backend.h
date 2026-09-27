// vulkan_backend.h — the Linux Vulkan backend (OpenDLSS-NR MetalFX).
//
// Full IBackend implementation on top of the runtime-loaded Vulkan loader
// (see vk_boot.h — no SDK headers): the Swin/ViT graph runs in the project's
// GLSL compute kernels compiled to SPIR-V (platforms/linux/shaders/*.comp),
// fp16 storage as one-half-per-uint32 SSBO words, E4M3 publications, storage
// images RGBA32F for the media ops, and the same block/transition driver
// semantics as the macOS MetalBackend (input embed -> per-block FFN / skip /
// attention / epilogue -> pool/up-GEMM and down-GEMM/upsample/skip encoder-
// decoder transitions -> head). Game mode uses Lanczos upscale + temporal
// blend as the non-MetalFX fallback scaler.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include "opendlss/backend.h"

namespace opendlss {

// Defined in vulkan_backend.cpp. Returns nullptr when no Vulkan device can be
// initialized (loader missing, no GPU, device creation failed) so that
// create_preferred_backend() falls back to the CPU reference.
std::unique_ptr<IBackend> create_vulkan_backend();

} // namespace opendlss
