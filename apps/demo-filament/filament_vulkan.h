// The demo's Filament-side Vulkan glue: the shared context that hands Filament the demo's device, and the queueing
// of GPU work (the NR pass) into Filament's command stream with the backend's images. This is the only translation
// unit that sees Filament's Vulkan headers; the rest of the demo talks to it through gpu_bridge.h handles.
#pragma once
#include <functional>
#include <vector>

#include "gpu_bridge.h"

namespace filament {
class Engine;
class Texture;
}

// A VulkanSharedContext for Engine::Builder::sharedContext(); lives as long as the engine
void* createSharedContext(const GpuDevice& device);
void destroySharedContext(void* sharedContext);

// Queues `work` on Filament's backend thread, at this point of the command stream: it gets the command buffer
// Filament is recording (outside of any render pass) and `textures` as the backend holds them (image, format,
// layout). The layouts the work leaves in `images` are reported back to Filament's tracking.
using GpuWork = std::function<void(void* commandBuffer, std::vector<GpuImage>& images)>;
void queueGpuWork(filament::Engine& engine, const std::vector<filament::Texture*>& textures, GpuWork work);
