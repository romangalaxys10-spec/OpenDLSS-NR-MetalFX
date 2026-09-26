#include "filament_vulkan.h"

#include <backend/VulkanInterop.h>
#include <backend/platforms/VulkanPlatform.h>
#include <filament/Engine.h>
#include <filament/Texture.h>

#include <memory>

using filament::backend::VulkanInteropContext;
using filament::backend::VulkanInteropTexture;
using filament::backend::VulkanPlatform;

void* createSharedContext(const GpuDevice& device) {
  auto* context = new VulkanPlatform::VulkanSharedContext();
  context->instance = (VkInstance)device.instance;
  context->physicalDevice = (VkPhysicalDevice)device.physicalDevice;
  context->logicalDevice = (VkDevice)device.device;
  context->graphicsQueueFamilyIndex = device.queueFamily;
  context->graphicsQueueIndex = device.rendererQueueIndex;
  context->debugUtilsEnabled = device.debugUtils;
  return context;
}

void destroySharedContext(void* sharedContext) { delete (VulkanPlatform::VulkanSharedContext*)sharedContext; }

void queueGpuWork(filament::Engine& engine, const std::vector<filament::Texture*>& textures, GpuWork work) {
  std::vector<uint32_t> handles;
  handles.reserve(textures.size());
  for (filament::Texture* texture : textures) handles.push_back(engine.getVulkanInteropTextureHandle(texture));
  engine.queueVulkanCommand([handles = std::move(handles), work = std::move(work)](VulkanInteropContext& context) {
    std::vector<GpuImage> images(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
      VulkanInteropTexture texture = context.getTexture(handles[i]);
      images[i].image = (uint64_t)(uintptr_t)texture.image;
      images[i].format = (uint32_t)texture.format;
      images[i].layout = (uint32_t)texture.layout;
      images[i].width = texture.width;
      images[i].height = texture.height;
    }
    work((void*)context.getCommandBuffer(), images);
    for (size_t i = 0; i < handles.size(); ++i) context.setTextureLayout(handles[i], (VkImageLayout)images[i].layout);
  });
}
