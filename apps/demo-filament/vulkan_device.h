// The demo's Vulkan instance and device (volk): the NR kernels' features and extensions (vk::DeviceRequirements)
// plus what a renderer needs (a swapchain, every supported core feature, two graphics queues when the family has
// them). Filament adopts the result through its shared-context path and the NR side through vk::Context.
#pragma once
#include <string>

#include "gpu_bridge.h"

class VulkanDevice {
 public:
  VulkanDevice();
  ~VulkanDevice();
  const GpuDevice& handles() const { return handles_; }
  const std::string& deviceName() const { return deviceName_; }

 private:
  GpuDevice handles_;
  std::string deviceName_;
  uint64_t messenger_ = 0;   // VkDebugUtilsMessengerEXT, with DLSS5_DEMO_VALIDATION=1
};
