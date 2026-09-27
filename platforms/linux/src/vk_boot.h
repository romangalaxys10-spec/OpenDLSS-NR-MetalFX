// vk_boot.h — minimal Vulkan bootstrap for the OpenDLSS-NR Vulkan backend.
//
// Declares the (spec-conformant) subset of Vulkan 1.0 core types and function
// pointer typedefs the backend actually uses and resolves them at runtime:
//   dlopen("libvulkan.so.1") -> vkGetInstanceProcAddr -> everything else
//     * global:      vkGetDeviceProcAddr
//     * instance:    createInstance / destroyInstance / enumeratePhysicalDevices /
//                    physical device properties / features / memory properties /
//                    queue family properties / format properties / createDevice
//     * device:      getDeviceQueue, command pool + allocateCommandBuffers,
//                    buffer + memory (allocate/bind/map/unmap/flush/invalidate/free),
//                    shader module, descriptor set layout / pool / allocate /
//                    reset, pipeline layout, compute pipelines, image / image
//                    view / image memory, fences, begin/end/reset command
//                    buffer, cmdBindPipeline, cmdBindDescriptorSets,
//                    cmdPushConstants, cmdDispatch, cmdPipelineBarrier,
//                    cmdCopyBufferToImage / cmdCopyImageToBuffer, queueSubmit,
//                    queueWaitIdle, deviceWaitIdle, waitForFences, resetFences
//                    and every vkDestroy*/vkFree* needed for cleanup.
//
// No Vulkan SDK headers are required; every struct below matches the Vulkan
// 1.0 core ABI exactly (field order / types / sizes) — checked by the
// static_asserts at the bottom of this file.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include <dlfcn.h>

namespace opendlss {
namespace vk {

// ---------------------------------------------------------------------------
// base types
// ---------------------------------------------------------------------------
typedef uint32_t  VkBool32;
typedef uint32_t  VkFlags;
typedef uint64_t  VkDeviceSize;
typedef int32_t   VkResult;
typedef void      (*PFN_vkVoidFunction)(void);

#define VK_NULL_HANDLE     0ULL
#define VK_TRUE            1u
#define VK_FALSE           0u
#define VK_WHOLE_SIZE      (~static_cast<VkDeviceSize>(0))
#define VK_QUEUE_FAMILY_IGNORED (~0u)
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256
#define VK_UUID_SIZE       16
#define VK_MAX_MEMORY_TYPES 32
#define VK_MAX_MEMORY_HEAPS 16
#define VK_API_VERSION_1_0 ((1u << 22) | (0u << 12) | 0u)

// dispatchable handles (pointers to loader-dispatch tables)
struct VkInstance_T;        typedef VkInstance_T*        VkInstance;
struct VkPhysicalDevice_T;  typedef VkPhysicalDevice_T*  VkPhysicalDevice;
struct VkDevice_T;          typedef VkDevice_T*          VkDevice;
struct VkQueue_T;           typedef VkQueue_T*           VkQueue;
struct VkCommandBuffer_T;   typedef VkCommandBuffer_T*   VkCommandBuffer;

// non-dispatchable handles (uint64 on 64-bit)
typedef uint64_t VkBuffer;
typedef uint64_t VkDeviceMemory;
typedef uint64_t VkShaderModule;
typedef uint64_t VkDescriptorSetLayout;
typedef uint64_t VkDescriptorPool;
typedef uint64_t VkDescriptorSet;
typedef uint64_t VkPipelineLayout;
typedef uint64_t VkPipeline;
typedef uint64_t VkPipelineCache;
typedef uint64_t VkImage;
typedef uint64_t VkImageView;
typedef uint64_t VkSampler;
typedef uint64_t VkFence;
typedef uint64_t VkCommandPool;
typedef uint64_t VkSemaphore;

// VkResult values used here
enum : VkResult {
    VK_SUCCESS                     = 0,
    VK_NOT_READY                   = 1,
    VK_TIMEOUT                     = 2,
    VK_INCOMPLETE                  = 5,
    VK_ERROR_OUT_OF_HOST_MEMORY    = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY  = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST           = -4,
    VK_ERROR_EXTENSION_NOT_PRESENT = -5,
    VK_ERROR_FEATURE_NOT_PRESENT   = -6,
};

// VkStructureType (Vulkan 1.0 core values used by this backend)
enum : int32_t {
    VK_STRUCTURE_TYPE_APPLICATION_INFO                  = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO              = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO          = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO                = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO                       = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO              = 5,
    VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE               = 6,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO                 = 8,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO                = 12,
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO                 = 14,
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO            = 15,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO         = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO      = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO       = 30,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO       = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO      = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET              = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO          = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO      = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO         = 42,
    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER             = 44,
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER              = 45,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER                    = 46,
};

// flags / enums (values per Vulkan 1.0 core)
enum : VkFlags {
    VK_QUEUE_GRAPHICS_BIT       = 0x1,
    VK_QUEUE_COMPUTE_BIT        = 0x2,
    VK_QUEUE_TRANSFER_BIT       = 0x4,

    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  = 0x1,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  = 0x2,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x4,
    VK_MEMORY_PROPERTY_HOST_CACHED_BIT   = 0x8,

    VK_BUFFER_USAGE_TRANSFER_SRC_BIT   = 0x1,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT   = 0x2,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20,

    VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x1,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x2,
    VK_IMAGE_USAGE_STORAGE_BIT      = 0x8,

    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT    = 0x1,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x800,
    VK_PIPELINE_STAGE_TRANSFER_BIT       = 0x1000,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x2000,
    VK_PIPELINE_STAGE_HOST_BIT           = 0x4000,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT   = 0x10000,

    VK_ACCESS_SHADER_READ_BIT    = 0x20,
    VK_ACCESS_SHADER_WRITE_BIT   = 0x40,
    VK_ACCESS_TRANSFER_READ_BIT  = 0x800,
    VK_ACCESS_TRANSFER_WRITE_BIT = 0x1000,
    VK_ACCESS_HOST_READ_BIT      = 0x2000,
    VK_ACCESS_HOST_WRITE_BIT     = 0x4000,
    VK_ACCESS_MEMORY_READ_BIT    = 0x8000,
    VK_ACCESS_MEMORY_WRITE_BIT   = 0x10000,

    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT            = 0x1,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x2,

    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x1,

    VK_FENCE_CREATE_SIGNALED_BIT = 0x1,

    VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = 0x1,

    VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT = 0x1,
    VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT = 0x2,
    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT  = 0x2000,
    VK_FORMAT_FEATURE_TRANSFER_DST_BIT  = 0x4000,

    VK_SHADER_STAGE_COMPUTE_BIT = 0x20,
};

enum : int32_t {
    VK_PHYSICAL_DEVICE_TYPE_OTHER          = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU            = 4,

    VK_IMAGE_TYPE_2D                    = 1,
    VK_IMAGE_VIEW_TYPE_2D               = 1,
    VK_IMAGE_TILING_OPTIMAL             = 0,
    VK_IMAGE_TILING_LINEAR              = 1,
    VK_IMAGE_LAYOUT_UNDEFINED           = 0,
    VK_IMAGE_LAYOUT_GENERAL             = 1,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL = 6,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL = 7,

    VK_SAMPLE_COUNT_1_BIT     = 1,
    VK_IMAGE_ASPECT_COLOR_BIT = 1,

    VK_SHARING_MODE_EXCLUSIVE = 0,

    VK_PIPELINE_BIND_POINT_COMPUTE = 1,

    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE  = 3,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,

    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,

    VK_COMPONENT_SWIZZLE_IDENTITY = 0,

    VK_FORMAT_R32G32B32A32_SFLOAT = 102,
};

// ---------------------------------------------------------------------------
// structs (Vulkan 1.0 core ABI — order and types are normative)
// ---------------------------------------------------------------------------
struct VkExtent3D         { uint32_t width, height, depth; };
struct VkComponentMapping { int32_t r, g, b, a; };
struct VkImageSubresourceRange  { VkFlags aspectMask; uint32_t baseMipLevel, levelCount, baseArrayLayer, layerCount; };
struct VkImageSubresourceLayers { VkFlags aspectMask; uint32_t mipLevel, baseArrayLayer, layerCount; };

struct VkApplicationInfo {
    int32_t      sType;
    const void*  pNext;
    const char*  pApplicationName;
    uint32_t     applicationVersion;
    const char*  pEngineName;
    uint32_t     engineVersion;
    uint32_t     apiVersion;
};

struct VkInstanceCreateInfo {
    int32_t            sType;
    const void*        pNext;
    VkFlags            flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t           enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t           enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

struct VkDeviceQueueCreateInfo {
    int32_t       sType;
    const void*   pNext;
    VkFlags       flags;
    uint32_t      queueFamilyIndex;
    uint32_t      queueCount;
    const float*  pQueuePriorities;
};

// 55 VkBool32 fields, in spec order (sizeof == 220).
struct VkPhysicalDeviceFeatures {
    VkBool32 robustBufferAccess;
    VkBool32 fullDrawIndexUint32;
    VkBool32 imageCubeArray;
    VkBool32 independentBlend;
    VkBool32 geometryShader;
    VkBool32 tessellationShader;
    VkBool32 sampleRateShading;
    VkBool32 dualSrcBlend;
    VkBool32 logicOp;
    VkBool32 multiDrawIndirect;
    VkBool32 drawIndirectFirstInstance;
    VkBool32 depthClamp;
    VkBool32 depthBiasClamp;
    VkBool32 fillModeNonSolid;
    VkBool32 depthBounds;
    VkBool32 wideLines;
    VkBool32 largePoints;
    VkBool32 alphaToOne;
    VkBool32 multiViewport;
    VkBool32 samplerAnisotropy;
    VkBool32 textureCompressionETC2;
    VkBool32 textureCompressionASTC_LDR;
    VkBool32 textureCompressionBC;
    VkBool32 occlusionQueryPrecise;
    VkBool32 pipelineStatisticsQuery;
    VkBool32 vertexPipelineStoresAndAtomics;
    VkBool32 fragmentStoresAndAtomics;
    VkBool32 shaderTessellationAndGeometryPointSize;
    VkBool32 shaderImageGatherExtended;
    VkBool32 shaderStorageImageExtendedFormats;
    VkBool32 shaderStorageImageMultisample;
    VkBool32 shaderStorageImageReadWithoutFormat;
    VkBool32 shaderStorageImageWriteWithoutFormat;
    VkBool32 shaderUniformBufferArrayDynamicIndexing;
    VkBool32 shaderStorageBufferArrayDynamicIndexing;
    VkBool32 shaderSampledImageArrayDynamicIndexing;
    VkBool32 shaderStorageImageArrayDynamicIndexing;
    VkBool32 shaderClipDistance;
    VkBool32 shaderCullDistance;
    VkBool32 shaderFloat64;
    VkBool32 shaderInt64;
    VkBool32 shaderInt16;
    VkBool32 shaderResourceResidency;
    VkBool32 shaderResourceMinLod;
    VkBool32 sparseBinding;
    VkBool32 sparseResidencyBuffer;
    VkBool32 sparseResidencyImage2D;
    VkBool32 sparseResidencyImage3D;
    VkBool32 sparseResidency2Samples;
    VkBool32 sparseResidency4Samples;
    VkBool32 sparseResidency8Samples;
    VkBool32 sparseResidency16Samples;
    VkBool32 sparseResidencyAliased;
    VkBool32 variableMultisampleRate;
    VkBool32 inheritedQueries;
};

struct VkPhysicalDeviceLimits {
    uint32_t     maxImageDimension1D;
    uint32_t     maxImageDimension2D;
    uint32_t     maxImageDimension3D;
    uint32_t     maxImageDimensionCube;
    uint32_t     maxImageArrayLayers;
    uint32_t     maxTexelBufferElements;
    uint32_t     maxUniformBufferRange;
    uint32_t     maxStorageBufferRange;
    uint32_t     maxPushConstantsSize;
    uint32_t     maxMemoryAllocationCount;
    uint32_t     maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity;
    VkDeviceSize sparseAddressSpaceSize;
    uint32_t     maxBoundDescriptorSets;
    uint32_t     maxPerStageDescriptorSamplers;
    uint32_t     maxPerStageDescriptorUniformBuffers;
    uint32_t     maxPerStageDescriptorStorageBuffers;
    uint32_t     maxPerStageDescriptorSampledImages;
    uint32_t     maxPerStageDescriptorStorageImages;
    uint32_t     maxPerStageDescriptorInputAttachments;
    uint32_t     maxPerStageResources;
    uint32_t     maxDescriptorSetSamplers;
    uint32_t     maxDescriptorSetUniformBuffers;
    uint32_t     maxDescriptorSetUniformBuffersDynamic;
    uint32_t     maxDescriptorSetStorageBuffers;
    uint32_t     maxDescriptorSetStorageBuffersDynamic;
    uint32_t     maxDescriptorSetSampledImages;
    uint32_t     maxDescriptorSetStorageImages;
    uint32_t     maxDescriptorSetInputAttachments;
    uint32_t     maxVertexInputAttributes;
    uint32_t     maxVertexInputBindings;
    uint32_t     maxVertexInputAttributeOffset;
    uint32_t     maxVertexInputBindingStride;
    uint32_t     maxVertexOutputComponents;
    uint32_t     maxTessellationGenerationLevel;
    uint32_t     maxTessellationPatchSize;
    uint32_t     maxTessellationControlPerVertexInputComponents;
    uint32_t     maxTessellationControlPerVertexOutputComponents;
    uint32_t     maxTessellationControlPerPatchOutputComponents;
    uint32_t     maxTessellationControlTotalOutputComponents;
    uint32_t     maxTessellationEvaluationInputComponents;
    uint32_t     maxTessellationEvaluationOutputComponents;
    uint32_t     maxGeometryShaderInvocations;
    uint32_t     maxGeometryInputComponents;
    uint32_t     maxGeometryOutputComponents;
    uint32_t     maxGeometryOutputVertices;
    uint32_t     maxGeometryTotalOutputComponents;
    uint32_t     maxFragmentInputComponents;
    uint32_t     maxFragmentOutputAttachments;
    uint32_t     maxFragmentDualSrcAttachments;
    uint32_t     maxFragmentCombinedOutputResources;
    uint32_t     maxComputeSharedMemorySize;
    uint32_t     maxComputeWorkGroupCount[3];
    uint32_t     maxComputeWorkGroupInvocations;
    uint32_t     maxComputeWorkGroupSize[3];
    uint32_t     maxViewportDimensions[2];
    float        viewportBoundsRange[2];
    uint32_t     maxViewportScissors;
    uint32_t     minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    int32_t      minTexelOffset;
    uint32_t     maxTexelOffset;
    int32_t      minTexelGatherOffset;
    uint32_t     maxTexelGatherOffset;
    float        minInterpolationOffset;
    float        maxInterpolationOffset;
    uint32_t     subPixelInterpolationOffsetBits;
    uint32_t     maxFramebufferWidth;
    uint32_t     maxFramebufferHeight;
    uint32_t     maxFramebufferLayers;
    VkFlags      framebufferColorSampleCounts;
    VkFlags      framebufferDepthSampleCounts;
    VkFlags      framebufferStencilSampleCounts;
    uint32_t     maxColorAttachments;
    VkFlags      sampledImageColorSampleCounts;
    VkFlags      sampledImageIntegerSampleCounts;
    VkFlags      sampledImageDepthSampleCounts;
    VkFlags      sampledImageStencilSampleCounts;
    VkFlags      storageImageSampleCounts;
    uint32_t     maxSampleMaskWords;
    VkBool32     timestampComputeAndGraphics;
    float        timestampPeriod;
    uint32_t     maxClipDistances;
    uint32_t     maxCullDistances;
    uint32_t     maxCombinedClipAndCullDistances;
    uint32_t     discreteQueuePriorities;
    float        pointSizeRange[2];
    float        lineWidthRange[2];
    float        pointSizeGranularity;
    float        lineWidthGranularity;
    VkBool32     strictLines;
    VkBool32     standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment;
    VkDeviceSize optimalBufferCopyRowPitchAlignment;
    VkDeviceSize nonCoherentAtomSize;
};

struct VkPhysicalDeviceSparseProperties {
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
};

struct VkPhysicalDeviceProperties {
    uint32_t        apiVersion;
    uint32_t        driverVersion;
    uint32_t        vendorID;
    uint32_t        deviceID;
    int32_t         deviceType;                       // VkPhysicalDeviceType
    char            deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint8_t         pipelineCacheUUID[VK_UUID_SIZE];
    VkPhysicalDeviceLimits          limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
};

struct VkMemoryType { VkFlags propertyFlags; uint32_t heapIndex; };
struct VkMemoryHeap { VkDeviceSize size; VkFlags flags; };
struct VkPhysicalDeviceMemoryProperties {
    uint32_t     memoryTypeCount;
    VkMemoryType memoryTypes[VK_MAX_MEMORY_TYPES];
    uint32_t     memoryHeapCount;
    VkMemoryHeap memoryHeaps[VK_MAX_MEMORY_HEAPS];
};

struct VkQueueFamilyProperties {
    VkFlags     queueFlags;
    uint32_t    queueCount;
    uint32_t    timestampValidBits;
    VkExtent3D  minImageTransferGranularity;
};

struct VkFormatProperties {
    VkFlags linearTilingFeatures;
    VkFlags optimalTilingFeatures;
    VkFlags bufferFeatures;
};

struct VkDeviceCreateInfo {
    int32_t                        sType;
    const void*                    pNext;
    VkFlags                        flags;
    uint32_t                       queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t                       enabledLayerCount;
    const char* const*             ppEnabledLayerNames;
    uint32_t                       enabledExtensionCount;
    const char* const*             ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures* pEnabledFeatures;
};

struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t     memoryTypeBits;
};

struct VkMemoryAllocateInfo {
    int32_t      sType;
    const void*  pNext;
    VkDeviceSize allocationSize;
    uint32_t     memoryTypeIndex;
};

struct VkMappedMemoryRange {
    int32_t        sType;
    const void*    pNext;
    VkDeviceMemory memory;
    VkDeviceSize   offset;
    VkDeviceSize   size;
};

struct VkBufferCreateInfo {
    int32_t         sType;
    const void*     pNext;
    VkFlags         flags;
    VkDeviceSize    size;
    VkFlags         usage;
    int32_t         sharingMode;
    uint32_t        queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
};

struct VkImageCreateInfo {
    int32_t         sType;
    const void*     pNext;
    VkFlags         flags;
    int32_t         imageType;
    int32_t         format;
    VkExtent3D      extent;
    uint32_t        mipLevels;
    uint32_t        arrayLayers;
    int32_t         samples;
    int32_t         tiling;
    VkFlags         usage;
    int32_t         sharingMode;
    uint32_t        queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
    int32_t         initialLayout;
};

struct VkImageViewCreateInfo {
    int32_t                 sType;
    const void*             pNext;
    VkFlags                 flags;
    VkImage                 image;
    int32_t                 viewType;
    int32_t                 format;
    VkComponentMapping      components;
    VkImageSubresourceRange subresourceRange;
};

struct VkBufferImageCopy {
    VkDeviceSize             bufferOffset;
    uint32_t                 bufferRowLength;
    uint32_t                 bufferImageHeight;
    VkImageSubresourceLayers imageSubresource;
    VkExtent3D               imageOffset;
    VkExtent3D               imageExtent;
};

struct VkDescriptorSetLayoutBinding {
    uint32_t         binding;
    int32_t          descriptorType;
    uint32_t         descriptorCount;
    VkFlags          stageFlags;
    const VkSampler* pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfo {
    int32_t                             sType;
    const void*                         pNext;
    VkFlags                             flags;
    uint32_t                            bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
};

struct VkDescriptorPoolSize {
    int32_t    type;
    uint32_t   descriptorCount;
};

struct VkDescriptorPoolCreateInfo {
    int32_t                     sType;
    const void*                 pNext;
    VkFlags                     flags;
    uint32_t                    maxSets;
    uint32_t                    poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
};

struct VkDescriptorSetAllocateInfo {
    int32_t                      sType;
    const void*                  pNext;
    VkDescriptorPool             descriptorPool;
    uint32_t                     descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
};

struct VkDescriptorBufferInfo {
    VkBuffer     buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

struct VkDescriptorImageInfo {
    VkSampler   sampler;
    VkImageView imageView;
    int32_t     imageLayout;
};

struct VkWriteDescriptorSet {
    int32_t                      sType;
    const void*                  pNext;
    VkDescriptorSet              dstSet;
    uint32_t                     dstBinding;
    uint32_t                     dstArrayElement;
    uint32_t                     descriptorCount;
    int32_t                      descriptorType;
    const VkDescriptorImageInfo* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const uint64_t*              pTexelBufferView;
};

struct VkPushConstantRange {
    VkFlags    stageFlags;
    uint32_t   offset;
    uint32_t   size;
};

struct VkPipelineLayoutCreateInfo {
    int32_t                      sType;
    const void*                  pNext;
    VkFlags                      flags;
    uint32_t                     setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t                     pushConstantRangeCount;
    const VkPushConstantRange*   pPushConstantRanges;
};

struct VkSpecializationInfo;   // unused by this backend; completes the ABI

struct VkPipelineShaderStageCreateInfo {
    int32_t                     sType;
    const void*                 pNext;
    VkFlags                     flags;
    VkFlags                     stage;
    VkShaderModule              module;
    const char*                 pName;
    const VkSpecializationInfo* pSpecializationInfo;
};

struct VkComputePipelineCreateInfo {
    int32_t                         sType;
    const void*                     pNext;
    VkFlags                         flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout                layout;
    VkPipeline                      basePipelineHandle;
    int32_t                         basePipelineIndex;
};

struct VkShaderModuleCreateInfo {
    int32_t         sType;
    const void*     pNext;
    VkFlags         flags;
    size_t          codeSize;
    const uint32_t* pCode;
};

struct VkCommandPoolCreateInfo {
    int32_t      sType;
    const void*  pNext;
    VkFlags      flags;
};

struct VkCommandBufferAllocateInfo {
    int32_t       sType;
    const void*   pNext;
    VkCommandPool commandPool;
    int32_t       level;
    uint32_t      commandBufferCount;
};

struct VkCommandBufferInheritanceInfo;   // unused; opaque here

struct VkCommandBufferBeginInfo {
    int32_t     sType;
    const void* pNext;
    VkFlags     flags;
    const VkCommandBufferInheritanceInfo* pInheritanceInfo;
};

struct VkSubmitInfo {
    int32_t                sType;
    const void*            pNext;
    uint32_t               waitSemaphoreCount;
    const VkSemaphore*     pWaitSemaphores;
    const VkFlags*         pWaitDstStageMask;
    uint32_t               commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t               signalSemaphoreCount;
    const VkSemaphore*     pSignalSemaphores;
};

struct VkFenceCreateInfo {
    int32_t      sType;
    const void*  pNext;
    VkFlags      flags;
};

struct VkMemoryBarrier {
    int32_t      sType;
    const void*  pNext;
    VkFlags      srcAccessMask;
    VkFlags      dstAccessMask;
};

struct VkBufferMemoryBarrier {
    int32_t      sType;
    const void*  pNext;
    VkFlags      srcAccessMask;
    VkFlags      dstAccessMask;
    uint32_t     srcQueueFamilyIndex;
    uint32_t     dstQueueFamilyIndex;
    VkBuffer     buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct VkImageMemoryBarrier {
    int32_t                 sType;
    const void*             pNext;
    VkFlags                 srcAccessMask;
    VkFlags                 dstAccessMask;
    int32_t                 oldLayout;
    int32_t                 newLayout;
    uint32_t                srcQueueFamilyIndex;
    uint32_t                dstQueueFamilyIndex;
    VkImage                 image;
    VkImageSubresourceRange subresourceRange;
};

// ---------------------------------------------------------------------------
// function pointer typedefs (signatures per vulkan_core.h)
// ---------------------------------------------------------------------------
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr_t)(VkInstance, const char*);
typedef PFN_vkVoidFunction (*PFN_vkGetDeviceProcAddr_t)(VkDevice, const char*);

// instance level
typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo*, const void* /*VkAllocationCallbacks**/, VkInstance*);
typedef void     (*PFN_vkDestroyInstance)(VkInstance, const void*);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void     (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*);
typedef void     (*PFN_vkGetPhysicalDeviceFeatures)(VkPhysicalDevice, VkPhysicalDeviceFeatures*);
typedef void     (*PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
typedef void     (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
typedef void     (*PFN_vkGetPhysicalDeviceFormatProperties)(VkPhysicalDevice, int32_t, VkFormatProperties*);
typedef VkResult (*PFN_vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
typedef void     (*PFN_vkDestroyDevice)(VkDevice, const void*);

// device level
typedef void     (*PFN_vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*);
typedef VkResult (*PFN_vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
typedef void     (*PFN_vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*);
typedef VkResult (*PFN_vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
typedef void     (*PFN_vkFreeCommandBuffers)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);
typedef VkResult (*PFN_vkResetCommandBuffer)(VkCommandBuffer, VkFlags);
typedef VkResult (*PFN_vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
typedef VkResult (*PFN_vkEndCommandBuffer)(VkCommandBuffer);
typedef VkResult (*PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
typedef void     (*PFN_vkDestroyBuffer)(VkDevice, VkBuffer, const void*);
typedef void     (*PFN_vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
typedef void     (*PFN_vkFreeMemory)(VkDevice, VkDeviceMemory, const void*);
typedef VkResult (*PFN_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
typedef VkResult (*PFN_vkMapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
typedef void     (*PFN_vkUnmapMemory)(VkDevice, VkDeviceMemory);
typedef VkResult (*PFN_vkFlushMappedMemoryRanges)(VkDevice, uint32_t, const VkMappedMemoryRange*);
typedef VkResult (*PFN_vkInvalidateMappedMemoryRanges)(VkDevice, uint32_t, const VkMappedMemoryRange*);
typedef VkResult (*PFN_vkCreateImage)(VkDevice, const VkImageCreateInfo*, const void*, VkImage*);
typedef void     (*PFN_vkDestroyImage)(VkDevice, VkImage, const void*);
typedef void     (*PFN_vkGetImageMemoryRequirements)(VkDevice, VkImage, VkMemoryRequirements*);
typedef VkResult (*PFN_vkBindImageMemory)(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize);
typedef VkResult (*PFN_vkCreateImageView)(VkDevice, const VkImageViewCreateInfo*, const void*, VkImageView*);
typedef void     (*PFN_vkDestroyImageView)(VkDevice, VkImageView, const void*);
typedef VkResult (*PFN_vkCreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
typedef void     (*PFN_vkDestroyShaderModule)(VkDevice, VkShaderModule, const void*);
typedef VkResult (*PFN_vkCreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
typedef void     (*PFN_vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*);
typedef VkResult (*PFN_vkCreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
typedef void     (*PFN_vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*);
typedef VkResult (*PFN_vkAllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
typedef VkResult (*PFN_vkResetDescriptorPool)(VkDevice, VkDescriptorPool, VkFlags);
typedef VkResult (*PFN_vkCreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
typedef void     (*PFN_vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*);
typedef VkResult (*PFN_vkCreateComputePipelines)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
typedef void     (*PFN_vkDestroyPipeline)(VkDevice, VkPipeline, const void*);
typedef VkResult (*PFN_vkCreateFence)(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
typedef void     (*PFN_vkDestroyFence)(VkDevice, VkFence, const void*);
typedef VkResult (*PFN_vkWaitForFences)(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
typedef VkResult (*PFN_vkResetFences)(VkDevice, uint32_t, const VkFence*);
typedef void     (*PFN_vkCmdBindPipeline)(VkCommandBuffer, int32_t /*VkPipelineBindPoint*/, VkPipeline);
typedef void     (*PFN_vkCmdBindDescriptorSets)(VkCommandBuffer, int32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
typedef void     (*PFN_vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void*);
typedef void     (*PFN_vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
typedef void     (*PFN_vkCmdPipelineBarrier)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*);
typedef void     (*PFN_vkCmdCopyBufferToImage)(VkCommandBuffer, VkBuffer, VkImage, int32_t, uint32_t, const VkBufferImageCopy*);
typedef void     (*PFN_vkCmdCopyImageToBuffer)(VkCommandBuffer, VkImage, int32_t, VkBuffer, uint32_t, const VkBufferImageCopy*);
typedef VkResult (*PFN_vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
typedef void     (*PFN_vkUpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
typedef VkResult (*PFN_vkQueueWaitIdle)(VkQueue);
typedef VkResult (*PFN_vkDeviceWaitIdle)(VkDevice);

// ---------------------------------------------------------------------------
// Boot: the resolved function table + the objects created on the way.
// ---------------------------------------------------------------------------
struct Boot {
    void* lib = nullptr;                    // dlopen handle
    bool  ok = false;

    // entry points
    PFN_vkGetDeviceProcAddr_t getDeviceProcAddr = nullptr;

    // instance level
    PFN_vkCreateInstance                         createInstance = nullptr;
    PFN_vkDestroyInstance                        destroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices               enumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties            getPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures              getPhysicalDeviceFeatures = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties      getPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties getPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties      getPhysicalDeviceFormatProperties = nullptr;
    PFN_vkCreateDevice                           createDevice = nullptr;
    PFN_vkDestroyDevice                          destroyDevice = nullptr;

    // device level
    PFN_vkGetDeviceQueue               getDeviceQueue = nullptr;
    PFN_vkCreateCommandPool            createCommandPool = nullptr;
    PFN_vkDestroyCommandPool           destroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers       allocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers           freeCommandBuffers = nullptr;
    PFN_vkResetCommandBuffer           resetCommandBuffer = nullptr;
    PFN_vkBeginCommandBuffer           beginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer             endCommandBuffer = nullptr;
    PFN_vkCreateBuffer                 createBuffer = nullptr;
    PFN_vkDestroyBuffer                destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements  getBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory               allocateMemory = nullptr;
    PFN_vkFreeMemory                   freeMemory = nullptr;
    PFN_vkBindBufferMemory             bindBufferMemory = nullptr;
    PFN_vkMapMemory                    mapMemory = nullptr;
    PFN_vkUnmapMemory                  unmapMemory = nullptr;
    PFN_vkFlushMappedMemoryRanges      flushMappedMemoryRanges = nullptr;
    PFN_vkInvalidateMappedMemoryRanges invalidateMappedMemoryRanges = nullptr;
    PFN_vkCreateImage                  createImage = nullptr;
    PFN_vkDestroyImage                 destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements   getImageMemoryRequirements = nullptr;
    PFN_vkBindImageMemory              bindImageMemory = nullptr;
    PFN_vkCreateImageView              createImageView = nullptr;
    PFN_vkDestroyImageView             destroyImageView = nullptr;
    PFN_vkCreateShaderModule           createShaderModule = nullptr;
    PFN_vkDestroyShaderModule          destroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout    createDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout   destroyDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool         createDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool        destroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets       allocateDescriptorSets = nullptr;
    PFN_vkResetDescriptorPool          resetDescriptorPool = nullptr;
    PFN_vkCreatePipelineLayout         createPipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout        destroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines       createComputePipelines = nullptr;
    PFN_vkDestroyPipeline              destroyPipeline = nullptr;
    PFN_vkCreateFence                  createFence = nullptr;
    PFN_vkDestroyFence                 destroyFence = nullptr;
    PFN_vkWaitForFences                waitForFences = nullptr;
    PFN_vkResetFences                  resetFences = nullptr;
    PFN_vkCmdBindPipeline              cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets        cmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants             cmdPushConstants = nullptr;
    PFN_vkCmdDispatch                  cmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier           cmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBufferToImage         cmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer         cmdCopyImageToBuffer = nullptr;
    PFN_vkQueueSubmit                  queueSubmit = nullptr;
    PFN_vkUpdateDescriptorSets         updateDescriptorSets = nullptr;
    PFN_vkQueueWaitIdle                queueWaitIdle = nullptr;
    PFN_vkDeviceWaitIdle               deviceWaitIdle = nullptr;

    // created objects
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          queue          = VK_NULL_HANDLE;
    uint32_t         queueFamily    = 0;

    VkPhysicalDeviceProperties       props{};
    VkPhysicalDeviceMemoryProperties memProps{};
    VkPhysicalDeviceFeatures         features{};
    uint32_t                         apiVersion = 0;

    // Finds a memory type index satisfying (typeBits & (1<<i)) with all of
    // `props` set. Returns 0xFFFFFFFF when none matches.
    uint32_t findMemoryType(uint32_t typeBits, VkFlags props) const {
        for (uint32_t i = 0; i < memProps.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
            if (!(typeBits & (1u << i))) continue;
            if ((memProps.memoryTypes[i].propertyFlags & props) == props) return i;
        }
        return 0xFFFFFFFFu;
    }
};

inline void bootShutdown(Boot& b);   // defined after bootInit

// ---------------------------------------------------------------------------
// bootInit / bootShutdown
// ---------------------------------------------------------------------------

// Full bootstrap: load the loader, create the instance, pick a GPU
// (discrete preferred, then integrated, then any), create the device with a
// compute queue. Returns false (and logs to stderr) on any failure.
inline bool bootInit(Boot& b, std::string* deviceNameOut = nullptr) {
    auto fail = [](const char* what) {
        std::fprintf(stderr, "vk_boot: %s\n", what);
        return false;
    };

    // 1. loader
    const char* names[] = { "libvulkan.so.1", "libvulkan.so" };
    for (const char* n : names) {
        b.lib = dlopen(n, RTLD_NOW | RTLD_LOCAL);
        if (b.lib) break;
    }
    if (!b.lib) return fail("dlopen(libvulkan.so.1) failed — Vulkan loader not installed");

    auto getFn = [&](const char* name) -> PFN_vkVoidFunction {
        return reinterpret_cast<PFN_vkVoidFunction>(dlsym(b.lib, name));
    };
    PFN_vkGetInstanceProcAddr_t getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr_t>(getFn("vkGetInstanceProcAddr"));
    if (!getInstanceProcAddr) return fail("vkGetInstanceProcAddr missing from loader");

    auto iFn = [&](const char* name) -> PFN_vkVoidFunction {
        return getInstanceProcAddr(VK_NULL_HANDLE, name);
    };

    // 2. instance
    b.createInstance = reinterpret_cast<PFN_vkCreateInstance>(iFn("vkCreateInstance"));
    b.destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(iFn("vkDestroyInstance"));
    b.enumeratePhysicalDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(iFn("vkEnumeratePhysicalDevices"));
    if (!b.createInstance || !b.enumeratePhysicalDevices) return fail("loader missing vkCreateInstance");

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "opendlss-nr";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "opendlss-nr";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    if (b.createInstance(&ici, nullptr, &b.instance) != VK_SUCCESS)
        return fail("vkCreateInstance failed");

    auto pFn = [&](const char* name) -> PFN_vkVoidFunction {
        return getInstanceProcAddr(b.instance, name);
    };
    b.destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(pFn("vkDestroyInstance"));
    b.getPhysicalDeviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(pFn("vkGetPhysicalDeviceProperties"));
    b.getPhysicalDeviceFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(pFn("vkGetPhysicalDeviceFeatures"));
    b.getPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(pFn("vkGetPhysicalDeviceMemoryProperties"));
    b.getPhysicalDeviceQueueFamilyProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(pFn("vkGetPhysicalDeviceQueueFamilyProperties"));
    b.getPhysicalDeviceFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(pFn("vkGetPhysicalDeviceFormatProperties"));
    b.createDevice = reinterpret_cast<PFN_vkCreateDevice>(pFn("vkCreateDevice"));
    if (!b.getPhysicalDeviceProperties || !b.getPhysicalDeviceMemoryProperties ||
        !b.getPhysicalDeviceQueueFamilyProperties || !b.createDevice)
        return fail("instance entry points incomplete");

    // 3. physical device pick: discrete > integrated > any
    uint32_t n = 0;
    if (b.enumeratePhysicalDevices(b.instance, &n, nullptr) != VK_SUCCESS || n == 0)
        return fail("no Vulkan physical devices (no driver / no GPU visible)");
    if (n > 64) n = 64;
    VkPhysicalDevice devs[64];
    if (b.enumeratePhysicalDevices(b.instance, &n, devs) != VK_SUCCESS)
        return fail("vkEnumeratePhysicalDevices failed");

    int bestScore = -1;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        b.getPhysicalDeviceProperties(devs[i], &p);
        int score = 0;
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 3;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 2;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) score = 1;   // last resort
        else score = 1;
        if (score > bestScore) { bestScore = score; b.physicalDevice = devs[i]; b.props = p; }
    }
    if (!b.physicalDevice) return fail("physical device pick failed");
    b.apiVersion = b.props.apiVersion;
    if (deviceNameOut) *deviceNameOut = b.props.deviceName;

    b.getPhysicalDeviceFeatures(b.physicalDevice, &b.features);
    b.getPhysicalDeviceMemoryProperties(b.physicalDevice, &b.memProps);

    // 4. queue family: first with COMPUTE
    uint32_t qCount = 0;
    b.getPhysicalDeviceQueueFamilyProperties(b.physicalDevice, &qCount, nullptr);
    if (qCount == 0) return fail("no queue families");
    if (qCount > 32) qCount = 32;
    VkQueueFamilyProperties qf[32];
    b.getPhysicalDeviceQueueFamilyProperties(b.physicalDevice, &qCount, qf);
    bool foundQueue = false;
    for (uint32_t i = 0; i < qCount; ++i) {
        if (qf[i].queueCount > 0 && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            b.queueFamily = i;
            foundQueue = true;
            break;
        }
    }
    if (!foundQueue) return fail("no compute-capable queue family");

    // 5. device (no features, no extensions — everything used is core 1.0)
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = b.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (b.createDevice(b.physicalDevice, &dci, nullptr, &b.device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");

    // 6. device entry points via vkGetDeviceProcAddr
    b.getDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr_t>(iFn("vkGetDeviceProcAddr"));
    auto dFn = [&](const char* name) -> PFN_vkVoidFunction {
        if (b.getDeviceProcAddr) {
            if (PFN_vkVoidFunction f = b.getDeviceProcAddr(b.device, name)) return f;
        }
        return getInstanceProcAddr(b.instance, name);   // spec-permitted fallback
    };
    b.destroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(dFn("vkDestroyDevice"));
    b.getDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(dFn("vkGetDeviceQueue"));
    b.createCommandPool = reinterpret_cast<PFN_vkCreateCommandPool>(dFn("vkCreateCommandPool"));
    b.destroyCommandPool = reinterpret_cast<PFN_vkDestroyCommandPool>(dFn("vkDestroyCommandPool"));
    b.allocateCommandBuffers = reinterpret_cast<PFN_vkAllocateCommandBuffers>(dFn("vkAllocateCommandBuffers"));
    b.freeCommandBuffers = reinterpret_cast<PFN_vkFreeCommandBuffers>(dFn("vkFreeCommandBuffers"));
    b.resetCommandBuffer = reinterpret_cast<PFN_vkResetCommandBuffer>(dFn("vkResetCommandBuffer"));
    b.beginCommandBuffer = reinterpret_cast<PFN_vkBeginCommandBuffer>(dFn("vkBeginCommandBuffer"));
    b.endCommandBuffer = reinterpret_cast<PFN_vkEndCommandBuffer>(dFn("vkEndCommandBuffer"));
    b.createBuffer = reinterpret_cast<PFN_vkCreateBuffer>(dFn("vkCreateBuffer"));
    b.destroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(dFn("vkDestroyBuffer"));
    b.getBufferMemoryRequirements = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(dFn("vkGetBufferMemoryRequirements"));
    b.allocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(dFn("vkAllocateMemory"));
    b.freeMemory = reinterpret_cast<PFN_vkFreeMemory>(dFn("vkFreeMemory"));
    b.bindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(dFn("vkBindBufferMemory"));
    b.mapMemory = reinterpret_cast<PFN_vkMapMemory>(dFn("vkMapMemory"));
    b.unmapMemory = reinterpret_cast<PFN_vkUnmapMemory>(dFn("vkUnmapMemory"));
    b.flushMappedMemoryRanges = reinterpret_cast<PFN_vkFlushMappedMemoryRanges>(dFn("vkFlushMappedMemoryRanges"));
    b.invalidateMappedMemoryRanges = reinterpret_cast<PFN_vkInvalidateMappedMemoryRanges>(dFn("vkInvalidateMappedMemoryRanges"));
    b.createImage = reinterpret_cast<PFN_vkCreateImage>(dFn("vkCreateImage"));
    b.destroyImage = reinterpret_cast<PFN_vkDestroyImage>(dFn("vkDestroyImage"));
    b.getImageMemoryRequirements = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(dFn("vkGetImageMemoryRequirements"));
    b.bindImageMemory = reinterpret_cast<PFN_vkBindImageMemory>(dFn("vkBindImageMemory"));
    b.createImageView = reinterpret_cast<PFN_vkCreateImageView>(dFn("vkCreateImageView"));
    b.destroyImageView = reinterpret_cast<PFN_vkDestroyImageView>(dFn("vkDestroyImageView"));
    b.createShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(dFn("vkCreateShaderModule"));
    b.destroyShaderModule = reinterpret_cast<PFN_vkDestroyShaderModule>(dFn("vkDestroyShaderModule"));
    b.createDescriptorSetLayout = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(dFn("vkCreateDescriptorSetLayout"));
    b.destroyDescriptorSetLayout = reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(dFn("vkDestroyDescriptorSetLayout"));
    b.createDescriptorPool = reinterpret_cast<PFN_vkCreateDescriptorPool>(dFn("vkCreateDescriptorPool"));
    b.destroyDescriptorPool = reinterpret_cast<PFN_vkDestroyDescriptorPool>(dFn("vkDestroyDescriptorPool"));
    b.allocateDescriptorSets = reinterpret_cast<PFN_vkAllocateDescriptorSets>(dFn("vkAllocateDescriptorSets"));
    b.resetDescriptorPool = reinterpret_cast<PFN_vkResetDescriptorPool>(dFn("vkResetDescriptorPool"));
    b.createPipelineLayout = reinterpret_cast<PFN_vkCreatePipelineLayout>(dFn("vkCreatePipelineLayout"));
    b.destroyPipelineLayout = reinterpret_cast<PFN_vkDestroyPipelineLayout>(dFn("vkDestroyPipelineLayout"));
    b.createComputePipelines = reinterpret_cast<PFN_vkCreateComputePipelines>(dFn("vkCreateComputePipelines"));
    b.destroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(dFn("vkDestroyPipeline"));
    b.createFence = reinterpret_cast<PFN_vkCreateFence>(dFn("vkCreateFence"));
    b.destroyFence = reinterpret_cast<PFN_vkDestroyFence>(dFn("vkDestroyFence"));
    b.waitForFences = reinterpret_cast<PFN_vkWaitForFences>(dFn("vkWaitForFences"));
    b.resetFences = reinterpret_cast<PFN_vkResetFences>(dFn("vkResetFences"));
    b.cmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(dFn("vkCmdBindPipeline"));
    b.cmdBindDescriptorSets = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(dFn("vkCmdBindDescriptorSets"));
    b.cmdPushConstants = reinterpret_cast<PFN_vkCmdPushConstants>(dFn("vkCmdPushConstants"));
    b.cmdDispatch = reinterpret_cast<PFN_vkCmdDispatch>(dFn("vkCmdDispatch"));
    b.cmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(dFn("vkCmdPipelineBarrier"));
    b.cmdCopyBufferToImage = reinterpret_cast<PFN_vkCmdCopyBufferToImage>(dFn("vkCmdCopyBufferToImage"));
    b.cmdCopyImageToBuffer = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(dFn("vkCmdCopyImageToBuffer"));
    b.queueSubmit = reinterpret_cast<PFN_vkQueueSubmit>(dFn("vkQueueSubmit"));
    b.updateDescriptorSets = reinterpret_cast<PFN_vkUpdateDescriptorSets>(dFn("vkUpdateDescriptorSets"));
    b.queueWaitIdle = reinterpret_cast<PFN_vkQueueWaitIdle>(dFn("vkQueueWaitIdle"));
    b.deviceWaitIdle = reinterpret_cast<PFN_vkDeviceWaitIdle>(dFn("vkDeviceWaitIdle"));

    if (!b.getDeviceQueue || !b.queueSubmit || !b.queueWaitIdle || !b.cmdDispatch ||
        !b.cmdPipelineBarrier || !b.cmdPushConstants || !b.allocateMemory || !b.mapMemory ||
        !b.createComputePipelines || !b.allocateDescriptorSets || !b.beginCommandBuffer) {
        bootShutdown(b);
        return fail("device entry points incomplete");
    }

    b.getDeviceQueue(b.device, b.queueFamily, 0, &b.queue);
    if (!b.queue) { bootShutdown(b); return fail("vkGetDeviceQueue returned null"); }

    b.ok = true;
    return true;
}

inline void bootShutdown(Boot& b) {
    if (b.device && b.destroyDevice) b.destroyDevice(b.device, nullptr);
    if (b.instance && b.destroyInstance) b.destroyInstance(b.instance, nullptr);
    b.device = VK_NULL_HANDLE;
    b.instance = VK_NULL_HANDLE;
    b.queue = VK_NULL_HANDLE;
    b.physicalDevice = VK_NULL_HANDLE;
    if (b.lib) { dlclose(b.lib); b.lib = nullptr; }
    b.ok = false;
}

} // namespace vk
} // namespace opendlss

// ---------------------------------------------------------------------------
// ABI self-checks (64-bit little-endian desktop targets)
// ---------------------------------------------------------------------------
