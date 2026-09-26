// nr_metal_context.mm — Metal device/queue/buffer/pipeline host (the twin of
// src/vulkan/vk_context.cpp) plus the geometry and tensor-layout helpers that
// the graph and the model share with the Vulkan port.
#import "nr_metal.h"

#include <algorithm>
#include <cmath>

namespace nr {
namespace metal {

// ---------------------------------------------------------------------------
// Geometry (identical to the Vulkan port; see docs/original-design/network.md)
// ---------------------------------------------------------------------------

static uint32_t alignUp(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Every level halves its input and rounds the result up to 4; the decoder
// doubles the chain back up. A dimension therefore needs enough headroom for
// all of those halvings to be exact, which is what the padded ("full") field
// provides: it is aligned to two to the power of the number of size reductions
// the graph makes on that axis.
static uint32_t fieldAlignment(uint32_t valid) {
  uint32_t reductions = 0, size = valid;
  for (int level = 0; level < 6; ++level) {
    const uint32_t half = alignUp((size + 1) / 2, 4);
    if (half < size) ++reductions;
    if (level == 0 && half % 8 != 0) ++reductions;
    size = half;
  }
  return 1u << reductions;
}

Geometry Geometry::fromValid(uint32_t validWidth, uint32_t validHeight) {
  Geometry g;
  g.validWidth = validWidth;
  g.validHeight = validHeight;
  const uint32_t alignWidth = fieldAlignment(validWidth), alignHeight = fieldAlignment(validHeight);
  g.fullWidth = std::max(320u, alignUp(validWidth, alignWidth));
  g.fullHeight = std::max(320u, alignUp(validHeight, alignHeight));
  // One more alignment step on the width when both axes are a multiple of four
  // alignments. It is native's rule; it has to be reproduced because the field
  // size decides the window grid and therefore the result inside the valid
  // rectangle as well.
  if (g.fullWidth % (4 * alignWidth) == 0 && g.fullHeight % (4 * alignHeight) == 0) g.fullWidth += alignWidth;
  uint32_t width = g.fullWidth, height = g.fullHeight;
  for (int level = 0; level < 6; ++level) {
    width = alignUp((width + 1) / 2, 4);
    height = alignUp((height + 1) / 2, 4);
    g.levels[level] = {width, height};
  }
  if (g.levels[0].width % 8 || g.levels[0].height % 8)
    throw std::runtime_error(
        "unsupported size " + std::to_string(validWidth) + "x" + std::to_string(validHeight) +
        ": level 0 is not a whole number of 8-pixel windows; use at least 33 pixels on each axis");
  return g;
}

void windowPhase(uint32_t index, uint32_t& shiftX, uint32_t& shiftY) {
  static const uint32_t phases[4][2] = {{0, 0}, {4, 4}, {4, 0}, {0, 4}};
  shiftX = phases[index & 3][0];
  shiftY = phases[index & 3][1];
}

static uint32_t standardHidden(uint32_t channels) {
  if (channels == 32 || channels == 64 || channels == 128 || channels == 256) return 128;
  throw std::runtime_error("no fused layout for " + std::to_string(channels) + " channels");
}

FusedLayout fusedLayout(uint32_t channels) {
  FusedLayout l;
  l.hidden = standardHidden(channels);
  l.heads = channels / 32;
  l.expand = 0;
  l.expertFfn = channels >= 64;
  l.expertCount = l.expertFfn ? channels / 32 : 0;
  uint32_t expandBytes = l.expertFfn ? l.expertCount * channels * 128 : channels * l.hidden;
  l.contractWeights = expandBytes;
  uint32_t ffnWeightBytes = l.expertFfn ? expandBytes + l.expertCount * 128 * 32 + l.expertCount * 32 * channels
                                        : expandBytes + l.hidden * channels;
  l.ffnCosSkip = ffnWeightBytes + 16;
  l.qkv = l.ffnCosSkip + channels * 2 + 16;
  l.relative = l.qkv + channels * channels * 3;
  l.scale = l.relative + l.heads * 8192;
  l.projection = l.scale + alignUp(l.heads * 4, 16);
  l.attnCosSkip = l.projection + channels * channels;
  l.endWithoutPadding = l.attnCosSkip + channels * 2;
  return l;
}

FusedLayout preFusedLayout() {
  FusedLayout l;
  l.hidden = 128; l.heads = 1;
  l.expand = 0; l.contractWeights = 4096; l.inputAdapter = 8208; l.ffnCosSkip = 9232; l.qkv = 9312;
  l.relative = 12384; l.scale = 20576; l.projection = 20592; l.attnCosSkip = 21616; l.endWithoutPadding = 21680;
  return l;
}

FusedLayout upsampleFusedLayout(uint32_t inputChannels, uint32_t channels) {
  if (inputChannels != channels * 2) throw std::runtime_error("upsample layout expects 2x input channels");
  FusedLayout l;
  l.hidden = standardHidden(channels);
  l.heads = channels / 32;
  uint32_t narrowPadding = channels == 32 ? 16 : 0;
  l.expand = 0;
  l.expertFfn = channels >= 64;
  l.expertCount = l.expertFfn ? channels / 32 : 0;
  uint32_t expandBytes = l.expertFfn ? l.expertCount * channels * 128 : channels * l.hidden;
  l.contractWeights = expandBytes;
  uint32_t ffnWeightBytes = l.expertFfn ? expandBytes + l.expertCount * 128 * 32 + l.expertCount * 32 * channels
                                        : expandBytes + l.hidden * channels;
  l.upsampleWeight = ffnWeightBytes;
  l.ffnCosSkip = l.upsampleWeight + inputChannels * channels + narrowPadding;
  l.transitionScale = l.ffnCosSkip + channels * 2 + narrowPadding;
  l.qkv = l.transitionScale + channels * 2;
  l.relative = l.qkv + channels * channels * 3;
  l.scale = l.relative + l.heads * 8192;
  l.projection = l.scale + alignUp(l.heads * 4, 16);
  l.attnCosSkip = l.projection + channels * channels;
  l.endWithoutPadding = l.attnCosSkip + channels * 2;
  return l;
}

FusedLayout postFusedLayout() {
  FusedLayout l;
  l.hidden = 128; l.heads = 1;
  l.expand = 0; l.contractWeights = 4096; l.ffnCosSkip = 8208; l.inputScale = 8272; l.adapterScale = 8336;
  l.qkv = 8400; l.relative = 11472; l.scale = 19664; l.projection = 19680; l.attnCosSkip = 20704;
  l.postWeights = 20784; l.endWithoutPadding = 21808;
  return l;
}

// Packed index permutations (identical to src/vulkan/nr_model.cpp).
uint32_t packedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u;
  uint32_t within = k & 31u;
  uint32_t half = within & 16u;
  uint32_t quarter = within & 15u;
  return base + half + (quarter >> 2) * 2 + (quarter & 1) + (((quarter & 2) != 0) ? 8 : 0);
}

uint32_t inversePackedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u;
  uint32_t within = k & 31u;
  return base + (within & 17u) + ((within & 2u) << 1) + ((within & 4u) << 1) + ((within & 8u) >> 2);
}

uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels) {
  uint32_t kTile = k >> 5, kIn = k & 31;
  uint32_t nTile = n >> 7, nIn = n & 127;
  uint32_t nHalf = nIn >> 6, nGroup = (nIn & 63) >> 4, nInGroup = nIn & 15;
  uint32_t lane = ((nInGroup & 7) << 2) | ((kIn & 15) >> 2);
  uint32_t byteInLane = ((nInGroup >> 3) << 3) | ((kIn >> 4) << 2) | (kIn & 3);
  return kTile * outputChannels * 32 + nTile * 4096 + nHalf * 2048 + nGroup * 512 + lane * 16 + byteInLane;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

Context::Context() {
  device_ = MTLCreateSystemDefaultDevice();
  NR_CHECK(device_ != nil, "no Metal device (macOS 13+ on Apple silicon required)");
  queue_ = [device_ newCommandQueue];
  NR_CHECK(queue_ != nil, "MTLCommandQueue creation failed");
  deviceName_ = [device_.name UTF8String];
  coreCount_ = 0;
  if (@available(macOS 10.15, *)) {
    // Best-effort GPU core estimate from the registry ID (M1: 8/16, M2: 10/16/24...).
    // Not required for correctness; the graph never sizes work off it.
  }
}

Context::~Context() = default;

id<MTLBuffer> Context::createBuffer(uint64_t size, const char* label) {
  id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
  NR_CHECK(buffer != nil, (std::string("buffer allocation failed: ") + (label ? label : "?")).c_str());
  if (label) buffer.label = [NSString stringWithUTF8String:label];
  return buffer;
}

void Context::upload(id<MTLBuffer> buffer, const void* data, uint64_t size) {
  NR_CHECK(buffer.length >= size, "upload exceeds buffer");
  memcpy(buffer.contents, data, size);
  if (buffer.storageMode == MTLStorageModeManaged) [buffer didModifyRange:NSMakeRange(0, size)];
}

std::vector<uint8_t> Context::download(id<MTLBuffer> buffer, uint64_t size, uint64_t offset) {
  std::vector<uint8_t> bytes((size_t)size);
  memcpy(bytes.data(), (const uint8_t*)buffer.contents + offset, (size_t)size);
  return bytes;
}

void Context::loadLibrary(const std::string& metallibPath) {
  NSError* error = nil;
  NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallibPath.c_str()]];
  library_ = [device_ newLibraryWithURL:url error:&error];
  if (!library_ && error) {
    throw std::runtime_error(std::string("metallib load failed: ") + error.localizedDescription.UTF8String);
  }
  NR_CHECK(library_ != nil, "metallib load failed");
}

void Context::loadSource(const std::string& source, const std::string& sourceName) {
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  options.languageVersion = MTLLanguageVersion3_1;
  library_ = [device_ newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                   options:options
                                     error:&error];
  if (!library_) {
    std::string detail = error ? error.localizedDescription.UTF8String : "unknown";
    throw std::runtime_error("shader compile failed (" + sourceName + "): " + detail);
  }
}

id<MTLFunction> Context::function(const std::string& name, const std::vector<uint32_t>& constantValues,
                                  const std::vector<std::string>& constantNames) {
  std::string key = name;
  for (uint32_t v : constantValues) key += ":" + std::to_string(v);
  auto it = functions_.find(key);
  if (it != functions_.end()) return it->second;

  id<MTLFunction> function;
  if (constantNames.empty()) {
    function = [library_ newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  } else {
    MTLFunctionConstantValues* values = [[MTLFunctionConstantValues alloc] init];
    for (size_t i = 0; i < constantNames.size(); ++i) {
      MTLDataType type = MTLDataTypeUInt;
      [values setConstantValue:&constantValues[i] type:type atIndex:i];
    }
    NSError* error = nil;
    function = [library_ newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                             constantValues:values
                                      error:&error];
    if (!function) {
      std::string detail = error ? error.localizedDescription.UTF8String : "unknown";
      throw std::runtime_error("function constants mismatch for " + name + ": " + detail);
    }
  }
  NR_CHECK(function != nil, ("missing kernel " + key).c_str());
  functions_[key] = function;
  return function;
}

id<MTLComputePipelineState> Context::pipeline(const std::string& name, const std::vector<uint32_t>& constantValues,
                                              const std::vector<std::string>& constantNames) {
  std::string key = name;
  for (uint32_t v : constantValues) key += ":" + std::to_string(v);
  auto it = pipelines_.find(key);
  if (it != pipelines_.end()) return it->second;
  id<MTLFunction> fn = function(name, constantValues, constantNames);
  NSError* error = nil;
  id<MTLComputePipelineState> state = [device_ newComputePipelineStateWithFunction:fn error:&error];
  if (!state) {
    std::string detail = error ? error.localizedDescription.UTF8String : "unknown";
    throw std::runtime_error("pipeline creation failed for " + key + ": " + detail);
  }
  pipelines_[key] = state;
  return state;
}

id<MTLCommandBuffer> Context::beginCommands() { return [queue_ commandBuffer]; }

void Context::endAndSubmit(id<MTLCommandBuffer> commands, bool wait) {
  [commands commit];
  if (wait) [commands waitUntilCompleted];
}

void Context::waitIdle() {
  id<MTLCommandBuffer> commands = [queue_ commandBuffer];
  [commands commit];
  [commands waitUntilCompleted];
}

bool Context::metalFxAvailable() {
  if (@available(macOS 13.0, *)) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device && [device supportsFamily:MTLGPUFamilyApple7];
  }
  return false;
}

uint32_t Context::metalFxMajorVersion() {
  if (@available(macOS 14.0, *)) return 2;   // spatial + temporal
  if (@available(macOS 13.0, *)) return 1;   // spatial
  return 0;
}

}  // namespace metal
}  // namespace nr
