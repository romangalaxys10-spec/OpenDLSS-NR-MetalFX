// nr_frame_vulkan.cpp — Vulkan instantiation of the shared frame session
// (Windows + Linux path: the original bit-exact Vulkan route).
#include "nr_frame.h"
#include "nr_numeric.h"
#include "nr_frame_impl.inc"
#include "vk_context.h"
#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"

#include <filesystem>

namespace nr {
namespace frame {

struct VkBackend {
  using Context = ::vk::Context;
  using Model = nr::Model;
  using Kernels = nr::Kernels;
  using Graph = nr::Graph;
  using Activation = nr::Activation;
  using Geometry = nr::Geometry;
  using Format = nr::Format;
  using PreprocessArgs = nr::Kernels::PreprocessArgs;
  struct FormatF32 {};
  struct GraphOptions {};

  static std::unique_ptr<Context> makeContext() { return std::make_unique<Context>(); }
  static std::unique_ptr<Model> makeModel(Context& context, const std::string& dir, bool verify) {
    return std::make_unique<Model>(context, dir, verify);
  }
  static std::unique_ptr<Kernels> makeKernels(Context& context) {
    const char* dir = getenv("OPENDLSS_SHADERS");
    std::string shaderDir = dir ? dir : std::string(OPENDLSS_INSTALL_SHADERS);
    return std::make_unique<Kernels>(context, shaderDir);
  }
  static std::unique_ptr<Graph> makeGraph(Context& context, Model& model, Kernels& kernels,
                                          const Geometry& geometry, GraphOptions) {
    typename Graph::Options options;
    options.fusedBlocks = false;   // Metal-mirrored reference route; DLSS5VK_* env still tunes Vulkan
    return std::make_unique<Graph>(context, model, kernels, geometry, options);
  }
  static void preprocess(Kernels& kernels, VkCommandBuffer commands, const Activation& proxy,
                         Activation& features, const PreprocessArgs& args) {
    kernels.preprocessFromProxy(commands, proxy.buffer, features, args);
  }
  static void loadShaders(Context&, Kernels&) {}   // shaders load in makeKernels (constructor takes the .spv dir)
  static std::unique_ptr<Activation> allocate(Context& context, const char* label, uint32_t rows, uint32_t channels,
                                              FormatF32) {
    auto activation = std::make_unique<Activation>();
    activation->buffer = context.createBuffer((VkDeviceSize)nr::alignRows(rows) * channels * 4, false, label);
    activation->format = Format::F32;
    activation->rows = rows;
    activation->channels = channels;
    activation->allocRows = nr::alignRows(rows);
    return activation;
  }
  static const ::vk::Buffer& buffer(const Activation& activation) { return activation.buffer; }
  static void upload(Context& context, const ::vk::Buffer& buffer, const void* data, uint64_t bytes) {
    context.upload(buffer, data, bytes);
  }
  static std::vector<uint8_t> download(Context& context, const ::vk::Buffer& buffer, uint64_t bytes) {
    return context.download(buffer, bytes);
  }
  static VkCommandBuffer beginCommands(Context& context) { return context.beginCommands(); }
  static void endAndSubmit(Context& context, VkCommandBuffer commands, bool wait) {
    context.endAndSubmit(commands, wait);
  }
  static const Activation& head(Graph& graph) { return graph.head(); }
  static std::string deviceName(Context& context) { return context.deviceName(); }
};

std::unique_ptr<Session> createSessionVulkan(const SessionOptions& options) {
  return std::make_unique<SessionImpl<VkBackend>>(options);
}

}  // namespace frame
}  // namespace nr
