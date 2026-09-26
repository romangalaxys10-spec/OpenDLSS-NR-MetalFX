// nr_frame_metal.mm — Metal instantiation of the shared frame session
// (Apple Silicon path: Metal + MetalFX, macOS 13+).
#include "nr_frame.h"
#include "nr_numeric.h"
#import "nr_metal.h"
#include "nr_frame_impl.inc"

namespace nr {
namespace frame {

struct MtlBackend {
  using Context = nr::metal::Context;
  using Model = nr::metal::Model;
  using Kernels = nr::metal::Kernels;
  using Graph = nr::metal::Graph;
  using Activation = nr::metal::Activation;
  using Geometry = nr::metal::Geometry;
  using Format = nr::metal::Format;
  using PreprocessArgs = nr::metal::PreprocessArgs;
  struct FormatF32 {};
  struct GraphOptions {};

  static std::unique_ptr<Context> makeContext() { return std::make_unique<Context>(); }
  static std::unique_ptr<Model> makeModel(Context& context, const std::string& dir, bool verify) {
    return std::make_unique<Model>(context, dir, verify);
  }
  static std::unique_ptr<Kernels> makeKernels(Context& context) { return std::make_unique<Kernels>(context); }
  static std::unique_ptr<Graph> makeGraph(Context& context, Model& model, Kernels& kernels,
                                          const Geometry& geometry, GraphOptions) {
    typename Graph::Options options;
    return std::make_unique<Graph>(context, model, kernels, geometry, options);
  }
  static void loadShaders(Context& context, Kernels&) {
    // Preferred: the precompiled metallib shipped next to the binary; dev
    // fallback: compile the .metal sources in src/metal/shaders.
    namespace fs = std::filesystem;
    for (const char* candidate : {"opendlss.metallib", "shaders/opendlss.metallib"}) {
      std::error_code ec;
      if (fs::exists(candidate, ec)) {
        context.loadLibrary(candidate);
        return;
      }
    }
    // Dev fallback: concatenate the kernel sources into one runtime library.
    std::string source;
    for (const char* name : {"nr_common.metal", "ops.metal", "preprocess.metal", "gemm_f16.metal", "gemm_reduce.metal",
                             "gemm_fp8.metal", "window_normalize.metal", "window_attend.metal",
                             "global_normalize.metal", "global_attend.metal"}) {
      std::ifstream file(std::string(NR_METAL_SHADER_DIR) + "/" + name, std::ios::binary);
      if (!file) throw std::runtime_error(std::string("missing shader source: ") + name);
      std::stringstream stream;
      stream << file.rdbuf();
      source += stream.str() + "\n";
    }
    context.loadSource(source, "opendlss-runtime");
  }
  static void preprocess(Kernels& kernels, id<MTLCommandBuffer> commands, const Activation& proxy,
                         Activation& features, const PreprocessArgs& args) {
    kernels.preprocessFromProxy(commands, proxy, features, args);
  }
  static std::unique_ptr<Activation> allocate(Context& context, const char* label, uint32_t rows, uint32_t channels,
                                              FormatF32) {
    auto activation = std::make_unique<Activation>();
    activation->buffer = context.createBuffer((uint64_t)nr::metal::alignRows(rows) * channels * 4, label);
    activation->format = Format::F32;
    activation->rows = rows;
    activation->channels = channels;
    activation->allocRows = nr::metal::alignRows(rows);
    return activation;
  }
  static id<MTLBuffer> buffer(const Activation& activation) { return activation.buffer; }
  static void upload(Context& context, id<MTLBuffer> buffer, const void* data, uint64_t bytes) {
    context.upload(buffer, data, bytes);
  }
  static std::vector<uint8_t> download(Context& context, id<MTLBuffer> buffer, uint64_t bytes) {
    return context.download(buffer, bytes);
  }
  static id<MTLCommandBuffer> beginCommands(Context& context) { return context.beginCommands(); }
  static void endAndSubmit(Context& context, id<MTLCommandBuffer> commands, bool wait) {
    context.endAndSubmit(commands, wait);
  }
  static const Activation& head(Graph& graph) { return graph.head(); }
  static std::string deviceName(Context& context) { return context.deviceName(); }
};

std::unique_ptr<Session> createSessionMetal(const SessionOptions& options) {
  return std::make_unique<SessionImpl<MtlBackend>>(options);
}

}  // namespace frame
}  // namespace nr
