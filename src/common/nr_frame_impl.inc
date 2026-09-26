// nr_frame_impl.inc — backend-generic NR frame session implementation.
// Included by nr_frame_vulkan.cpp (Vulkan backend) and nr_frame_metal.mm
// (Metal backend); B is the backend adapter tag (see below in each includer).
//
// The two backends expose the same API shape by design (Context/Model/Kernels/
// Graph, Activation with buffer/rows/channels/format), so this template carries
// the whole product pipeline once.

#include <cstring>
#include <fstream>

namespace nr {
namespace frame {

template <typename B>
class SessionImpl final : public Session {
 public:
  SessionImpl(const SessionOptions& options) : options_(options), params_(options.params) {
    using GeometryOfB = typename B::Geometry;
    geometry_ = GeometryOfB::fromValid(params_.validWidth, params_.validHeight);
    context_ = B::makeContext();
    model_ = B::makeModel(*context_, options.modelDirectory, /*verifyHashes=*/true);
    kernels_ = B::makeKernels(*context_);
    B::loadShaders(*context_, *kernels_);
    graph_ = B::makeGraph(*context_, *model_, *kernels_, geometry_, typename B::GraphOptions{});
    const uint32_t fullRows = geometry_.fullWidth * geometry_.fullHeight;
    features_ = B::allocate(*context_, "input features", fullRows, 16, typename B::FormatF32{});
    proxy_ = B::allocate(*context_, "proxy", params_.sourceWidth * params_.sourceHeight, 4,
                         typename B::FormatF32{});
    typename B::PreprocessArgs args;
    args.fullWidth = geometry_.fullWidth;
    args.fullHeight = geometry_.fullHeight;
    args.validWidth = params_.validWidth;
    args.validHeight = params_.validHeight;
    args.sourceWidth = params_.sourceWidth;
    args.sourceHeight = params_.sourceHeight;
    args.seed = params_.seed;
    args.autoMask = params_.autoMask;
    args.localTone = params_.localTone;
    args.localStructure = params_.localStructure;
    args.skinStructure = params_.skinStructure;
    args.style = params_.style;
    preprocess_ = args;
    deviceName_ = B::deviceName(*context_);
  }

  void processFrame(const float* proxy, uint32_t proxyStridePixels, std::vector<uint8_t>& out,
                    SessionStats* stats) override {
    auto commands = B::beginCommands(*context_);
    const uint32_t srcPixels = params_.sourceWidth * params_.sourceHeight;
    // Pack the proxy rows (stride-aware) into the [pixels][4] buffer.
    if (proxyStridePixels == params_.sourceWidth) {
      B::upload(*context_, B::buffer(*proxy_), proxy, (uint64_t)srcPixels * 4 * sizeof(float));
    } else {
      std::vector<float> packed((size_t)srcPixels * 4);
      for (uint32_t y = 0; y < params_.sourceHeight; ++y)
        memcpy(&packed[(size_t)y * params_.sourceWidth * 4],
               proxy + (size_t)y * proxyStridePixels * 4, params_.sourceWidth * 4 * sizeof(float));
      B::upload(*context_, B::buffer(*proxy_), packed.data(), (uint64_t)srcPixels * 4 * sizeof(float));
    }
    B::preprocess(*kernels_, commands, *proxy_, *features_, preprocess_);
    graph_->record(commands, *features_);
    B::endAndSubmit(*context_, commands, /*wait=*/true);

    const uint32_t fullRows = geometry_.fullWidth * geometry_.fullHeight;
    std::vector<uint8_t> headBytes = B::download(*context_, B::buffer(B::head(*graph_)), (uint64_t)fullRows * 16);
    const float* head = reinterpret_cast<const float*>(headBytes.data());
    // Centred proxy lanes (4..6) recomputed on the host, bit-exact with the kernel.
    std::vector<float> composed((size_t)params_.validWidth * params_.validHeight * 4);
    for (uint32_t y = 0; y < params_.validHeight; ++y) {
      for (uint32_t x = 0; x < params_.validWidth; ++x) {
        const uint32_t sx = x < params_.validWidth ? x : 2 * params_.validWidth - x - 2;
        const uint32_t sy = y < params_.validHeight ? y : 2 * params_.validHeight - y - 2;
        const uint32_t ix = ((2u * sx + 1u) * params_.sourceWidth) / (2u * params_.validWidth);
        const uint32_t iy = ((2u * sy + 1u) * params_.sourceHeight) / (2u * params_.validHeight);
        const float* proxyPixel = proxy + ((size_t)iy * proxyStridePixels + ix) * 4;
        const size_t outIndex = ((size_t)y * params_.validWidth + x) * 4;
        for (uint32_t c = 0; c < 3; ++c) {
          float value = num::roundF16(num::roundF16(num::roundF16(proxyPixel[c]) - 0.5f) * 0.125f);
          composed[outIndex + c] = composedValue(head[((size_t)y * geometry_.fullWidth + x) * 4 + c], value);
        }
        composed[outIndex + 3] = proxyPixel[3];
      }
    }
    packRgba8(composed.data(), (size_t)params_.validWidth * params_.validHeight, out);
    if (stats) {
      stats->dispatches = kernels_->dispatchCount();
      stats->gpuMilliseconds = 0.0;   // timing lives in the bench/profile commands
      stats->device = deviceName_;
    }
  }

  const Params& params() const override { return params_; }

 private:
  SessionOptions options_;
  Params params_;
  typename B::Geometry geometry_;
  std::unique_ptr<typename B::Context> context_;
  std::unique_ptr<typename B::Model> model_;
  std::unique_ptr<typename B::Kernels> kernels_;
  std::unique_ptr<typename B::Graph> graph_;
  std::unique_ptr<typename B::Activation> features_;
  std::unique_ptr<typename B::Activation> proxy_;
  typename B::PreprocessArgs preprocess_{};
  std::string deviceName_;
};

}  // namespace frame
}  // namespace nr
