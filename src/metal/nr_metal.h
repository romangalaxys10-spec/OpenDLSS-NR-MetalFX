// nr_metal.h — Metal backend of OpenDLSS-NR-MetalFX (Apple Silicon, macOS 13+).
//
// The Metal twin of the original Vulkan host (src/vk_*): one device, one queue,
// shared-memory storage buffers, compute pipelines with function constants,
// and the exact operand contracts of the network kernels. The 71-block graph
// runs the reference (unfused) kernel decomposition with barrier-separated
// launches; every inter-kernel tensor is published through the same bit-exact
// E4M3/F16 rounding rules as the Vulkan route.
//
// MetalFX integration lives in MetalFxSpatial / MetalFxTemporal (this header).
#pragma once

#include <CoreFoundation/CoreFoundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define NR_CHECK(cond, message)                                                                    \
  do {                                                                                             \
    if (!(cond)) throw std::runtime_error(std::string("nr::metal: ") + (message));                  \
  } while (0)

namespace nr {
namespace metal {

// ---------------------------------------------------------------------------
// Types (mirrors of the Vulkan-side contracts)
// ---------------------------------------------------------------------------

enum class Format { E4, F16, F32 };

inline uint32_t formatBytes(Format format) { return format == Format::E4 ? 1 : format == Format::F16 ? 2 : 4; }
inline uint32_t alignRows(uint32_t rows) { return (rows + 63) & ~63u; }

// A [rows][channels] activation tensor. Rows are allocated padded to 64 so
// A-operand staging never reads past the buffer (same rule as the Vulkan port).
struct Activation {
  id<MTLBuffer> buffer;
  Format format = Format::E4;
  uint32_t rows = 0;
  uint32_t channels = 0;
  uint32_t allocRows = 0;
  std::string label;
  uint64_t validBytes() const { return (uint64_t)rows * channels * formatBytes(format); }
};

// A model tensor: bytes inside a packed stage + the raw aux upload.
struct Tensor {
  std::string name;
  int block = 0;
  int layer = 0;
  std::string parameter;
  std::string stage;
  uint32_t stageOffset = 0;
  uint32_t byteLength = 0;
  const uint8_t* bytes = nullptr;  // into the owning stage
  id<MTLBuffer> raw;               // raw packed bytes on the GPU (aux vectors, scales)
};

// Packed index permutations of the model layout (identical to the Vulkan port).
uint32_t packedInputIndex(uint32_t k);
uint32_t inversePackedInputIndex(uint32_t k);
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels);

struct Geometry {
  uint32_t validWidth = 0, validHeight = 0;
  uint32_t fullWidth = 0, fullHeight = 0;
  struct Level { uint32_t width, height; } levels[6];
  static Geometry fromValid(uint32_t validWidth, uint32_t validHeight);
  uint32_t vitTokens() const { return levels[5].width * levels[5].height; }
  uint32_t paddedVitTokens() const { return (vitTokens() + 63) & ~63u; }
};

struct FusedLayout {
  uint32_t hidden = 128, heads = 1;
  uint32_t expand = 0, contractWeights = 0, inputAdapter = 0, ffnCosSkip = 0, qkv = 0, relative = 0, scale = 0,
           projection = 0, attnCosSkip = 0, endWithoutPadding = 0;
  uint32_t upsampleWeight = 0, transitionScale = 0, inputScale = 0, adapterScale = 0, postWeights = 0;
  bool expertFfn = false;
  uint32_t expertCount = 0;
};
FusedLayout fusedLayout(uint32_t channels);
FusedLayout preFusedLayout();
FusedLayout upsampleFusedLayout(uint32_t inputChannels, uint32_t channels);
FusedLayout postFusedLayout();
void windowPhase(uint32_t index, uint32_t& shiftX, uint32_t& shiftY);

// ---------------------------------------------------------------------------
// Context: device, queue, buffers, pipelines
// ---------------------------------------------------------------------------

class Context {
 public:
  Context();
  ~Context();

  id<MTLDevice> device() const { return device_; }
  id<MTLCommandQueue> queue() const { return queue_; }
  const std::string& deviceName() const { return deviceName_; }
  uint32_t gpuCoreCount() const { return coreCount_; }

  // Buffers live in shared memory (Apple silicon is unified); StorageModeShared
  // on Apple silicon, Managed fallback elsewhere.
  id<MTLBuffer> createBuffer(uint64_t size, const char* label);
  void upload(id<MTLBuffer> buffer, const void* data, uint64_t size);
  std::vector<uint8_t> download(id<MTLBuffer> buffer, uint64_t size, uint64_t offset = 0);

  // Shader library + pipeline cache. `constants` mirrors the Vulkan
  // specialization-key scheme ("shader:c0:c1:...").
  void loadLibrary(const std::string& metallibPath);          // preferred: precompiled metallib
  void loadSource(const std::string& source, const std::string& sourceName);  // dev fallback
  id<MTLFunction> function(const std::string& name, const std::vector<uint32_t>& constantValues,
                           const std::vector<std::string>& constantNames);
  id<MTLComputePipelineState> pipeline(const std::string& name, const std::vector<uint32_t>& constantValues,
                                       const std::vector<std::string>& constantNames);

  id<MTLCommandBuffer> beginCommands();
  void endAndSubmit(id<MTLCommandBuffer> commands, bool wait = true);
  void computeBarrier(id<MTLCommandBuffer> commands) { [commands computeBarrier]; }
  void waitIdle();

  static bool metalFxAvailable();
  static uint32_t metalFxMajorVersion();

 private:
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::string deviceName_;
  uint32_t coreCount_ = 0;
  std::map<std::string, id<MTLFunction>> functions_;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;
};

// ---------------------------------------------------------------------------
// Model: manifest.json + packed E4M3 stage files, host re-layout
// ---------------------------------------------------------------------------

class Model {
 public:
  Model(Context& context, const std::string& directory, bool verifyHashes = true);
  ~Model();

  const Tensor& tensor(int block, int layer = 0, const std::string& parameter = "layer") const;

  // Plain [K][Nmatrix] E4M3 matrix for the MMA staging: [K / batchK][batchK / 32][Nmatrix][32]
  // (tileMajor) or [K / batchK][Nmatrix][batchK] (N-major). Identical byte math to the Vulkan port.
  id<MTLBuffer> fp8Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                          bool swizzleK = true, uint32_t batchK = 0, bool tileMajor = true);
  std::vector<uint8_t> fp8MatrixBytes(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                                      bool swizzleK, uint32_t batchK, bool tileMajor,
                                      const std::vector<uint32_t>* columnSource = nullptr);
  // fp8Matrix with the output columns permuted (PTX-style expert layouts).
  id<MTLBuffer> fp8MatrixPermuted(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                                  uint32_t batchK, const std::vector<uint32_t>& columnSource, const std::string& tag);
  // Plain [K][paddedN] f16 matrix (the pre adapter and the post head).
  id<MTLBuffer> f16Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t N, uint32_t& paddedN);
  // Learned 64x64 attention prior per head as f16 [heads][64 query][64 physical key].
  id<MTLBuffer> relativeBias(const Tensor& tensor, uint32_t relativeByteOffset, uint32_t heads);

  size_t nanWeightsReplaced() const { return nanWeights_; }
  uint32_t blockCount() const { return blockCount_; }

 private:
  struct Stage {
    std::string id;
    std::vector<uint8_t> bytes;
  };
  Context& context_;
  std::vector<Stage> stages_;
  std::map<std::string, Tensor> tensors_;
  std::map<std::string, id<MTLBuffer>> matrices_;
  size_t nanWeights_ = 0;
  uint32_t blockCount_ = 0;
};

inline uint16_t auxHalf(const Tensor& tensor, uint32_t byteOffset, uint32_t column) {
  const uint8_t* p = tensor.bytes + byteOffset + column * 2;
  return (uint16_t)(p[0] | (p[1] << 8));
}
inline float auxF32(const Tensor& tensor, uint32_t byteOffset) {
  float value;
  memcpy(&value, tensor.bytes + byteOffset, 4);
  return value;
}

// ---------------------------------------------------------------------------
// Kernels: dispatch helpers with the exact operand contracts of the network
// (the reference, unfused route; the fused megakernel routes remain available
// on the Vulkan/NVIDIA backend).
// ---------------------------------------------------------------------------

struct GemmFp8Args {
  const Activation* input = nullptr;
  uint32_t inputColumnBase = 0;
  id<MTLBuffer> weights;                 // plain [batches*K][Nmatrix] E4M3
  uint32_t Nmatrix = 0;
  uint32_t weightColumnOffset = 0;
  Activation* output = nullptr;          // F16 unless quantize (E4)
  uint32_t outputColumnOffset = 0;
  Activation* dualOutput = nullptr;      // E4 copy of the published value
  const Activation* residual = nullptr;
  bool scaleResidual = false;
  const Tensor* auxTensor = nullptr;
  uint32_t auxByteOffset = 0;
  bool silu = false;
  bool quantize = true;
  uint32_t rows = 0, K = 0, N = 0;
  uint32_t batches = 1;
  bool broadcastInput = false;
  uint32_t partition = 0;                // 0 or 256/512/1024 -> split-K (f16 partials, reduce pass)
};

struct GemmF16Args {
  const Activation* input = nullptr;     // F16
  id<MTLBuffer> weights;                 // plain [K][paddedN] f16
  uint32_t paddedN = 0;
  Activation* output = nullptr;          // F16 / E4 / F32 by format
  Activation* dualOutput = nullptr;      // E4
  uint32_t rows = 0, K = 0, N = 0;
};

struct PreprocessArgs {
  uint32_t fullWidth, fullHeight, validWidth, validHeight, sourceWidth, sourceHeight, seed;
  bool autoMask;
  float localTone, localStructure, skinStructure, style;
};

class Kernels {
 public:
  Kernels(Context& context);
  ~Kernels();

  void setSiluTable(const std::vector<uint16_t>& table);

  void gemmFp8(id<MTLCommandBuffer> commands, const GemmFp8Args& args);
  void gemmF16(id<MTLCommandBuffer> commands, const GemmF16Args& args);

  void preprocessFromProxy(id<MTLCommandBuffer> commands, const Activation& proxy, Activation& features,
                           const PreprocessArgs& args);
  void convertF32ToF16(id<MTLCommandBuffer> commands, const Activation& input, Activation& output);
  void quantize(id<MTLCommandBuffer> commands, const Activation& input, Activation& output);
  void downsample2x(id<MTLCommandBuffer> commands, const Activation& input, Activation& output, uint32_t inWidth,
                    uint32_t inHeight, uint32_t outWidth, uint32_t outHeight);
  void postBlend(id<MTLCommandBuffer> commands, const Activation& upsampleSource, const Activation& adapter,
                 const Tensor& tensor, uint32_t inputScaleByteOffset, uint32_t adapterScaleByteOffset,
                 Activation& rawOutput, Activation& quantizedOutput, uint32_t inWidth, uint32_t inHeight,
                 uint32_t outWidth, uint32_t outHeight);
  void upsampleResidual(id<MTLCommandBuffer> commands, const Activation& projection, const Activation& skip,
                        const Tensor& tensor, uint32_t scaleByteOffset, Activation& output, Activation* rawOutput,
                        uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight);

  void windowNormalize(id<MTLCommandBuffer> commands, const Activation& qkv, const Tensor& tensor,
                       uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads);
  void windowAttend(id<MTLCommandBuffer> commands, const Activation& normalized, id<MTLBuffer> prior,
                    Activation& attended, uint32_t width, uint32_t height, uint32_t heads, uint32_t shiftX,
                    uint32_t shiftY);
  void globalNormalize(id<MTLCommandBuffer> commands, const Activation& qkv, const Tensor& tensor,
                       uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads);
  void globalAttend(id<MTLCommandBuffer> commands, const Activation& normalized, Activation& attended, uint32_t tokens,
                    uint32_t paddedTokens, uint32_t heads);

  uint32_t dispatchCount() const { return dispatches_; }
  void resetDispatchCount() { dispatches_ = 0; }

 private:
  void dispatchOps(id<MTLCommandBuffer> commands, uint32_t mode, const std::vector<id<MTLBuffer>>& buffers,
                   uint32_t count, uint32_t channels, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth,
                   uint32_t outHeight, uint32_t auxOffsetA, uint32_t auxOffsetB, uint32_t dual);

 private:
  struct Op {  // one encoded dispatch (bound once, encoded fast)
    id<MTLComputePipelineState> state;
    std::vector<id<MTLBuffer>> buffers;
    std::vector<uint64_t> bufferOffsets;
    uint8_t push[128];
    uint32_t pushBytes = 0;
    uint32_t groups[3] = {0, 0, 0};
  };
  void dispatchLinear(id<MTLCommandBuffer> commands, const char* kernel, const std::vector<uint32_t>& constants,
                      const std::vector<std::string>& constantNames, const std::vector<id<MTLBuffer>>& buffers,
                      const void* push, uint32_t pushBytes, uint32_t count);
  void dispatchGrid(id<MTLCommandBuffer> commands, const char* kernel, const std::vector<uint32_t>& constants,
                    const std::vector<std::string>& constantNames, const std::vector<id<MTLBuffer>>& buffers,
                    const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z);
  id<MTLComputePipelineState> pipelineFor(const char* kernel, const std::vector<uint32_t>& constants,
                                          const std::vector<std::string>& constantNames);

  Context& context_;
  id<MTLBuffer> siluTable_;
  id<MTLBuffer> partialsBuffer_;   // fixed split-K scratch (recorded dispatches hold its address)
  uint32_t dispatches_ = 0;
  std::vector<uint8_t> pushScratch_;
};

// ---------------------------------------------------------------------------
// Graph: the 71-block network (reference decomposition, barrier-separated)
// ---------------------------------------------------------------------------

class Graph {
 public:
  static constexpr uint32_t kBlockCount = 71;

  struct Options {
    bool captureBoundaries = false;   // copy every block/transition output (parity checks)
    bool captureIntermediates = false;
  };
  Graph(Context& context, Model& model, Kernels& kernels, const Geometry& geometry, Options options);
  ~Graph();

  // Record the complete network. `inputFeatures` is f32 [fullWidth*fullHeight][16].
  void record(id<MTLCommandBuffer> commands, const Activation& inputFeatures);

  const Activation& head() const { return *head_; }   // f32 [full rows][4]
  const std::map<std::string, Activation*>& boundaries() const { return boundaries_; }
  const Geometry& geometry() const { return geometry_; }

 private:
  struct Temporaries {
    Activation* ffn = nullptr;            // E4 [rows][hidden]
    Activation* ffnNarrow = nullptr;      // E4 [rows][channels] (expert W2 outputs)
    Activation* ffnResidual = nullptr;    // F16 [rows][channels]
    Activation* ffnQuantized = nullptr;   // E4 [rows][channels]
    Activation* qkv = nullptr;            // F16 [rows][3*channels]
    Activation* normalized = nullptr;     // E4 [rows][3*channels]
    Activation* attended = nullptr;       // E4 [rows][channels]
  };
  struct SplitTemporaries {
    Activation* branch = nullptr;      // E4 [rows][512]
    Activation* middle = nullptr;      // E4 [rows][256]
    Activation* layer0 = nullptr;      // E4 [rows][512]
    Activation* ffnResidual = nullptr; // E4 [rows][512]
    Activation* qkv = nullptr;         // F16 [rows][1536]
    Activation* normalized = nullptr;  // E4
    Activation* attended = nullptr;    // E4 [rows][512]
  };
  Temporaries createTemporaries(const std::string& label, uint32_t rows, uint32_t channels);
  SplitTemporaries createSplitTemporaries(const std::string& label, uint32_t rows);

  void encodeFusedBlock(id<MTLCommandBuffer> commands, Temporaries& temps, const Activation& state,
                        Activation* output, int block, uint32_t channels, uint32_t width, uint32_t height,
                        uint32_t phase, const FusedLayout& layout, const Tensor& tensor,
                        const Activation* ffnSkipOverride, Activation* rawOutput);
  void encodeSplitBlock(id<MTLCommandBuffer> commands, SplitTemporaries& temps, const Activation& state,
                        Activation* output, int block, uint32_t width, uint32_t height, uint32_t phase,
                        Activation* rawOutput);
  void encodeVit(id<MTLCommandBuffer> commands, Activation& state, uint32_t tokens);
  void capture(id<MTLCommandBuffer> commands, const std::string& name, const Activation& source);

  Activation* allocate(const std::string& label, uint32_t rows, uint32_t channels, Format format);

  Context& context_;
  Model& model_;
  Kernels& kernels_;
  Geometry geometry_;
  Options options_;
  std::vector<std::unique_ptr<Activation>> activations_;
  std::map<std::string, Activation*> allocationsByKey_;
  std::map<std::string, Activation*> boundaries_;
  Activation* head_ = nullptr;
  uint32_t windowPhase_[7] = {};
  uint32_t takeWindowPhase(int level) { return windowPhase_[level]++; }
};

// ---------------------------------------------------------------------------
// MetalFX: Apple's upscalers wrapped for the OpenDLSS-NR pipeline
// ---------------------------------------------------------------------------

// Spatial scaler (images + video): renders the NR network at the input
// resolution and MetalFX scales it to the output resolution with its adaptive
// detail-preserving filter.
class MetalFxSpatial {
 public:
  struct Desc {
    uint32_t inputWidth, inputHeight, outputWidth, outputHeight;
    bool hdr = false;             // RGBA16Float content
    bool edgeAdaptive = true;     // MTLFXSpatialScalerTypeAdaptive vs Bilinear
  };
  static bool supported(id<MTLDevice> device);
  MetalFxSpatial(Context& context, const Desc& desc);
  id<MTLTexture> inputTexture() const { return input_; }
  id<MTLTexture> outputTexture() const { return output_; }
  // Encoding: color conversion + scale into the caller's command buffer.
  void encode(id<MTLCommandBuffer> commands);
  // Full frame path: copies `rgba` (input resolution, RGBA8 or RGBA16F) in,
  // scales, copies the output resolution result out.
  void process(id<MTLCommandBuffer> commands, const void* rgba, void* outRgba);

 private:
  Context& context_;
  id<MTLFXSpatialScaler> scaler_;
  id<MTLTexture> input_, output_;
  id<MTLTexture> rgbaIn_, rgbaOut_;
  Desc desc_;
  bool hdr_ = false;
};

// Temporal scaler (games): low-res render + motion vectors + depth + exposure
// -> upsampled, anti-aliased, temporally accumulated frames. Combined with the
// NR network (which runs on the upscaled frame) this is the game path of the
// pipeline; see apps/demo-game-macos for the end-to-end usage.
class MetalFxTemporal {
 public:
  struct Desc {
    uint32_t inputWidth, inputHeight, outputWidth, outputHeight;
    bool hdr = false;
    bool hasDepth = true;
    bool hasExposure = true;
    bool hasMotionVectors = true;   // required by MTLFX
  };
  static bool supported(id<MTLDevice> device);
  MetalFxTemporal(Context& context, const Desc& desc);

  id<MTLTexture> colorTexture() const { return color_; }
  id<MTLTexture> motionTexture() const { return motion_; }
  id<MTLTexture> depthTexture() const { return depth_; }
  id<MTLTexture> exposureTexture() const { return exposure_; }
  id<MTLTexture> outputTexture() const { return output_; }

  void encode(id<MTLCommandBuffer> commands);
  void reset();

 private:
  Context& context_;
  id<MTLFXTemporalScaler> scaler_;
  id<MTLTexture> color_, motion_, depth_, exposure_, output_;
  Desc desc_;
};

}  // namespace metal
}  // namespace nr
