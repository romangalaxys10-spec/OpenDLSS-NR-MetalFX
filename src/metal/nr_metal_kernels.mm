// nr_metal_kernels.mm — compute dispatch wrappers with the exact operand
// contracts of the network (the twin of src/vulkan/kernels.cpp for the
// reference, unfused route). Pipeline keys mirror the Vulkan specialization
// scheme; pushes are encoded as bytes at buffer(30), which the kernels read as
// `constant Push& pc`.
#import "nr_metal.h"

#import "nr_numeric.h"

namespace nr {
namespace metal {

namespace {
constexpr uint32_t kPushIndex = 30;
}  // namespace

Kernels::Kernels(Context& context) : context_(context) {}

Kernels::~Kernels() = default;

void Kernels::setSiluTable(const std::vector<uint16_t>& table) {
  if (table.size() != 65536) throw std::runtime_error("SiLU table must have 65536 entries");
  if (!siluTable_) siluTable_ = context_.createBuffer(65536 * 2, "SiLU table");
  context_.upload(siluTable_, table.data(), 65536 * 2);
}

id<MTLComputePipelineState> Kernels::pipelineFor(const char* kernel, const std::vector<uint32_t>& constants,
                                                 const std::vector<std::string>& constantNames) {
  return context_.pipeline(kernel, constants, constantNames);
}

// Encodes one dispatch: pipeline, buffers (0..N), push bytes at 30, grid.
static void encodeOp(id<MTLCommandBuffer> commands, id<MTLComputePipelineState> state,
                     const std::vector<id<MTLBuffer>>& buffers, const void* push, uint32_t pushBytes,
                     uint32_t x, uint32_t y, uint32_t z) {
  id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
  [encoder setComputePipelineState:state];
  for (size_t i = 0; i < buffers.size(); ++i) {
    if (buffers[i]) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
  }
  if (pushBytes) [encoder setBytes:push length:pushBytes atIndex:kPushIndex];
  MTLSize grid = MTLSizeMake(x, y, z);
  NSUInteger threads = [state maxTotalThreadsPerThreadgroup];
  NSUInteger width = [state threadExecutionWidth];
  MTLSize tg;
  if (threads >= 256 && threads % 128 == 0) tg = MTLSizeMake(std::min<NSUInteger>(threads, 256), 1, 1);
  else if (threads >= 128) tg = MTLSizeMake(128, 1, 1);
  else tg = MTLSizeMake(std::min<NSUInteger>(threads, width), 1, 1);
  (void)width;
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [encoder endEncoding];
}

void Kernels::dispatchLinear(id<MTLCommandBuffer> commands, const char* kernel,
                             const std::vector<uint32_t>& constants, const std::vector<std::string>& constantNames,
                             const std::vector<id<MTLBuffer>>& buffers, const void* push, uint32_t pushBytes,
                             uint32_t count) {
  id<MTLComputePipelineState> state = pipelineFor(kernel, constants, constantNames);
  encodeOp(commands, state, buffers, push, pushBytes, count, 1, 1);
  ++dispatches_;
}

void Kernels::dispatchGrid(id<MTLCommandBuffer> commands, const char* kernel,
                           const std::vector<uint32_t>& constants, const std::vector<std::string>& constantNames,
                           const std::vector<id<MTLBuffer>>& buffers, const void* push, uint32_t pushBytes,
                           uint32_t x, uint32_t y, uint32_t z) {
  id<MTLComputePipelineState> state = pipelineFor(kernel, constants, constantNames);
  encodeOp(commands, state, buffers, push, pushBytes, x, y, z);
  ++dispatches_;
}

// ---------------------------------------------------------------------------
// gemmFp8 — the FP8 GEMM workhorse. Shapes/flags mirror kernels.cpp::gemmFp8.
// ---------------------------------------------------------------------------

void Kernels::gemmFp8(id<MTLCommandBuffer> commands, const GemmFp8Args& a) {
  NR_CHECK(a.input && a.input->format == Format::E4, "FP8 GEMM input must be E4");
  NR_CHECK(a.K % 32 == 0 && a.N % 16 == 0 && a.Nmatrix % 16 == 0, "FP8 GEMM shape");
  NR_CHECK((a.outputColumnOffset % 16) == 0 && (a.auxByteOffset % 16) == 0, "GEMM epilogue alignment");
  NR_CHECK(!a.scaleResidual || a.auxTensor, "scaled residual needs an aux tensor");
  NR_CHECK(a.output->allocRows >= alignRows(a.rows) && a.input->allocRows >= alignRows(a.rows), "GEMM rows");

  uint32_t flags = 0;
  if (a.residual) flags |= 1u;                  // F_RESIDUAL
  if (a.scaleResidual) flags |= 2u;             // F_SCALE_RESIDUAL
  if (a.silu) flags |= 8u;                      // F_SILU
  if (a.quantize) flags |= 16u;                 // F_QUANTIZE
  if (a.dualOutput) flags |= 32u;               // F_DUAL
  if (a.residual && a.residual->format == Format::E4) flags |= 64u;   // F_RESIDUAL_E4
  if (a.broadcastInput) flags |= 128u;          // F_BROADCAST_INPUT

  // The partition count is the network's K split: the f16 partial sums are
  // combined in that order, so it is part of the arithmetic, not a tuning
  // knob. Metal v1 always takes the split-K + reduce route for partitioned
  // GEMMs (the in-kernel PARTITION accumulator of the GLSL kernel implements
  // the same reduction order when SPLITK == 1; both produce identical bytes).
  uint32_t splits = (a.partition && a.K > a.partition) ? a.K / a.partition : 1;

  // Column tiles: 16 columns per tile, up to 8 (kernel cap MAX_TILES).
  uint32_t tiles = std::min<uint32_t>(a.N / 16, 8u);
  if (tiles == 0u) tiles = 1u;

  std::vector<uint32_t> constants;
  std::vector<std::string> names = {"K", "TILES", "FLAGS", "PARTITION", "SPLITK"};
  constants.push_back(a.K);
  constants.push_back(tiles);
  // Publication moves to the reduce pass when split-K; seed-residual still only on split 0.
  uint32_t kernelFlags = flags;
  if (splits > 1) kernelFlags &= ~(8u | 16u | 32u);
  constants.push_back(kernelFlags);
  constants.push_back(splits > 1 ? 0u : a.partition);
  constants.push_back(splits);

  struct Push {
    uint32_t rows, N, Nmatrix, weightColumnOffset, inputStride, inputColumnBase, outputStride, outputColumnOffset,
        auxHalfOffset, batches, columnGroups, splitStride;
  } push{a.rows, a.N, a.Nmatrix, a.weightColumnOffset, a.input->channels, a.inputColumnBase, a.output->channels,
         a.outputColumnOffset, a.auxByteOffset / 2, a.batches, a.N / (tiles * 16u), alignRows(a.rows) * a.N};

  id<MTLBuffer> partials = nil;
  std::vector<id<MTLBuffer>> buffers(7);
  buffers[0] = a.input->buffer;
  buffers[1] = a.weights;
  if (splits > 1) {
    // One fixed scratch per Kernels instance (allocations must not move while
    // recorded dispatches reference them).
    if (!partialsBuffer_) partialsBuffer_ = context_.createBuffer(16ull << 20, "split-K partials");
    partials = partialsBuffer_;
    NR_CHECK((uint64_t)splits * push.splitStride * 2 <= 16ull << 20, "split-K partials exceed the scratch buffer");
    buffers[2] = partials;
  } else if (!a.quantize) {
    buffers[2] = a.output->buffer;   // kernel writes raw f16 there when not quantizing
  }
  if (a.residual && a.residual->format == Format::F16) buffers[3] = a.residual->buffer;
  if (a.auxTensor) buffers[4] = a.auxTensor->raw;
  if (a.quantize) buffers[5] = a.output->buffer;
  if (a.dualOutput) buffers[5] = a.dualOutput->buffer;
  if (a.residual && a.residual->format == Format::E4) buffers[6] = a.residual->buffer;

  uint32_t rowGroups = (a.rows + 63u) / 64u;
  uint32_t gy = std::min(rowGroups, 65535u), gz = (rowGroups + 65534) / 65535;
  dispatchGrid(commands, "nr_gemm_fp8", constants, names, buffers, &push, sizeof(push),
               a.batches * (a.N / (tiles * 16u)) * splits, gy, gz);

  if (splits > 1) {
    uint32_t publishFlags = flags & (8u | 16u | 32u);
    std::vector<uint32_t> reduceConstants = {splits, publishFlags};
    std::vector<std::string> reduceNames = {"SPLITK", "FLAGS"};
    struct ReducePush {
      uint32_t rows, N, splitStride, outputStride, outputColumnOffset;
    } reducePush{a.rows, a.N, push.splitStride, a.output->channels, a.outputColumnOffset};
    std::vector<id<MTLBuffer>> reduceBuffers(7);
    reduceBuffers[0] = partials;
    if (!a.quantize) reduceBuffers[2] = a.output->buffer;
    if (a.quantize) reduceBuffers[5] = a.output->buffer;
    if (a.dualOutput) reduceBuffers[5] = a.dualOutput->buffer;
    dispatchLinear(commands, "nr_gemm_reduce", reduceConstants, reduceNames, reduceBuffers, &reducePush,
                   sizeof(reducePush), a.rows * a.N / 8);
  }
}

// ---------------------------------------------------------------------------
// gemmF16 — the two f16 boundaries (16 -> 32 input adapter, 32 -> 4 head).
// ---------------------------------------------------------------------------

void Kernels::gemmF16(id<MTLCommandBuffer> commands, const GemmF16Args& a) {
  NR_CHECK(a.input && a.input->format == Format::F16, "f16 GEMM input must be f16");
  NR_CHECK(a.K % 16 == 0 && a.paddedN % 16 == 0 && a.N <= a.paddedN, "f16 GEMM shape");
  NR_CHECK(a.output->allocRows >= alignRows(a.rows), "f16 GEMM rows");
  uint32_t flags = 0;
  if (a.output->format == Format::F32) flags |= 256u;       // F_OUT_F32
  else if (a.output->format == Format::E4) flags |= 16u;    // F_QUANTIZE
  if (a.dualOutput) flags |= 32u;                           // F_DUAL

  uint32_t tileN = a.paddedN;
  NR_CHECK(tileN <= 64, "f16 GEMM paddedN must be <= 64 (kernel cap)");
  std::vector<uint32_t> constants = {a.K, tileN, flags};
  std::vector<std::string> names = {"K", "TILE_N", "FLAGS"};
  struct Push {
    uint32_t rows, N, inputStride, outputStride;
  } push{a.rows, a.N, a.input->channels, a.output->channels};

  std::vector<id<MTLBuffer>> buffers(8);
  buffers[0] = a.input->buffer;
  buffers[1] = a.weights;
  if (a.output->format == Format::F32) buffers[7] = a.output->buffer;
  else if (flags & 16u) buffers[5] = a.output->buffer;
  else buffers[2] = a.output->buffer;
  if (a.dualOutput) buffers[5] = a.dualOutput->buffer;

  uint32_t rowGroups = (a.rows + 63u) / 64u;
  uint32_t gy = std::min(rowGroups, 65535u), gz = (rowGroups + 65534) / 65535;
  dispatchGrid(commands, "nr_gemm_f16", constants, names, buffers, &push, sizeof(push), 1, gy, gz);
}

// ---------------------------------------------------------------------------
// ops (5 modes) + preprocess
// ---------------------------------------------------------------------------

void Kernels::preprocessFromProxy(id<MTLCommandBuffer> commands, const Activation& proxy, Activation& features,
                                  const PreprocessArgs& args) {
  NR_CHECK(features.format == Format::F32 && features.channels == 16, "features must be f32 [rows][16]");
  struct Push {
    uint32_t fullWidth, fullHeight, validWidth, validHeight, sourceWidth, sourceHeight, seed;
    float autoMask, localTone, localStructure, skinStructure, style;
  } push{args.fullWidth, args.fullHeight, args.validWidth, args.validHeight, args.sourceWidth, args.sourceHeight,
         args.seed, args.autoMask ? 1.0f : 0.0f, args.localTone, args.localStructure, args.skinStructure, args.style};
  std::vector<id<MTLBuffer>> buffers(8);
  buffers[0] = proxy.buffer;
  buffers[7] = features.buffer;
  dispatchGrid(commands, "nr_preprocess", {}, {}, buffers, &push, sizeof(push), (args.fullWidth + 7) / 8,
               (args.fullHeight + 7) / 8, 1);
}

void Kernels::dispatchOps(id<MTLCommandBuffer> commands, uint32_t mode, const std::vector<id<MTLBuffer>>& buffers,
                          uint32_t count, uint32_t channels, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth,
                          uint32_t outHeight, uint32_t auxOffsetA, uint32_t auxOffsetB, uint32_t dual) {
  struct Push {
    uint32_t count, channels, inWidth, inHeight, outWidth, outHeight, auxOffsetA, auxOffsetB, dual;
  } push{count, channels, inWidth, inHeight, outWidth, outHeight, auxOffsetA, auxOffsetB, dual};
  std::vector<uint32_t> constants = {mode};
  std::vector<std::string> names = {"MODE"};
  dispatchLinear(commands, "nr_ops", constants, names, buffers, &push, sizeof(push), count / 8);
}

void Kernels::convertF32ToF16(id<MTLCommandBuffer> commands, const Activation& input, Activation& output) {
  std::vector<id<MTLBuffer>> buffers(7);
  buffers[0] = input.buffer;
  buffers[6] = output.buffer;
  uint32_t count = input.rows * input.channels;
  dispatchOps(commands, 0u, buffers, count, input.channels, 0, 0, 0, 0, 0, 0, 0);
}

void Kernels::quantize(id<MTLCommandBuffer> commands, const Activation& input, Activation& output) {
  std::vector<id<MTLBuffer>> buffers(7);
  buffers[1] = input.buffer;
  buffers[5] = output.buffer;
  uint32_t count = input.rows * input.channels;
  dispatchOps(commands, 1u, buffers, count, input.channels, 0, 0, 0, 0, 0, 0, 0);
}

void Kernels::downsample2x(id<MTLCommandBuffer> commands, const Activation& input, Activation& output,
                           uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight) {
  std::vector<id<MTLBuffer>> buffers(7);
  buffers[1] = input.buffer;
  buffers[5] = output.buffer;
  uint32_t count = output.rows * output.channels;
  dispatchOps(commands, 2u, buffers, count, output.channels, inWidth, inHeight, outWidth, outHeight, 0, 0, 0);
}

void Kernels::postBlend(id<MTLCommandBuffer> commands, const Activation& upsampleSource, const Activation& adapter,
                        const Tensor& tensor, uint32_t inputScaleByteOffset, uint32_t adapterScaleByteOffset,
                        Activation& rawOutput, Activation& quantizedOutput, uint32_t inWidth, uint32_t inHeight,
                        uint32_t outWidth, uint32_t outHeight) {
  std::vector<id<MTLBuffer>> buffers(7);
  buffers[2] = upsampleSource.buffer;   // In8: 2x upsampled low-res E4
  buffers[3] = adapter.buffer;          // Skip8: block-0 E4
  buffers[4] = tensor.raw;              // Aux
  buffers[5] = quantizedOutput.buffer;  // Out8
  buffers[6] = rawOutput.buffer;        // Out16
  uint32_t count = rawOutput.rows * rawOutput.channels;
  dispatchOps(commands, 3u, buffers, count, quantizedOutput.channels, inWidth, inHeight, outWidth, outHeight,
              inputScaleByteOffset / 2, adapterScaleByteOffset / 2, 0);
}

void Kernels::upsampleResidual(id<MTLCommandBuffer> commands, const Activation& projection, const Activation& skip,
                               const Tensor& tensor, uint32_t scaleByteOffset, Activation& output,
                               Activation* rawOutput, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth,
                               uint32_t outHeight) {
  std::vector<id<MTLBuffer>> buffers(7);
  buffers[1] = projection.buffer;   // In16: low-res f16 projection
  buffers[3] = skip.buffer;         // Skip8: E4 skip
  buffers[4] = tensor.raw;          // Aux
  buffers[5] = output.buffer;       // Out8
  if (rawOutput) buffers[6] = rawOutput->buffer;
  uint32_t count = output.rows * output.channels;
  dispatchOps(commands, 4u, buffers, count, output.channels, inWidth, inHeight, outWidth, outHeight,
              scaleByteOffset / 2, 0, rawOutput ? 1u : 0u);
}

// ---------------------------------------------------------------------------
// Window / global attention (unfused pair)
// ---------------------------------------------------------------------------

void Kernels::windowNormalize(id<MTLCommandBuffer> commands, const Activation& qkv, const Tensor& tensor,
                              uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads) {
  struct Push {
    uint32_t tokens, heads, channels, scaleWordOffset;
  } push{tokens, heads, qkv.channels / 3, scaleByteOffset / 4};
  std::vector<id<MTLBuffer>> buffers(6);
  buffers[1] = qkv.buffer;
  buffers[4] = tensor.raw;
  buffers[5] = normalized.buffer;
  dispatchLinear(commands, "nr_window_normalize", {}, {}, buffers, &push, sizeof(push),
                 ((uint64_t)tokens * heads + 255) / 256);
}

void Kernels::windowAttend(id<MTLCommandBuffer> commands, const Activation& normalized, id<MTLBuffer> prior,
                           Activation& attended, uint32_t width, uint32_t height, uint32_t heads, uint32_t shiftX,
                           uint32_t shiftY) {
  uint32_t windowsX = (width + shiftX + 7) / 8;
  uint32_t windowsY = (height + shiftY + 7) / 8;
  struct Push {
    uint32_t width, height, channels, heads, shiftX, shiftY, windowsX, windowCount;
  } push{width, height, attended.channels, heads, shiftX, shiftY, windowsX, windowsX * windowsY};
  std::vector<id<MTLBuffer>> buffers(6);
  buffers[0] = normalized.buffer;
  buffers[1] = prior;
  buffers[5] = attended.buffer;
  dispatchGrid(commands, "nr_window_attend", {}, {}, buffers, &push, sizeof(push), heads, windowsX * windowsY, 1);
}

void Kernels::globalNormalize(id<MTLCommandBuffer> commands, const Activation& qkv, const Tensor& tensor,
                              uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads) {
  struct Push {
    uint32_t tokens, heads, channels, scaleWordOffset;
  } push{tokens, heads, qkv.channels / 3, scaleByteOffset / 4};
  std::vector<id<MTLBuffer>> buffers(6);
  buffers[1] = qkv.buffer;
  buffers[4] = tensor.raw;
  buffers[5] = normalized.buffer;
  dispatchLinear(commands, "nr_global_normalize", {}, {}, buffers, &push, sizeof(push),
                 ((uint64_t)tokens * heads + 255) / 256);
}

void Kernels::globalAttend(id<MTLCommandBuffer> commands, const Activation& normalized, Activation& attended,
                           uint32_t tokens, uint32_t paddedTokens, uint32_t heads) {
  NR_CHECK(paddedTokens <= 256, "global attend route caps padded tokens at 256 (streamed route: Vulkan/NVIDIA)");
  struct Push {
    uint32_t tokens, heads, channels;
  } push{tokens, heads, attended.channels};
  std::vector<uint32_t> constants = {paddedTokens};
  std::vector<std::string> names = {"PADDED_TOKENS"};
  std::vector<id<MTLBuffer>> buffers(6);
  buffers[0] = normalized.buffer;
  buffers[5] = attended.buffer;
  dispatchGrid(commands, "nr_global_attend", constants, names, buffers, &push, sizeof(push), heads,
               (paddedTokens + 15) / 16, 1);
}

}  // namespace metal
}  // namespace nr
