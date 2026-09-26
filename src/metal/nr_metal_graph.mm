// nr_metal_graph.mm — the 71-block NR network graph recorded into one Metal
// command buffer (the twin of src/vulkan/nr_graph.cpp for the reference,
// unfused route): pre adapter + block 0, encoder 32/64/128/256, split-512
// branches, the global ViT, the decoder chain and the post block + RGBA head.
// Every inter-GEMM boundary is E4M3 published through the shared bit-exact
// conversions; launches are barrier-separated (computeBarrier between ops).
#import "nr_metal.h"

#import <algorithm>

namespace nr {
namespace metal {

Graph::Graph(Context& context, Model& model, Kernels& kernels, const Geometry& geometry, Options options)
    : context_(context), model_(model), kernels_(kernels), geometry_(geometry), options_(options) {}

Graph::~Graph() = default;

Activation* Graph::allocate(const std::string& label, uint32_t rows, uint32_t channels, Format format) {
  std::string key = label + "/" + std::to_string(rows) + "x" + std::to_string(channels) + "/" +
                    std::to_string((int)format);
  auto it = allocationsByKey_.find(key);
  if (it != allocationsByKey_.end()) return it->second;
  auto activation = std::make_unique<Activation>();
  activation->buffer = context_.createBuffer((uint64_t)alignRows(rows) * channels * formatBytes(format), label.c_str());
  activation->format = format;
  activation->rows = rows;
  activation->channels = channels;
  activation->allocRows = alignRows(rows);
  activation->label = label;
  Activation* pointer = activation.get();
  activations_.push_back(std::move(activation));
  allocationsByKey_[key] = pointer;
  return pointer;
}

Graph::Temporaries Graph::createTemporaries(const std::string& label, uint32_t rows, uint32_t channels) {
  Temporaries temps;
  temps.ffn = allocate(label + " FFN", rows, 128, Format::E4);
  temps.ffnResidual = allocate(label + " FFN residual", rows, channels, Format::F16);
  temps.ffnQuantized = allocate(label + " FFN quantized", rows, channels, Format::E4);
  temps.qkv = allocate(label + " QKV", rows, channels * 3, Format::F16);
  temps.normalized = allocate(label + " normalized QKV", rows, channels * 3, Format::E4);
  temps.attended = allocate(label + " attended", rows, channels, Format::E4);
  if (channels >= 64) temps.ffnNarrow = allocate(label + " FFN narrow", rows, 32, Format::E4);
  return temps;
}

Graph::SplitTemporaries Graph::createSplitTemporaries(const std::string& label, uint32_t rows) {
  SplitTemporaries temps;
  temps.branch = allocate(label + " branch", rows, 512, Format::E4);
  temps.middle = allocate(label + " middle", rows, 8 * 256, Format::E4);
  temps.layer0 = allocate(label + " layer0", rows, 512, Format::E4);
  temps.ffnResidual = allocate(label + " FFN residual", rows, 512, Format::E4);
  temps.qkv = allocate(label + " QKV", rows, 1536, Format::F16);
  temps.normalized = allocate(label + " normalized", rows, 1536, Format::E4);
  temps.attended = allocate(label + " attended", rows, 512, Format::E4);
  return temps;
}

void Graph::capture(id<MTLCommandBuffer> commands, const std::string& name, const Activation& source) {
  if (!options_.captureBoundaries && !options_.captureIntermediates) return;
  if (boundaries_.count(name)) return;
  Activation* copy = allocate("capture " + name, source.rows, source.channels, source.format);
  id<MTLBlitCommandEncoder> encoder = [commands blitCommandEncoder];
  [encoder copyFromBuffer:source.buffer sourceOffset:0 toBuffer:copy->buffer destinationOffset:0
                     size:(source.rows * source.channels * formatBytes(source.format))];
  [encoder endEncoding];
  boundaries_[name] = copy;
}

// One dense fused-layout block through the reference decomposition:
// expand GEMM (SiLU, E4) -> contract GEMM (+ scaled skip) -> QKV GEMM ->
// window normalize + attend -> projection GEMM (+ scaled ffn residual).
void Graph::encodeFusedBlock(id<MTLCommandBuffer> commands, Temporaries& temps, const Activation& state,
                             Activation* output, int block, uint32_t channels, uint32_t width, uint32_t height,
                             uint32_t phase, const FusedLayout& layout, const Tensor& tensor,
                             const Activation* ffnSkipOverride, Activation* rawOutput) {
  const uint32_t rows = width * height;
  const Activation* residual = ffnSkipOverride ? ffnSkipOverride : &state;
  std::string prefix = "block " + std::to_string(block) + " c" + std::to_string(channels);

  if (layout.expertFfn) {
    // C/32 expert paths, each C -> 128 -> 32 -> C, then one K = C GEMM chains
    // every expert into the scaled skip.
    const uint32_t experts = layout.expertCount;
    const uint32_t w2Base = layout.expand + experts * channels * 128;
    const uint32_t w3Base = w2Base + experts * 128 * 32;
    id<MTLBuffer> w1Weights = model_.fp8Matrix(tensor, layout.expand, experts * channels, 128, true, channels);
    id<MTLBuffer> w2Weights = model_.fp8Matrix(tensor, w2Base, experts * 128, 32, true, 128);
    id<MTLBuffer> w3Weights = model_.fp8Matrix(tensor, w3Base, channels, channels);
    GemmFp8Args w1;
    w1.input = &state; w1.rows = rows; w1.K = channels; w1.N = 128; w1.batches = experts; w1.broadcastInput = true;
    w1.weights = w1Weights; w1.Nmatrix = 128;
    w1.output = temps.ffn; w1.silu = true; w1.quantize = true;
    kernels_.gemmFp8(commands, w1);
    GemmFp8Args w2;
    w2.input = temps.ffn; w2.rows = rows; w2.K = 128; w2.N = 32; w2.batches = experts;
    w2.weights = w2Weights; w2.Nmatrix = 32;
    w2.output = temps.ffnNarrow; w2.quantize = true;
    kernels_.gemmFp8(commands, w2);
    GemmFp8Args w3;
    w3.input = temps.ffnNarrow; w3.rows = rows; w3.K = channels; w3.N = channels;
    w3.weights = w3Weights; w3.Nmatrix = channels;
    w3.output = temps.ffnResidual; w3.quantize = false; w3.dualOutput = temps.ffnQuantized;
    w3.residual = residual; w3.scaleResidual = true; w3.auxTensor = &tensor; w3.auxByteOffset = layout.ffnCosSkip;
    kernels_.gemmFp8(commands, w3);
  } else {
    GemmFp8Args expand;
    expand.input = &state; expand.rows = rows; expand.K = channels; expand.N = layout.hidden;
    expand.weights = model_.fp8Matrix(tensor, layout.expand, channels, layout.hidden);
    expand.Nmatrix = layout.hidden;
    expand.output = temps.ffn; expand.silu = true; expand.quantize = true;
    kernels_.gemmFp8(commands, expand);
    GemmFp8Args contract;
    contract.input = temps.ffn; contract.rows = rows; contract.K = layout.hidden; contract.N = channels;
    contract.weights = model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, channels);
    contract.Nmatrix = channels;
    contract.output = temps.ffnResidual; contract.quantize = false; contract.dualOutput = temps.ffnQuantized;
    contract.residual = residual; contract.scaleResidual = true; contract.auxTensor = &tensor;
    contract.auxByteOffset = layout.ffnCosSkip;
    kernels_.gemmFp8(commands, contract);
  }
  (void)prefix;
  if (options_.captureIntermediates) {
    capture(commands, "block-" + std::to_string(block) + "/ffnResidual", *temps.ffnResidual);
    capture(commands, "block-" + std::to_string(block) + "/ffnQuantized", *temps.ffnQuantized);
  }

  uint32_t shiftX, shiftY;
  windowPhase(phase, shiftX, shiftY);
  id<MTLBuffer> qkvWeights = model_.fp8Matrix(tensor, layout.qkv, channels, channels * 3);
  {
    GemmFp8Args qkv;
    qkv.input = temps.ffnQuantized; qkv.rows = rows; qkv.K = channels; qkv.N = channels * 3;
    qkv.weights = qkvWeights; qkv.Nmatrix = channels * 3;
    qkv.output = temps.qkv; qkv.quantize = false;
    kernels_.gemmFp8(commands, qkv);
  }
  if (options_.captureIntermediates) capture(commands, "block-" + std::to_string(block) + "/qkv", *temps.qkv);
  kernels_.windowNormalize(commands, *temps.qkv, tensor, layout.scale, *temps.normalized, rows, layout.heads);
  if (options_.captureIntermediates)
    capture(commands, "block-" + std::to_string(block) + "/normalized", *temps.normalized);
  kernels_.windowAttend(commands, *temps.normalized, model_.relativeBias(tensor, layout.relative, layout.heads),
                        *temps.attended, width, height, layout.heads, shiftX, shiftY);
  if (options_.captureIntermediates) capture(commands, "block-" + std::to_string(block) + "/attended", *temps.attended);

  GemmFp8Args projection;
  projection.input = temps.attended; projection.rows = rows; projection.K = channels; projection.N = channels;
  projection.weights = model_.fp8Matrix(tensor, layout.projection, channels, channels);
  projection.Nmatrix = channels;
  projection.residual = layout.expertFfn ? (const Activation*)temps.ffnQuantized : (const Activation*)temps.ffnResidual;
  projection.scaleResidual = true; projection.auxTensor = &tensor; projection.auxByteOffset = layout.attnCosSkip;
  if (rawOutput) {
    projection.output = rawOutput; projection.quantize = false; projection.dualOutput = output;
  } else {
    projection.output = output; projection.quantize = true;
  }
  kernels_.gemmFp8(commands, projection);
}

// Eight independent 512 -> 64 -> 256 -> 64 FFN branches (SiLU after the
// 256-wide middle only), concatenated and contracted, then 16-head window
// attention. Every inter-GEMM boundary is E4M3.
void Graph::encodeSplitBlock(id<MTLCommandBuffer> commands, SplitTemporaries& temps, const Activation& state,
                             Activation* output, int block, uint32_t width, uint32_t height, uint32_t phase,
                             Activation* rawOutput) {
  const uint32_t rows = width * height;
  const uint32_t channels = 512;
  const Tensor& branchTensor = model_.tensor(block, 0);
  const Tensor& contract = model_.tensor(block, 1);
  const Tensor& qkvTensor = model_.tensor(block, 2);
  const Tensor& projection = model_.tensor(block, 3);
  const uint32_t branches = 8, branchChannels = 64, middleChannels = 256;
  const uint32_t w2Base = branches * channels * branchChannels;
  const uint32_t w3Base = w2Base + branches * branchChannels * middleChannels;
  const uint32_t heads = 16;
  const uint32_t qkvRelative = channels * channels * 3, qkvScale = qkvRelative + heads * 8192;

  GemmFp8Args w1;
  w1.input = &state; w1.rows = rows; w1.K = channels; w1.N = channels;
  w1.weights = model_.fp8Matrix(branchTensor, 0, channels, channels); w1.Nmatrix = channels;
  w1.output = temps.branch; w1.quantize = true;
  kernels_.gemmFp8(commands, w1);
  {
    id<MTLBuffer> w2Weights =
        model_.fp8Matrix(branchTensor, w2Base, branches * branchChannels, middleChannels, true, branchChannels);
    id<MTLBuffer> w3Weights =
        model_.fp8Matrix(branchTensor, w3Base, branches * middleChannels, branchChannels, true, middleChannels);
    GemmFp8Args w2;
    w2.input = temps.branch; w2.rows = rows; w2.K = branchChannels; w2.N = middleChannels; w2.batches = branches;
    w2.weights = w2Weights; w2.Nmatrix = middleChannels; w2.output = temps.middle; w2.silu = true; w2.quantize = true;
    kernels_.gemmFp8(commands, w2);
    GemmFp8Args w3;
    w3.input = temps.middle; w3.rows = rows; w3.K = middleChannels; w3.N = branchChannels; w3.batches = branches;
    w3.weights = w3Weights; w3.Nmatrix = branchChannels; w3.output = temps.layer0; w3.quantize = true;
    kernels_.gemmFp8(commands, w3);
  }

  GemmFp8Args ffn;
  ffn.input = temps.layer0; ffn.rows = rows; ffn.K = channels; ffn.N = channels;
  ffn.weights = model_.fp8Matrix(contract, 0, channels, channels); ffn.Nmatrix = channels;
  ffn.output = temps.ffnResidual; ffn.quantize = true;
  ffn.residual = &state; ffn.scaleResidual = true; ffn.auxTensor = &contract; ffn.auxByteOffset = channels * channels;
  kernels_.gemmFp8(commands, ffn);

  uint32_t shiftX, shiftY;
  windowPhase(phase, shiftX, shiftY);
  {
    GemmFp8Args qkv;
    qkv.input = temps.ffnResidual; qkv.rows = rows; qkv.K = channels; qkv.N = channels * 3;
    qkv.weights = model_.fp8Matrix(qkvTensor, 0, channels, channels * 3); qkv.Nmatrix = channels * 3;
    qkv.output = temps.qkv; qkv.quantize = false;
    kernels_.gemmFp8(commands, qkv);
    kernels_.windowNormalize(commands, *temps.qkv, qkvTensor, qkvScale, *temps.normalized, rows, heads);
    kernels_.windowAttend(commands, *temps.normalized, model_.relativeBias(qkvTensor, qkvRelative, heads),
                          *temps.attended, width, height, heads, shiftX, shiftY);
  }
  if (options_.captureIntermediates) {
    capture(commands, "block-" + std::to_string(block) + "/qkv", *temps.qkv);
    capture(commands, "block-" + std::to_string(block) + "/attended", *temps.attended);
  }

  GemmFp8Args proj;
  proj.input = temps.attended; proj.rows = rows; proj.K = channels; proj.N = channels;
  proj.weights = model_.fp8Matrix(projection, 0, channels, channels); proj.Nmatrix = channels;
  proj.residual = temps.ffnResidual; proj.scaleResidual = true; proj.auxTensor = &projection;
  proj.auxByteOffset = channels * channels;
  if (rawOutput) {
    proj.output = rawOutput; proj.quantize = false; proj.dualOutput = output;
  } else {
    proj.output = output; proj.quantize = true;
  }
  kernels_.gemmFp8(commands, proj);
}

// Global ViT: eight 1024-channel blocks over the coarsest tokens.
void Graph::encodeVit(id<MTLCommandBuffer> commands, Activation& state, uint32_t tokens) {
  const uint32_t channels = 1024, heads = 32, ffnChannels = 4096;
  const uint32_t padded = geometry_.paddedVitTokens();
  Activation* expanded = allocate("ViT FFN 4096", tokens, ffnChannels, Format::E4);
  Activation* ffnResidual = allocate("ViT FFN residual", tokens, channels, Format::E4);
  Activation* qkv = allocate("ViT QKV", tokens, channels * 3, Format::F16);
  Activation* normalized = allocate("ViT normalized QKV", padded, channels * 3, Format::E4);
  Activation* attended = allocate("ViT attended", tokens, channels, Format::E4);
  NR_CHECK(padded <= 256, "the Metal single-kernel ViT route caps padded tokens at 256 (see docs/PERFORMANCE.md)");

  for (int block = 31; block <= 38; ++block) {
    const Tensor& expand = model_.tensor(block, 0);
    const Tensor& contract = model_.tensor(block, 1);
    const Tensor& qkvTensor = model_.tensor(block, 2);
    const Tensor& projection = model_.tensor(block, 4);
    GemmFp8Args e;
    e.input = &state; e.rows = tokens; e.K = channels; e.N = ffnChannels;
    e.weights = model_.fp8Matrix(expand, 0, channels, ffnChannels); e.Nmatrix = ffnChannels;
    e.output = expanded; e.silu = true; e.quantize = true;
    kernels_.gemmFp8(commands, e);
    GemmFp8Args c;
    c.input = expanded; c.rows = tokens; c.K = ffnChannels; c.N = channels; c.partition = 1024;
    c.weights = model_.fp8Matrix(contract, 0, ffnChannels, channels); c.Nmatrix = channels;
    c.output = ffnResidual; c.quantize = true;
    c.residual = &state; c.scaleResidual = true; c.auxTensor = &contract; c.auxByteOffset = ffnChannels * channels;
    kernels_.gemmFp8(commands, c);
    GemmFp8Args q;
    q.input = ffnResidual; q.rows = tokens; q.K = channels; q.N = channels * 3; q.partition = 512;
    q.weights = model_.fp8Matrix(qkvTensor, heads * 4, channels, channels * 3); q.Nmatrix = channels * 3;
    q.output = qkv; q.quantize = false;
    kernels_.gemmFp8(commands, q);
    kernels_.globalNormalize(commands, *qkv, qkvTensor, 0, *normalized, tokens, heads);
    if (options_.captureIntermediates)
      capture(commands, "block-" + std::to_string(block) + "/normalized", *normalized);
    kernels_.globalAttend(commands, *normalized, *attended, tokens, padded, heads);
    if (options_.captureIntermediates) capture(commands, "block-" + std::to_string(block) + "/attended", *attended);
    GemmFp8Args p;
    p.input = attended; p.rows = tokens; p.K = channels; p.N = channels; p.partition = 256;
    p.weights = model_.fp8Matrix(projection, 0, channels, channels); p.Nmatrix = channels;
    p.output = &state; p.quantize = true;
    p.residual = ffnResidual; p.scaleResidual = true; p.auxTensor = &projection; p.auxByteOffset = channels * channels;
    kernels_.gemmFp8(commands, p);
    capture(commands, "block-" + std::to_string(block), state);
  }
}

void Graph::record(id<MTLCommandBuffer> commands, const Activation& inputFeatures) {
  kernels_.resetDispatchCount();
  const Geometry& g = geometry_;
  const uint32_t fullRows = g.fullWidth * g.fullHeight;
  NR_CHECK(inputFeatures.format == Format::F32 && inputFeatures.rows == fullRows && inputFeatures.channels == 16,
           "input features must be f32 [fullWidth*fullHeight][16]");
  boundaries_.clear();
  for (uint32_t& phase : windowPhase_) phase = 0;

  // ---- Encoder 32 pre: FP16 input adapter, full-resolution block 0, downsample.
  const Tensor& preTensor = model_.tensor(0);
  FusedLayout preLayout = preFusedLayout();
  if (preTensor.byteLength != preLayout.endWithoutPadding + 16)
    throw std::runtime_error("unexpected block0 layout");
  Activation* adapter = allocate("retained full block0", fullRows, 32, Format::E4);
  Activation* resized = allocate("block0 downsample", g.levels[0].width * g.levels[0].height, 32, Format::E4);
  Activation* adapterRaw = allocate("raw FP16 block0", fullRows, 32, Format::F16);
  {
    Activation* inputHalf = allocate("input features f16", fullRows, 16, Format::F16);
    kernels_.convertF32ToF16(commands, inputFeatures, *inputHalf);
    Activation* projectedFp16 = allocate("full FP16 input adapter", fullRows, 32, Format::F16);
    Activation* projected = allocate("full FP8 input adapter", fullRows, 32, Format::E4);
    {
      uint32_t paddedN = 0;
      GemmF16Args pre;
      pre.input = inputHalf;
      pre.weights = model_.f16Matrix(preTensor, preLayout.inputAdapter, 16, 32, paddedN);
      pre.paddedN = paddedN; pre.output = projectedFp16; pre.dualOutput = projected;
      pre.rows = fullRows; pre.K = 16; pre.N = 32;
      kernels_.gemmF16(commands, pre);
    }
    Temporaries fullTemps = createTemporaries("pre block0", fullRows, 32);
    encodeFusedBlock(commands, fullTemps, *projected, adapter, 0, 32, g.fullWidth, g.fullHeight,
                     takeWindowPhase(6), preLayout, preTensor, projectedFp16, adapterRaw);
    if (options_.captureIntermediates) capture(commands, "block-0/adapterRaw", *adapterRaw);
  }
  capture(commands, "block-0", *adapter);

  const Geometry::Level d0 = g.levels[0], d1 = g.levels[1], d2 = g.levels[2], d3 = g.levels[3], d4 = g.levels[4],
                        d5 = g.levels[5];
  const uint32_t rows0 = d0.width * d0.height;
  kernels_.downsample2x(commands, *adapterRaw, *resized, g.fullWidth, g.fullHeight, d0.width, d0.height);
  capture(commands, "transition-0-1", *resized);

  // ---- Encoder 32: blocks 1-4.
  Activation* state = resized;
  Activation* scratch = allocate("encoder 32 state", rows0, 32, Format::E4);
  Activation* transitionRaw32 = allocate("raw FP16 block4", rows0, 32, Format::F16);
  Temporaries latentTemps = createTemporaries("encoder 32", rows0, 32);
  const uint32_t rows1 = d1.width * d1.height;
  Activation* downsampled32 = allocate("encoder 32 downsample", rows1, 32, Format::E4);
  for (int block = 1; block <= 4; ++block) {
    encodeFusedBlock(commands, latentTemps, *state, scratch, block, 32, d0.width, d0.height, takeWindowPhase(0),
                     fusedLayout(32), model_.tensor(block), nullptr, block == 4 ? transitionRaw32 : nullptr);
    std::swap(state, scratch);
    capture(commands, "block-" + std::to_string(block), *state);
  }
  Activation* skip32 = state;
  kernels_.downsample2x(commands, *transitionRaw32, *downsampled32, d0.width, d0.height, d1.width, d1.height);
  capture(commands, "pooled-4-5", *downsampled32);
  Activation* next64 = allocate("encoder 64 input", rows1, 64, Format::E4);
  {
    GemmFp8Args t;
    t.input = downsampled32; t.rows = rows1; t.K = 32; t.N = 64;
    t.weights = model_.fp8Matrix(model_.tensor(4), fusedLayout(32).endWithoutPadding, 32, 64); t.Nmatrix = 64;
    t.output = next64; t.quantize = true;
    kernels_.gemmFp8(commands, t);
  }
  capture(commands, "transition-4-5", *next64);

  // ---- Encoder fused stages 64 / 128 / 256.
  struct FusedStage { Geometry::Level level, next; uint32_t channels; int first, last, levelIndex; };
  const FusedStage encoderStages[] = {{d1, d2, 64, 5, 8, 1}, {d2, d3, 128, 9, 14, 2}, {d3, d4, 256, 15, 22, 3}};
  Activation* skips[3] = {};
  Activation* stageInput = next64;
  for (int s = 0; s < 3; ++s) {
    const FusedStage& stage = encoderStages[s];
    const uint32_t rows = stage.level.width * stage.level.height;
    const uint32_t nextRows = stage.next.width * stage.next.height;
    std::string label = "encoder " + std::to_string(stage.channels);
    Activation* st = stageInput;
    Activation* sc = allocate(label + " state", rows, stage.channels, Format::E4);
    Activation* raw = allocate(label + " raw transition", rows, stage.channels, Format::F16);
    Temporaries temps = createTemporaries(label, rows, stage.channels);
    for (int block = stage.first; block <= stage.last; ++block) {
      encodeFusedBlock(commands, temps, *st, sc, block, stage.channels, stage.level.width, stage.level.height,
                       takeWindowPhase(stage.levelIndex), fusedLayout(stage.channels), model_.tensor(block), nullptr,
                       block == stage.last ? raw : nullptr);
      std::swap(st, sc);
      capture(commands, "block-" + std::to_string(block), *st);
    }
    skips[s] = st;
    Activation* pooled = allocate(label + " downsample", nextRows, stage.channels, Format::E4);
    kernels_.downsample2x(commands, *raw, *pooled, stage.level.width, stage.level.height, stage.next.width,
                          stage.next.height);
    capture(commands, "pooled-" + std::to_string(stage.last) + "-" + std::to_string(stage.last + 1), *pooled);
    Activation* next = allocate(label + " next stage", nextRows, stage.channels * 2, Format::E4);
    GemmFp8Args t;
    t.input = pooled; t.rows = nextRows; t.K = stage.channels; t.N = stage.channels * 2;
    t.weights = model_.fp8Matrix(model_.tensor(stage.last), fusedLayout(stage.channels).endWithoutPadding,
                                 stage.channels, stage.channels * 2);
    t.Nmatrix = stage.channels * 2; t.output = next; t.quantize = true;
    kernels_.gemmFp8(commands, t);
    capture(commands, "transition-" + std::to_string(stage.last) + "-" + std::to_string(stage.last + 1), *next);
    stageInput = next;
  }
  Activation* skip64 = skips[0];
  Activation* skip128 = skips[1];
  Activation* skip256 = skips[2];

  // ---- Encoder 512 (split blocks 23-30) and the pooled ViT input.
  const uint32_t rows4 = d4.width * d4.height;
  {
    Activation* st = stageInput;
    Activation* sc = allocate("encoder 512 state", rows4, 512, Format::E4);
    Activation* raw = allocate("encoder 512 raw transition", rows4, 512, Format::F16);
    SplitTemporaries temps = createSplitTemporaries("encoder 512", rows4);
    for (int block = 23; block <= 30; ++block) {
      encodeSplitBlock(commands, temps, *st, sc, block, d4.width, d4.height, takeWindowPhase(4),
                       block == 30 ? raw : nullptr);
      std::swap(st, sc);
      capture(commands, "block-" + std::to_string(block), *st);
    }
    Activation* skip512 = st;
    const uint32_t tokens = g.vitTokens();
    Activation* pooled = allocate("encoder 512 pooled", tokens, 512, Format::E4);
    kernels_.downsample2x(commands, *raw, *pooled, d4.width, d4.height, d5.width, d5.height);
    Activation* vitState = allocate("ViT state", tokens, 1024, Format::E4);
    GemmFp8Args t;
    t.input = pooled; t.rows = tokens; t.K = 512; t.N = 1024;
    t.weights = model_.fp8Matrix(model_.tensor(30, 4), 0, 512, 1024); t.Nmatrix = 1024;
    t.output = vitState; t.quantize = true;
    kernels_.gemmFp8(commands, t);

    // ---- ViT 31-38.
    encodeVit(commands, *vitState, tokens);

    // ---- Decoder 512 (39-47): projected ViT output upsampled onto the encoder skip.
    Activation* projected512 = allocate("decoder 512 projection", tokens, 512, Format::F16);
    GemmFp8Args p;
    p.input = vitState; p.rows = tokens; p.K = 1024; p.N = 512; p.partition = 256;
    p.weights = model_.fp8Matrix(model_.tensor(39), 0, 1024, 512); p.Nmatrix = 512;
    p.output = projected512; p.quantize = false;
    kernels_.gemmFp8(commands, p);
    Activation* merged = allocate("decoder 512 skip merge", rows4, 512, Format::E4);
    kernels_.upsampleResidual(commands, *projected512, *skip512, model_.tensor(39), 1024 * 512, *merged, nullptr,
                              d5.width, d5.height, d4.width, d4.height);
    capture(commands, "block-39", *merged);
    Activation* dst = merged;
    Activation* dsc = allocate("decoder 512 state", rows4, 512, Format::E4);
    SplitTemporaries dtemps = createSplitTemporaries("decoder 512", rows4);
    for (int block = 40; block <= 47; ++block) {
      encodeSplitBlock(commands, dtemps, *dst, dsc, block, d4.width, d4.height, takeWindowPhase(4), nullptr);
      std::swap(dst, dsc);
      capture(commands, "block-" + std::to_string(block), *dst);
    }
    stageInput = dst;
  }

  // ---- Decoder fused stages 256 / 128 / 64 / 32.
  struct DecoderStage { Geometry::Level low, high; uint32_t channels; int first, last, levelIndex; Activation* skip; };
  const DecoderStage decoderStages[] = {{d4, d3, 256, 48, 55, 3, skip256}, {d3, d2, 128, 56, 61, 2, skip128},
                                        {d2, d1, 64, 62, 65, 1, skip64}, {d1, d0, 32, 66, 69, 0, skip32}};
  for (const DecoderStage& stage : decoderStages) {
    const uint32_t lowRows = stage.low.width * stage.low.height;
    const uint32_t rows = stage.high.width * stage.high.height;
    std::string label = "decoder " + std::to_string(stage.channels);
    const Tensor& transition = model_.tensor(stage.first);
    FusedLayout layout = upsampleFusedLayout(stage.channels * 2, stage.channels);
    if (transition.byteLength != layout.endWithoutPadding + 16)
      throw std::runtime_error("unexpected upsample layout for block " + std::to_string(stage.first));
    Activation* projection = allocate(label + " projection", lowRows, stage.channels, Format::F16);
    GemmFp8Args p;
    p.input = stageInput; p.rows = lowRows; p.K = stage.channels * 2; p.N = stage.channels;
    p.weights = model_.fp8Matrix(transition, layout.upsampleWeight, stage.channels * 2, stage.channels);
    p.Nmatrix = stage.channels; p.output = projection; p.quantize = false;
    kernels_.gemmFp8(commands, p);
    Activation* merged = allocate(label + " skip merge", rows, stage.channels, Format::E4);
    Activation* rawMerged =
        stage.channels == 32 ? allocate(label + " raw skip merge", rows, 32, Format::F16) : nullptr;
    kernels_.upsampleResidual(commands, *projection, *stage.skip, transition, layout.transitionScale, *merged,
                              rawMerged, stage.low.width, stage.low.height, stage.high.width, stage.high.height);
    Activation* st = merged;
    Activation* sc = allocate(label + " state", rows, stage.channels, Format::E4);
    Temporaries temps = createTemporaries(label, rows, stage.channels);
    for (int block = stage.first; block <= stage.last; ++block) {
      int index = block - stage.first;
      encodeFusedBlock(commands, temps, *st, sc, block, stage.channels, stage.high.width, stage.high.height,
                       takeWindowPhase(stage.levelIndex),
                       block == stage.first ? layout : fusedLayout(stage.channels), model_.tensor(block),
                       index == 0 ? rawMerged : nullptr, nullptr);
      std::swap(st, sc);
      capture(commands, "block-" + std::to_string(block), *st);
    }
    stageInput = st;
  }

  // ---- Full-resolution post block 70 and the RGBA head.
  {
    const Tensor& tensor = model_.tensor(70);
    FusedLayout layout = postFusedLayout();
    if (tensor.byteLength != layout.endWithoutPadding) throw std::runtime_error("unexpected block70 layout");
    head_ = allocate("RGBA neural head", fullRows, 4, Format::F32);
    Activation* rawMerged = allocate("post raw merge", fullRows, 32, Format::F16);
    Activation* merged = allocate("post merge", fullRows, 32, Format::E4);
    kernels_.postBlend(commands, *stageInput, *adapter, tensor, layout.inputScale, layout.adapterScale, *rawMerged,
                       *merged, d0.width, d0.height, g.fullWidth, g.fullHeight);
    Activation* rawBlockOutput = allocate("post raw block output", fullRows, 32, Format::F16);
    Temporaries temps = createTemporaries("post", fullRows, 32);
    encodeFusedBlock(commands, temps, *merged, nullptr, 70, 32, g.fullWidth, g.fullHeight,
                     takeWindowPhase(6), layout, tensor, rawMerged, rawBlockOutput);
    uint32_t paddedN = 0;
    GemmF16Args post;
    post.input = rawBlockOutput;
    post.weights = model_.f16Matrix(tensor, layout.postWeights, 32, 4, paddedN);
    post.paddedN = paddedN; post.output = head_; post.rows = fullRows; post.K = 32; post.N = 4;
    kernels_.gemmF16(commands, post);
  }
}

}  // namespace metal
}  // namespace nr
