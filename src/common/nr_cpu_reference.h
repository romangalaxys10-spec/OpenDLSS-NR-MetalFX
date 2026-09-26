// nr_cpu_reference.h - portable CPU reference of the network's exact
// arithmetic (FP8 GEMM accumulation order, half publications, window
// attention). Adapted verbatim from the original src/reference.cpp; Tensor
// is replaced by TensorView{bytes, byteLength} so it compiles without any
// GPU headers, on every platform (tests, parity, the Metal backend).
#pragma once
#include <cstdint>
#include <vector>

namespace cpuref {

struct TensorView {
  const uint8_t* bytes = nullptr;
  uint32_t byteLength = 0;
};

float roundF16(float value);
float fp8Domain(float value);
float mpCubicSilu(float value);
std::vector<uint16_t> siluTable();

float adaFp8Fdpa16(const float* a, const float* b, int count, float accumulator);
float adaF16Fdpa8(const float* a, const float* b, int count, float accumulator);

// Packed index permutations (identical to the model layout).
uint32_t packedInputIndex(uint32_t k);
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels);

struct GemmRef {
  const TensorView* tensor = nullptr;
  uint32_t weightByteOffset = 0;
  uint32_t K = 0;
  uint32_t Nmatrix = 0;
  uint32_t weightColumnOffset = 0;
  uint32_t partition = 0;
  bool swizzleInput = true;
};
float gemmFp8Element(const GemmRef& gemm, const float* inputRow, uint32_t column, float initial);

void windowNormalizeRef(const float* qkvRow, uint32_t head, float scale, float* out);

}  // namespace cpuref
