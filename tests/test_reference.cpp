// test_reference.cpp - the CPU reference of the network arithmetic.
#include "test_main.hpp"
#include "nr_cpu_reference.h"
#include "nr_numeric.h"

#include <cmath>

NR_TEST(silu_cubic_matches_half_grid) {
  // mpCubicSilu is identical to the half-FMA formulation on every f16 input.
  std::vector<uint16_t> table = cpuref::siluTable();
  NR_CHECK(table.size() == 65536);
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    float input = num::f16ToF32((uint16_t)bits);
    float expected = num::f16Bits(cpuref::mpCubicSilu(input));
    NR_CHECK(table[bits] == (uint16_t)expected);
  }
}

NR_TEST(silu_spot_values) {
  NR_CHECK(cpuref::mpCubicSilu(0.0f) == 0.0f);
  float one = cpuref::mpCubicSilu(1.0f);
  NR_CHECK(std::fabs(one - 1.2861328125f) < 0.001f);
  NR_CHECK(std::fabs(cpuref::mpCubicSilu(0.5f) - 0.55224609375f) < 0.001f);
}

NR_TEST(ada_fp8_contract_shape) {
  // 16-product group with a zero accumulator: products of E4M3 domain values.
  float a[16], b[16];
  for (int i = 0; i < 16; ++i) {
    a[i] = (float)(i - 8) * 0.25f;
    b[i] = (float)(i % 4) * 0.5f + 0.25f;
  }
  float value = cpuref::adaFp8Fdpa16(a, b, 16, 0.0f);
  float expected = 0.0f;
  for (int i = 0; i < 16; ++i) expected += a[i] * b[i];
  NR_CHECK(std::fabs(value - expected) < 0.01f);   // bounded by F13 + half rounding
  NR_CHECK(cpuref::adaFp8Fdpa16(a, b, 16, value) != 0.0f);
}

NR_TEST(fp8_domain_rounds_into_grid) {
  for (int i = -800; i <= 800; ++i) {
    float value = (float)i * 0.371f;
    float quantized = cpuref::fp8Domain(value);
    float back = num::e4m3ToF32(num::e4m3FromF32(num::roundF16(quantized)));
    NR_CHECK(std::fabs(back - quantized) < 0.02f + std::fabs(quantized) * 0.01f);
  }
}
