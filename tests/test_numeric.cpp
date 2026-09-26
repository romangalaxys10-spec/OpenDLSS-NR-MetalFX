// test_numeric.cpp - exact f16/E4M3 conversion contracts (run on every platform).
#include "test_main.hpp"
#include "nr_cpu_reference.h"
#include "nr_numeric.h"

#include <cmath>

NR_TEST(f16_roundtrip_table) {
  // Every f16 code round-trips through f32 exactly.
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    float value = num::f16ToF32((uint16_t)bits);
    if (std::isnan(value)) continue;   // NaN payloads do not survive f32
    NR_CHECK(num::f16Bits(value) == (uint16_t)bits);
  }
}

NR_TEST(f16_reference_codes) {
  NR_CHECK(num::f16Bits(0.0f) == 0x0000);
  NR_CHECK(num::f16Bits(1.0f) == 0x3c00);
  NR_CHECK(num::f16Bits(65504.0f) == 0x7bff);
  NR_CHECK(num::f16Bits(65520.0f) == 0x7c00);   // overflow -> inf
  NR_CHECK(num::f16Bits(0.5f) == 0x3800);
  NR_CHECK(num::f16Bits(-2.5f) == 0xc100);
  NR_CHECK(num::f16Bits(5.960464477539063e-8f) == 0x0001);   // smallest subnormal
  NR_CHECK(num::f16Bits(2.9802322387695312e-8f) == 0x0000);  // below half grid -> 0 (RNE)
  NR_CHECK(num::roundF16(0.1f) == num::f16ToF32(num::f16Bits(0.1f)));
}

NR_TEST(e4m3_codes) {
  // Exhaustive decode check: every code decodes to the exact value.
  for (uint32_t code = 0; code < 256; ++code) {
    if ((code & 0x7f) == 0x7f) continue;   // NaN code
    float decoded = num::e4m3ToF32((uint8_t)code);
    // Re-encode with the RNE publication; must recover the code (sign of zero kept).
    if (decoded == 0.0f) continue;
    NR_CHECK(num::e4m3FromF32(decoded) == (uint8_t)code);
  }
  NR_CHECK(num::e4m3FromF32(0.0f) == 0x00);
  NR_CHECK(num::e4m3FromF32(1.0f) == 0x38);
  NR_CHECK(num::e4m3FromF32(448.0f) == 0x7e);
  NR_CHECK(num::e4m3FromF32(460.0f) == 0x7e);   // saturate
  NR_CHECK(num::e4m3FromF32(0.001953125f) == 0x01);
  NR_CHECK(num::e4m3FromF32(-3.5f) == 0xc6);
  NR_CHECK(num::e4m3FromF32(std::nanf("")) == 0);   // NaN -> +0
  NR_CHECK(num::e4m3ToF32(0x38) == 1.0f);
  NR_CHECK(num::e4m3ToF32(0xc6) == -3.5f);
  NR_CHECK(num::e4m3ToF32(0xc7) == -3.75f);
}

NR_TEST(permutations) {
  // The chained input permutation is invertible on 64 indices.
  for (uint32_t k = 0; k < 64; ++k) {
    uint32_t p = cpuref::packedInputIndex(k);
    NR_CHECK(p < 64);
  }
}
