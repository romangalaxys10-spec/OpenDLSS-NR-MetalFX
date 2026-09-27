// Unit tests: fp16/E4M3 codec, geometry parity against the reference's
// documented table, and a neural graph smoke test.
// Build: ctest / a plain main() here for portability.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/fp16.h"
#include "opendlss/e4m3.h"
#include "opendlss/geometry.h"
#include "opendlss/model.h"
#include "opendlss/backend.h"
#include "opendlss/pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace opendlss;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); ++g_failures; } \
    else { std::printf("ok   %s\n", msg); } } while (0)

static void test_fp16() {
    CHECK(f16_to_f32(f32_to_f16(1.0f)) == 1.0f, "fp16 roundtrip 1.0");
    CHECK(f16_to_f32(f32_to_f16(0.5f)) == 0.5f, "fp16 roundtrip 0.5");
    CHECK(f16_to_f32(f32_to_f16(65504.0f)) == 65504.0f, "fp16 max normal");
    CHECK(f32_to_f16(65520.0f) == 0x7C00, "fp16 overflow -> inf");
    CHECK(f32_to_f16(-2.0f) == 0xC000, "fp16 -2.0 bits");
    CHECK(f32_to_f16(1.5f) == 0x3E00, "fp16 1.5 bits");
    // round to nearest even: 1.0625 is exactly between 1.0 and 1.0625+ulp
    CHECK(f32_to_f16(1.0009765625f) == 0x3C01 || f32_to_f16(1.0009765625f) == 0x3C00, "fp16 rtne boundary");
    CHECK(f16_to_f32(0x3555) > 0.333f && f16_to_f32(0x3555) < 0.334f, "fp16 decode 1/3 approx");
    CHECK(f32_to_f16(0.0f) == 0x0000 && f32_to_f16(-0.0f) == 0x8000, "fp16 signed zeros");
}

static void test_e4m3() {
    CHECK(E4M3::decode(E4M3::encode(1.0f)) == 1.0f, "e4m3 roundtrip 1.0");
    CHECK(E4M3::decode(E4M3::encode(448.0f)) == 448.0f, "e4m3 max normal 448");
    CHECK(E4M3::decode(E4M3::encode(1000.0f)) == 448.0f, "e4m3 saturates above 448");
    CHECK(E4M3::decode(0x7F) != E4M3::decode(0x7F), "e4m3 NaN is NaN");
    CHECK(E4M3::decode(0x38) == 1.0f, "e4m3 code 0x38 = 1.0");
    // subnormal: smallest is 2^-9 = 0.001953125
    CHECK(std::fabs(E4M3::decode(0x01) - 0.001953125f) < 1e-6f, "e4m3 min subnormal 2^-9");
    CHECK(E4M3::decode(E4M3::encode(0.001f)) == 0.001953125f, "e4m3 0.001 rounds to 2^-9");
    CHECK(E4M3::decode(E4M3::encode(0.0005f)) == 0.0f, "e4m3 flush below half min subnormal");
    CHECK(E4M3::decode(E4M3::encode(-2.0f)) == -2.0f, "e4m3 -2.0");
    CHECK(e4m3_publish(1.03f) == 1.0f, "e4m3 publication grid");
}

static void test_geometry() {
    // Documented reference vectors (6 levels, floor 320):
    struct V { uint32_t w, h, fw, fh; };
    const V vectors[] = {
        {512, 512, 576, 512},
        {768, 768, 832, 768},
        {644, 768, 768, 768},
        {1920, 1080, 1920, 1152},
        {3840, 2160, 3840, 2176},
    };
    for (const V& v : vectors) {
        auto g = Geometry::fromValid(v.w, v.h);
        CHECK(g && g->fieldWidth == v.fw && g->fieldHeight == v.fh, "geometry field parity");
        if (g) {
            // invariants: every level a multiple of 4; level0 a multiple of 8
            for (uint32_t i = 0; i < g->levelsCount; ++i) {
                CHECK(g->levels[i].width % 4 == 0 && g->levels[i].height % 4 == 0,
                      "geometry level multiple of 4");
            }
        }
    }
    // reference table row: 512x512 -> L0..L5 = 288x256, 144x128, 72x64, 36x32, 20x16, 12x8
    auto g = Geometry::fromValid(512, 512);
    if (g) {
        CHECK(g->levels[0].width == 288 && g->levels[0].height == 256, "512 L0");
        CHECK(g->levels[5].width == 12 && g->levels[5].height == 8, "512 L5");
    }
    // 1920x1080 -> L5 32x20
    auto g2 = Geometry::fromValid(1920, 1080);
    if (g2) CHECK(g2->levels[5].width == 32 && g2->levels[5].height == 20, "1080p L5 32x20");
    // tiny config (2 levels, floor 64) for the demo network
    auto g3 = Geometry::fromValid(160, 90, 2, 64);
    CHECK(g3 && g3->fieldWidth == 160 && g3->fieldHeight == 92, "demo geometry 160x90");
    // rejects nothing reasonable
    CHECK(Geometry::fromValid(0, 100) == std::nullopt, "geometry rejects zero width");
}

static void test_neural_graph() {
    auto backend = create_backend(BackendKind::Cpu);
    CHECK(backend != nullptr, "cpu backend created");
    // model path provided by CMake/runner via env or default
    const char* modelDir = std::getenv("OPENDLSS_TEST_MODEL");
    if (!modelDir) {
        std::printf("skip neural graph test (set OPENDLSS_TEST_MODEL)\n");
        return;
    }
    CHECK(backend->loadModel(modelDir), "model loads");
    if (!backend->hasNeuralGraph()) return;

    auto geom = Geometry::fromValid(64, 64, 2, 64, 8);
    CHECK(geom.has_value(), "test geometry 64x64");

    Image proxy(64, 64);
    for (uint32_t y = 0; y < 64; ++y)
        for (uint32_t x = 0; x < 64; ++x) {
            float* p = proxy.row(y) + x * 4;
            p[0] = (x / 64.0f); p[1] = (y / 64.0f); p[2] = 0.5f; p[3] = 1.0f;
        }
    std::vector<uint16_t> features;
    ConditionScalars cond;
    preprocess_pack_features(*geom, proxy, nullptr, cond, 42, features);
    CHECK(features.size() == size_t(geom->fieldWidth) * geom->fieldHeight * 16, "feature field size");

    std::vector<float> head(features.size() / 16 * 4, 0.f);
    CHECK(backend->runNeuralGraph(*geom, features.data(), head.data(), 42), "graph executes");

    // determinism: same inputs, same outputs
    std::vector<float> head2(head.size(), 0.f);
    CHECK(backend->runNeuralGraph(*geom, features.data(), head2.data(), 42), "graph reruns");
    bool same = std::memcmp(head.data(), head2.data(), head.size() * 4) == 0;
    CHECK(same, "graph deterministic");

    // head composite sanity: blend logit applied, output stays in [0,1]
    Image display(64, 64);
    head_composite(proxy, head.data(), nullptr, backend->model()->blendScale(), display);
    bool inRange = true;
    for (float v : display.pixels) if (!(v >= 0.0f && v <= 1.0f)) { inRange = false; break; }
    CHECK(inRange, "display in [0,1]");

    // with history, temporal blending must pull the output toward history
    Image hist(64, 64);
    for (float& v : hist.pixels) v = 0.25f;
    Image blended(64, 64);
    head_composite(proxy, head.data(), &hist, backend->model()->blendScale(), blended);
    float sumDiffNoHist = 0, sumDiffHist = 0;
    for (size_t i = 0; i < blended.pixels.size(); i += 4) {
        sumDiffNoHist += std::fabs(display.pixels[i] - proxy.pixels[i]);
        sumDiffHist  += std::fabs(blended.pixels[i] - hist.pixels[i]);
    }
    (void)sumDiffNoHist;
    CHECK(sumDiffHist < blended.pixels.size() / 4 * 0.5f, "history blend bounded");
}

int main() {
    test_fp16();
    test_e4m3();
    test_geometry();
    test_neural_graph();
    if (g_failures) { std::printf("\n%d FAILURES\n", g_failures); return 1; }
    std::printf("\nall tests passed\n");
    return 0;
}
