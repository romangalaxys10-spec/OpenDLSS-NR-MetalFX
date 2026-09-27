// Geometry implementation — see geometry.h for the specification being ported.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/geometry.h"

#include <cstdio>

namespace opendlss {

namespace {

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// How many powers of two the graph shrinks a dimension by, per the reference:
// every halving counts, plus one extra when level 0 is not a whole number of
// 8-pixel windows (the decoder's whole-window upsample + crop rule).
uint32_t reductions_for(uint32_t v, uint32_t levelsCount, uint32_t windowSize) {
    uint32_t size = v;
    uint32_t reductions = 0;
    for (uint32_t level = 0; level < levelsCount; ++level) {
        uint32_t half = align_up(ceil_div(size, 2), 4);
        if (half < size) ++reductions;
        if (level == 0 && (half % windowSize) != 0) ++reductions;
        size = half;
    }
    return reductions;
}

} // namespace

std::optional<Geometry> Geometry::fromValid(uint32_t validW, uint32_t validH,
                                            uint32_t levelsCount, uint32_t minField,
                                            uint32_t windowSize) {
    if (levelsCount == 0 || levelsCount > kMaxLevels) return std::nullopt;
    if (validW == 0 || validH == 0) return std::nullopt;

    const uint32_t alignW = 1u << reductions_for(validW, levelsCount, windowSize);
    const uint32_t alignH = 1u << reductions_for(validH, levelsCount, windowSize);

    uint32_t fieldW = align_up(validW, alignW);
    uint32_t fieldH = align_up(validH, alignH);
    fieldW = fieldW > minField ? fieldW : minField;
    fieldH = fieldH > minField ? fieldH : minField;

    // The reference's unexplained final quirk, reproduced because the field size
    // decides the window grid and therefore the result inside the valid rectangle too.
    if (fieldW % (4 * alignW) == 0 && fieldH % (4 * alignH) == 0) fieldW += alignW;

    Geometry g;
    g.validWidth = validW;
    g.validHeight = validH;
    g.fieldWidth = fieldW;
    g.fieldHeight = fieldH;
    g.levelsCount = levelsCount;
    g.windowSize = windowSize;

    uint32_t w = fieldW, h = fieldH;
    for (uint32_t level = 0; level < levelsCount; ++level) {
        w = poolSize(w);
        h = poolSize(h);
        g.levels[level] = Level{w, h};
    }
    return g;
}

std::string geometry_to_string(const Geometry& g) {
    std::string s;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "valid %ux%u -> field %ux%u",
                  g.validWidth, g.validHeight, g.fieldWidth, g.fieldHeight);
    s += buf;
    for (uint32_t i = 0; i < g.levelsCount; ++i) {
        std::snprintf(buf, sizeof(buf), ", L%u %ux%u", i, g.levels[i].width, g.levels[i].height);
        s += buf;
    }
    return s;
}

} // namespace opendlss
