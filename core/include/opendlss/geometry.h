// Padded-field geometry — a faithful port of the reference implementation's
// Geometry::fromValid (see the original OpenDLSS-NR docs/network.md), extended
// to a configurable number of pooling levels so small demo networks can use a
// small floor. With levelsCount = 6 and minField = 320 the numbers are exactly
// the reference's (e.g. 1920x1080 -> field 1920x1152).
//
//   level[k] = alignUp(ceil(level[k-1] / 2), 4),  level[-1] = the padded field
//
//   reductions(v):  size = v
//                   for level in 0..L-1:
//                       half = alignUp(ceil(size / 2), 4)
//                       if half < size:              reductions += 1
//                       if level == 0 and half % 8:  reductions += 1   (whole 8-px windows)
//                       size = half
//   align = 1 << reductions
//   field = max(minField, alignUp(valid, align))
//   if fieldW % (4*alignW) == 0 and fieldH % (4*alignH) == 0: fieldW += alignW
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace opendlss {

// alignUp shared helper (kept out of std for C++20 parity across targets).
inline uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

struct Level {
    uint32_t width = 0, height = 0;
};

struct Geometry {
    uint32_t validWidth = 0, validHeight = 0;   // the requested size
    uint32_t fieldWidth = 0, fieldHeight = 0;   // the padded field
    Level levels[8];                            // 0..levelsCount-1
    uint32_t levelsCount = 0;
    uint32_t windowSize = 8;                    // attention window (level 0 windows are 8x8)

    static constexpr uint32_t kMaxLevels = 8;

    // Compute the padded field and every level size. Returns nullopt for sizes
    // the graph cannot run (too small for whole windows after the floor, etc.).
    static std::optional<Geometry> fromValid(uint32_t validW, uint32_t validH,
                                             uint32_t levelsCount = 6,
                                             uint32_t minField = 320,
                                             uint32_t windowSize = 8);
    // Token counts at a level (tokens = width*height of that level).
    uint64_t tokensAt(uint32_t level) const {
        return uint64_t(levels[level].width) * levels[level].height;
    }
    uint64_t fieldTokens() const { return uint64_t(fieldWidth) * fieldHeight; }

    // The 2x2 box-pool halving used between levels: alignUp(ceil(size/2), 4).
    static uint32_t poolSize(uint32_t size) {
        uint32_t half = (size + 1) / 2;         // ceil(size/2)
        return align_up(half, 4);
    }
};

std::string geometry_to_string(const Geometry& g);

} // namespace opendlss
