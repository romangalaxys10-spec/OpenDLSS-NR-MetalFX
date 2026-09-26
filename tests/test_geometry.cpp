// test_geometry.cpp - field alignment and layout arithmetic.
#include "test_main.hpp"
#include "nr_geometry.h"

NR_TEST(geometry_1080p) {
  nr::Geometry g = nr::Geometry::fromValid(1920, 1080);
  NR_CHECK(g.fullWidth >= 1920 && g.fullHeight >= 1080);
  NR_CHECK(g.levels[0].width % 8 == 0 && g.levels[0].height % 8 == 0);
  for (int level = 1; level < 6; ++level) {
    NR_CHECK(g.levels[level].width <= g.levels[level - 1].width);
    NR_CHECK(g.levels[level].height <= g.levels[level - 1].height);
  }
  NR_CHECK(g.paddedVitTokens() % 64 == 0);
}

NR_TEST(geometry_square_min) {
  nr::Geometry g = nr::Geometry::fromValid(768, 768);
  // The padded field carries the native asymmetric width rule (one extra
  // alignment step when both axes hit four alignments); both must exceed the
  // valid size and level 0 must tile with 8-pixel windows.
  NR_CHECK(g.fullWidth >= 768 && g.fullHeight >= 768);
  NR_CHECK(g.levels[0].width % 8 == 0 && g.levels[0].height % 8 == 0);
  NR_CHECK(g.vitTokens() > 0);
}

NR_TEST(layout_offsets_ordered) {
  nr::FusedLayout l = nr::fusedLayout(128);
  NR_CHECK(l.expand < l.contractWeights);
  NR_CHECK(l.contractWeights < l.ffnCosSkip);
  NR_CHECK(l.ffnCosSkip < l.qkv);
  NR_CHECK(l.qkv < l.relative);
  NR_CHECK(l.relative < l.scale);
  NR_CHECK(l.scale < l.projection);
  NR_CHECK(l.projection < l.attnCosSkip);
  NR_CHECK(l.attnCosSkip < l.endWithoutPadding);
  nr::FusedLayout pre = nr::preFusedLayout();
  NR_CHECK(pre.inputAdapter == 8208);
  nr::FusedLayout post = nr::postFusedLayout();
  NR_CHECK(post.postWeights == 20784);
}

NR_TEST(window_phase_cycle) {
  uint32_t x, y;
  nr::windowPhase(0, x, y);
  NR_CHECK(x == 0 && y == 0);
  nr::windowPhase(1, x, y);
  NR_CHECK(x == 4 && y == 4);
  nr::windowPhase(2, x, y);
  NR_CHECK(x == 4 && y == 0);
  nr::windowPhase(3, x, y);
  NR_CHECK(x == 0 && y == 4);
}
