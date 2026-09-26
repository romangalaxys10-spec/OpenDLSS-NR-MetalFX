// nr_geometry.h - portable geometry/layout header: resolves the backend's
// canonical definitions (nr::metal on Apple, nr:: on Windows/Linux).
#pragma once

#ifdef __APPLE__
  #include "nr_metal.h"
  namespace nr {
    using metal::Geometry;
    using metal::FusedLayout;
    using metal::fusedLayout;
    using metal::preFusedLayout;
    using metal::upsampleFusedLayout;
    using metal::postFusedLayout;
    using metal::windowPhase;
  }  // namespace nr
#else
  #include "nr_graph.h"
  // The upstream windowPhase is file-local (nr_graph.cpp); the shim re-exports
  // the identical table for tests and the shared pipeline.
  namespace nr {
    inline void windowPhase(uint32_t index, uint32_t& shiftX, uint32_t& shiftY) {
      static const uint32_t phases[4][2] = {{0, 0}, {4, 4}, {4, 0}, {0, 4}};
      shiftX = phases[index & 3][0];
      shiftY = phases[index & 3][1];
    }
  }  // namespace nr
#endif
