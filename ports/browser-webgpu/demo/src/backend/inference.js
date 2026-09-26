// The geometry the runtime reasons about, and the handle it holds between the model and the pipeline.
//
// The graph itself lives in the port (src/graph.js); nothing here builds kernels. This exists because the
// runtime asks the backend for the network's field shape before it has a pipeline, and keeps a handle it can
// hang resolution-dependent limits on.

import { geometryFromValid } from '../../../src/geometry.js';
import { roundF16 } from '../../../src/numerics.js';

/** The padded field and its six pooling levels, in the shape the runtime reads. */
export function createRuntimeProfile(validWidth, validHeight) {
  if (!Number.isInteger(validWidth) || !Number.isInteger(validHeight) || validWidth < 1 || validHeight < 1) {
    throw new Error('Runtime profile dimensions must be positive integers');
  }
  const geometry = geometryFromValid(validWidth, validHeight);
  return {
    label: `Network geometry ${validWidth}x${validHeight}`,
    sourceDimensions: [validWidth, validHeight],
    fullDimensions: [geometry.fullWidth, geometry.fullHeight],
    dimensions: geometry.levels.map((level) => [level.width, level.height]),
    geometry,
  };
}

/** One publication to the half grid, which the runtime uses when it builds a proxy on the CPU. */
export const roundNativeF16 = roundF16;

export class DlssNrBrowserInference {
  static async create(model, onProgress, { maxVitTokens = 0 } = {}) {
    const inference = new DlssNrBrowserInference();
    inference.model = model;
    inference.device = model.device;
    inference.maxVitTokens = maxVitTokens;
    // The runtime sets a storage-binding limit here so that asking for resize headroom cannot change which
    // numerical path the graph takes. This port's graph has no such choice: its tiling is fixed.
    inference.hierarchy = { storageBindingLimit: 0 };
    // The runtime hands its progress reporter here and nowhere else, so this is where the pipeline finds it.
    // Recording the graph is the long wait at startup - a few hundred pipelines to compile - and without
    // this the overlay would sit on whatever it last said for the whole of it.
    inference.onProgress = onProgress ?? null;
    return inference;
  }

  async ensureProductionTokens(tokens) { this.maxVitTokens = Math.max(this.maxVitTokens, tokens); }

  /**
   * Discard the recorded graph. The runtime does this once at startup: the first graph is recorded while the
   * browser is still compiling the cold shader set, and it throws that one away rather than present it. Here
   * the graph belongs to the pipeline, which the runtime destroys alongside this call, so there is nothing
   * left to drop - the device, the weights and the compiled kernels are all retained, which is the point.
   */
  resetProductionGraph() {}

  /** The diagnostic path, which this port does not carry. */
  async run() { throw new Error('the diagnostic inference path is not part of this port'); }
}
