// The pipeline the runtime drives: take a rendered frame, run the network on it, put the result on the canvas.
//
// This is the only place the demo and the port meet. The runtime hands over a frame that has already crossed
// from WebGL on the CPU, in two typed arrays: RGBA16F colour and RG16F velocity in uv units. Two slots let it
// stage the next frame while this one is still running, which is why every call takes a slot index.
//
// Everything below the `Network` call belongs to the port and is shared with the parity harness: the same
// graph, the same kernels, the same bytes.

import { Network } from '../../../src/network.js';
import { align } from '../../../src/gpu.js';
import { normalizeNrSettings } from './nr-settings.js';

const SLOTS = 2;
const PARAM_WORDS = 18;

export class DlssNrProductionPipeline {
  static async create(inference, canvas, options = {}) {
    const pipeline = new DlssNrProductionPipeline();
    pipeline.inference = inference;
    pipeline.device = inference.device;
    pipeline.canvas = canvas;
    pipeline.paperWhite = options.paperWhite ?? 1;
    pipeline.colorStrength = options.colorStrength ?? 1;
    pipeline.settings = normalizeNrSettings({});
    pipeline.busy = false;
    pipeline.network = null;
    pipeline.geometry = { validWidth: 0, validHeight: 0 };
    pipeline.frameIndex = 0;
    pipeline.context = canvas.getContext('webgpu');
    pipeline.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    return pipeline;
  }

  /**
   * Build the graph for this field if it is not the one already standing. Recording the graph is asynchronous
   * where the rest of the pipeline is not, which is why the runtime awaits this.
   */
  async ensureGeometry(runtimeProfile) {
    const [width, height] = runtimeProfile.sourceDimensions;
    if (this.network && this.geometry.validWidth === width && this.geometry.validHeight === height) return;
    await this.build(runtimeProfile);
  }

  async build(runtimeProfile) {
    const [width, height] = runtimeProfile.sourceDimensions;
    if (this.network) { this.network.destroy(); this.network = null; }
    const device = this.device;
    const resources = {};

    const network = await Network.create({
      weights: this.inference.model.weightsUrl,
      width,
      height,
      device,
      model: this.inference.model.weights,
      shaderBase: new URL('/', location.href),
      onProgress: this.inference.onProgress ?? undefined,
      extraShaders: [{ path: 'shaders/frame.wgsl', entryPoints: ['input_features', 'compose'] }],
      before: (net) => {
        this.blendScale = net.model.blendScale();
        // Two staged frames, copied into the graph's own inputs when their turn comes.
        resources.slots = Array.from({ length: SLOTS }, (unused, index) => ({
          color: device.createBuffer({ label: `staged colour ${index}`, size: width * height * 8,
                                       usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
          motion: device.createBuffer({ label: `staged velocity ${index}`, size: width * height * 4,
                                        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
        }));
        resources.scene = device.createBuffer({ label: 'rendered frame', size: width * height * 8,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        resources.motion = device.createBuffer({ label: 'velocity', size: width * height * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        resources.history = [0, 1].map((index) => device.createBuffer({
          label: `history ${index}`, size: width * height * 8,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }));
        resources.image = device.createBuffer({ label: 'presented image',
          size: align(width * 4, 256) * height,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const g = net.geometry;
        net.recorder.pass('input_features', {
          1: resources.scene, 2: resources.history[0], 4: resources.motion, 5: net.features.buffer,
        }, this.params(net, this.settings, false), [Math.ceil(g.fullWidth / 8), Math.ceil(g.fullHeight / 8)],
           'input features');
        resources.featuresPass = net.recorder.passes.at(-1);
      },
      after: (net) => {
        net.recorder.pass('compose', {
          1: resources.scene, 2: resources.history[0], 3: net.graph.head.buffer, 4: resources.motion,
          6: resources.history[1], 7: resources.image,
        }, this.params(net, this.settings, false), [Math.ceil(width / 8), Math.ceil(height / 8)], 'compose');
        resources.composePass = net.recorder.passes.at(-1);
        // The graph is recorded once, so the history buffers cannot alternate by frame parity; one is read,
        // the other written, and a copy swaps them.
        net.recorder.copy(resources.history[1], resources.history[0], width * height * 8, 'history swap');
      },
    });

    this.canvas.width = width;
    this.canvas.height = height;
    this.context.configure({ device, format: this.canvasFormat, alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST });

    this.network = network;
    this.blendScale = network.model.blendScale();
    this.resources = resources;
    this.geometry = { validWidth: width, validHeight: height,
                      fullWidth: network.geometry.fullWidth, fullHeight: network.geometry.fullHeight };
    this.historyValid = false;
  }

  params(network, settings, historyValid) {
    const buffer = new ArrayBuffer(PARAM_WORDS * 4);
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    const g = network.geometry;
    words[0] = g.fullWidth;
    words[1] = g.fullHeight;
    words[2] = g.validWidth;
    words[3] = g.validHeight;
    words[4] = this.frameIndex & 0xffff;
    words[5] = historyValid ? 1 : 0;
    words[6] = settings.enabled === false ? 0 : 1;
    words[7] = align(g.validWidth * 4, 256) / 4;
    floats[8] = this.paperWhite;
    floats[9] = settings.style ?? 0;
    floats[10] = settings.localTone ?? 1;
    floats[11] = settings.localStructure ?? 1;
    floats[12] = settings.skinStructure ?? -1;
    floats[13] = settings.autoMask ? 1 : 0;
    floats[14] = this.blendScale;
    floats[15] = settings.intensity ?? 1;
    floats[16] = this.colorStrength;
    words[17] = 1;   // the frame came from WebGL: its rows and its velocity v are flipped
    return words;
  }

  /** Stage a frame. Returns the milliseconds spent, which the runtime reports as upload time. */
  upload(slotIndex, rgba16, motion = null) {
    const slot = this.resources?.slots?.[slotIndex];
    const { validWidth, validHeight } = this.geometry;
    if (!slot) throw new Error('DLSS-NR production pipeline has no geometry yet');
    if (rgba16.byteLength !== validWidth * validHeight * 8) {
      throw new Error(`DLSS-NR production upload has ${rgba16.byteLength} bytes, expected ` +
                      `${validWidth * validHeight * 8}`);
    }
    const started = performance.now();
    this.device.queue.writeBuffer(slot.color, 0, rgba16);
    if (motion) {
      if (motion.byteLength !== validWidth * validHeight * 4) throw new Error('NR motion geometry mismatch');
      this.device.queue.writeBuffer(slot.motion, 0, motion);
    }
    return performance.now() - started;
  }

  async render(slotIndex, runtimeProfile, settings = this.settings, temporal = {}) {
    const controls = normalizeNrSettings(settings);
    if (this.busy) throw new Error('Production inference is already in flight');
    await this.ensureGeometry(runtimeProfile);
    this.busy = true;
    try {
      const { device, network, resources } = this;
      const historyValid = this.historyValid && !temporal.reset;
      const words = this.params(network, controls, historyValid);
      device.queue.writeBuffer(network.recorder.paramsBuffer, resources.featuresPass.index * 256, words);
      device.queue.writeBuffer(network.recorder.paramsBuffer, resources.composePass.index * 256, words);

      const preprocessStart = performance.now();
      const slot = resources.slots[slotIndex];
      const stage = device.createCommandEncoder({ label: 'stage' });
      stage.copyBufferToBuffer(slot.color, 0, resources.scene, 0, resources.scene.size);
      stage.copyBufferToBuffer(slot.motion, 0, resources.motion, 0, resources.motion.size);
      device.queue.submit([stage.finish()]);
      const preprocessMilliseconds = performance.now() - preprocessStart;

      const networkStart = performance.now();
      await network.run();
      const networkMilliseconds = performance.now() - networkStart;

      const presentStart = performance.now();
      const { validWidth: width, validHeight: height } = this.geometry;
      const present = device.createCommandEncoder({ label: 'present' });
      present.copyBufferToTexture({ buffer: resources.image, bytesPerRow: align(width * 4, 256) },
                                  { texture: this.context.getCurrentTexture() }, { width, height });
      device.queue.submit([present.finish()]);
      const presentationMilliseconds = performance.now() - presentStart;

      this.historyValid = true;
      this.frameIndex += 1;
      return { neural: this.canvas,
               timings: { preprocessMilliseconds, networkMilliseconds, presentationMilliseconds } };
    } finally {
      this.busy = false;
    }
  }

  resetHistory() { this.historyValid = false; }

  destroy() {
    for (const slot of this.resources?.slots ?? []) { slot.color.destroy(); slot.motion.destroy(); }
    for (const key of ['scene', 'motion', 'image']) this.resources?.[key]?.destroy();
    for (const buffer of this.resources?.history ?? []) buffer.destroy();
    this.network?.destroy();
    this.network = null;
    this.resources = null;
    this.geometry = { validWidth: 0, validHeight: 0 };
  }
}
