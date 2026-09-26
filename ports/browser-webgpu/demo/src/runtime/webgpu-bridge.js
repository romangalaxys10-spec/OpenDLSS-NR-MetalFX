import {normalizeNrSettings} from '/backend/nr-settings.js';
import {readNrSettings, mountNrControls} from './nr-controls.js';
import {demoUi} from './demo-ui.js';
import {viewerConverged} from './viewer-convergence.js';
import { DlssNrWebGpuModel } from '/backend/model-loader.js';
import {
  createRuntimeProfile,
  DlssNrBrowserInference,
  roundNativeF16,
} from '/backend/inference.js';
import { DlssSrWebGpuModel } from '/backend/sr-model.js';
import { DlssSrBrowserInference } from '/backend/sr-inference.js';
import { DlssNrProductionPipeline } from '/backend/production-pipeline.js';

const outputCanvas = document.querySelector('#dlssWebGpuOutput');
const resizeHoldCanvas = document.querySelector('#dlssResizeHold');
const status = document.querySelector('#dlssWebGpuStatus');
const HALF_TO_FLOAT = buildHalfFloatTable();
const PAPER_WHITE_SCALE = 1;
const TRANSFER_STRENGTH = 1;
const COLOR_STRENGTH = 1;
const DEVICE_CAPACITY_HEADROOM = 1.35;
const RESIZE_SETTLE_MILLISECONDS = 450;
const USE_EXTERNAL_CONTROL_MASK = false;
const diagnosticQuery = new URLSearchParams(location.search);
let sceneLabel = globalThis.dlssSceneLabel || 'Bistro';
const CAPTURE_NATIVE_REFERENCE_HEAD = diagnosticQuery.get('srReferenceHead') === '1';
const CAPTURE_NATIVE_ENC0_DOWN = diagnosticQuery.get('srReferenceEnc0Down') === '1';
const CAPTURE_NATIVE_DEC5_INPUT = diagnosticQuery.get('srReferenceDec5') === '1';
const CAPTURE_NATIVE_DEC4_INPUT = diagnosticQuery.get('srReferenceDec4') === '1';
const CAPTURE_NATIVE_DEC0_INPUT = diagnosticQuery.get('srReferenceDec0') === '1';
const CAPTURE_NR_REFERENCE = diagnosticQuery.get('nrReference') === '1';
const DEC0_PRECISION_SWEEP = diagnosticQuery.get('srDec0Sweep') === '1';
const DEC0_PRECISION = diagnosticQuery.get('srDec0Precision') ?? 'f32';
const DEC5_PRECISION = diagnosticQuery.get('srDec5Precision') ?? 'f32';
const DEC4_PRECISION = diagnosticQuery.get('srDec4Precision') ?? 'f32';
const PRODUCTION_RUNTIME = !(CAPTURE_NATIVE_REFERENCE_HEAD || CAPTURE_NATIVE_ENC0_DOWN
  || CAPTURE_NATIVE_DEC5_INPUT || CAPTURE_NATIVE_DEC4_INPUT || CAPTURE_NATIVE_DEC0_INPUT
  || CAPTURE_NR_REFERENCE || DEC0_PRECISION_SWEEP);
const outputContext = PRODUCTION_RUNTIME
  ? null : outputCanvas.getContext('2d', { alpha: false });
const SR_CAPTURE_SET = CAPTURE_NATIVE_DEC0_INPUT
  ? 'dec0-native-input'
  : CAPTURE_NATIVE_DEC4_INPUT
  ? 'dec4-native-input'
  : CAPTURE_NATIVE_DEC5_INPUT
  ? 'dec5-native-input'
  : CAPTURE_NATIVE_REFERENCE_HEAD ? 'head-contract' : 'full-remap';

const state = {
  settings: readNrSettings(),
  settingsRevision: 0,
  viewRevision: 0,
  viewer: null,
  bridge: null,
  frame: null,
  frameCount: 0,
  snapshot: null,
  model: null,
  inference: null,
  productionPipeline: null,
  captureSlots: [],
  captureGeometry: '',
  nextCaptureSlot: 0,
  pendingProductionFrame: null,
  srModel: null,
  srInference: null,
  modelCapacity: null,
  result: null,
  running: false,
  live: false,
  compareSource: false,
  activeSlot: null,
  captureRequested: false,
  rerunRequested: false,
  neuralReady: false,
  neuralVisible: false,
  resizePending: false,
  resizeSettled: false,
  resizeRevision: 0,
  startupGraphWarmed: false,
  startupWarmupFrames: 0,
  viewportWidth: window.innerWidth,
  viewportHeight: window.innerHeight,
  initialRunScheduled: false,
  statusHistory: [],
};

function setStatus(message, error = false) {
  if (error) globalThis.dlssLoading?.fail(message);
  status.textContent = message;
  status.dataset.state = error ? 'error' : 'active';
  state.statusHistory.push({ time: performance.now(), message, error });
  if (state.statusHistory.length > 160) state.statusHistory.shift();
  if (error) console.error(`[dlss-webgpu] ${message}`);
  else console.info(`[dlss-webgpu:progress] ${message}`);
}

async function preservePresentedFrame() {
  if (!resizeHoldCanvas || !state.neuralVisible || !outputCanvas.width || !outputCanvas.height) return;
  try {
    const bitmap = await createImageBitmap(outputCanvas);
    if (!state.resizePending || state.compareSource) {
      bitmap.close();
      return;
    }
    resizeHoldCanvas.width = bitmap.width;
    resizeHoldCanvas.height = bitmap.height;
    const bitmapContext = resizeHoldCanvas.getContext('bitmaprenderer');
    if (bitmapContext) bitmapContext.transferFromImageBitmap(bitmap);
    else {
      resizeHoldCanvas.getContext('2d', {alpha: false})?.drawImage(bitmap, 0, 0);
      bitmap.close();
    }
    resizeHoldCanvas.classList.add('visible');
  } catch (error) {
    // The original WebGPU canvas remains visible until its replacement is
    // configured, so snapshot failure is a graceful degradation.
    console.warn('[dlss-webgpu] unable to retain resize presentation', error);
  }
}

function releasePresentedFrame() {
  state.resizePending = false;
  state.resizeSettled = false;
  resizeHoldCanvas?.classList.remove('visible');
}

function profileFullDimensions(runtimeProfile) {
  return runtimeProfile.fullDimensions ?? [runtimeProfile.dimensions[0][0] * 2, runtimeProfile.dimensions[0][1] * 2];
}

function destroyGpuRuntime() {
  state.productionPipeline?.destroy();
  state.productionPipeline = null;
  state.srModel?.destroy();
  state.srModel = null;
  state.srInference = null;
  state.model?.destroy();
  state.model = null;
  state.inference = null;
  state.modelCapacity = null;
  state.startupGraphWarmed = false;
}

function describeFrame(frame) {
  return `${frame.renderWidth}×${frame.renderHeight} → ${frame.outputWidth}×${frame.outputHeight}`;
}

function buildHalfFloatTable() {
  const table = new Float32Array(65536);
  for (let bits = 0; bits < table.length; bits += 1) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 31;
    const mantissa = bits & 1023;
    if (exponent === 0) table[bits] = sign * mantissa * 2 ** -24;
    else if (exponent === 31) table[bits] = mantissa ? Number.NaN
      : sign * Number.POSITIVE_INFINITY;
    else table[bits] = sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return table;
}

const outputHalfScratch = new ArrayBuffer(4);
const outputHalfFloat = new Float32Array(outputHalfScratch);
const outputHalfBits = new Uint32Array(outputHalfScratch);

function truncateOutputF16(value) {
  outputHalfFloat[0] = value;
  const bits = outputHalfBits[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  let halfBits;
  if (exponent === 0xff) halfBits = sign | (mantissa ? 0x7e00 : 0x7c00);
  else {
    const halfExponent = exponent - 112;
    if (halfExponent >= 31) halfBits = sign | 0x7c00;
    else if (halfExponent <= 0) halfBits = halfExponent < -10 ? sign
      : sign | ((mantissa | 0x800000) >>> (14 - halfExponent));
    else halfBits = sign | (halfExponent << 10) | (mantissa >>> 13);
  }
  const halfExponent = (halfBits >>> 10) & 31;
  const halfMantissa = halfBits & 1023;
  const halfSign = halfBits & 0x8000 ? -1 : 1;
  if (halfExponent === 0) return halfSign * halfMantissa * 2 ** -24;
  if (halfExponent === 31) return halfMantissa ? Number.NaN
    : halfSign * Number.POSITIVE_INFINITY;
  return halfSign * (1 + halfMantissa / 1024) * 2 ** (halfExponent - 15);
}

function clearGlErrors(gl) {
  for (let count = 0; count < 32 && gl.getError() !== gl.NO_ERROR; count += 1) { /* clear */ }
}

function readTarget(renderer, target, width, height, channels, ArrayType, format, type, label,
  destination = null) {
  const gl = renderer.getContext();
  const previousTarget = renderer.getRenderTarget();
  const previousFace = renderer.getActiveCubeFace?.() ?? 0;
  const previousLevel = renderer.getActiveMipmapLevel?.() ?? 0;
  const previousPackAlignment = gl.getParameter(gl.PACK_ALIGNMENT);
  const expectedValues = width * height * channels;
  const values = destination ?? new ArrayType(expectedValues);
  if (!(values instanceof ArrayType) || values.length !== expectedValues) {
    throw new Error(`${label} destination has ${values.length}/${expectedValues} values`);
  }
  try {
    renderer.setRenderTarget(target);
    if (typeof gl.readBuffer === 'function') gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    clearGlErrors(gl);
    gl.readPixels(0, 0, width, height, format, type, values);
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      throw new Error(`${label} readPixels failed with WebGL error 0x${error.toString(16)}`);
    }
  } finally {
    gl.pixelStorei(gl.PACK_ALIGNMENT, previousPackAlignment);
    renderer.setRenderTarget(previousTarget, previousFace, previousLevel);
  }
  return values;
}

function ensureCaptureSlots(width, height) {
  const geometry = `${width}x${height}`;
  if (state.captureGeometry === geometry) return;
  state.captureGeometry = geometry;
  state.nextCaptureSlot = 0;
  state.captureSlots = Array.from({ length: 2 }, (_, index) => ({
    index,
    color: new Uint16Array(width * height * 4),
    motion: new Uint16Array(width * height * 2),
  }));
}

function captureProductionColor(frame, preferredSlot = null) {
  const { renderer, renderWidth: width, renderHeight: height } = frame;
  if (width !== frame.outputWidth || height !== frame.outputHeight) {
    throw new Error(`DLSS-NR production requires 1:1 input/output; received ${describeFrame(frame)}`);
  }
  ensureCaptureSlots(width, height);
  const slotIndex = preferredSlot ?? state.nextCaptureSlot;
  state.nextCaptureSlot = (slotIndex + 1) % state.captureSlots.length;
  const slot = state.captureSlots[slotIndex];
  const gl = renderer.getContext();
  const started = performance.now();
  readTarget(renderer, frame.color, width, height, 4, Uint16Array,
    gl.RGBA, gl.HALF_FLOAT, 'RGBA16F color', slot.color);
  readTarget(renderer, frame.motion, width, height, 2, Uint16Array,
    gl.RG, gl.HALF_FLOAT, 'RG16F NR motion', slot.motion);
  return {
    frame,
    slotIndex,
    color: slot.color,
    motion: slot.motion,
    sourceFrame: state.frameCount,
    viewRevision: state.viewRevision,
    resizeRevision: state.resizeRevision,
    convergenceFrame: state.viewer.renderer.frameCount,
    renderWidth: width,
    renderHeight: height,
    outputWidth: frame.outputWidth,
    outputHeight: frame.outputHeight,
    reset: frame.reset,
    readbackMilliseconds: performance.now() - started,
    uploaded: false,
    uploadMilliseconds: 0,
  };
}

function captureContract(frame) {
  const { renderer, renderWidth: rw, renderHeight: rh, outputWidth: ow, outputHeight: oh } = frame;
  const gl = renderer.getContext();
  const started = performance.now();
  const snapshot = {
    color: readTarget(renderer, frame.color, rw, rh, 4, Uint16Array,
      gl.RGBA, gl.HALF_FLOAT, 'RGBA16F color'),
    depth: readTarget(renderer, frame.depth, rw, rh, 1, Float32Array,
      gl.RED, gl.FLOAT, 'R32F depth'),
    motion: readTarget(renderer, frame.motion, rw, rh, 2, Uint16Array,
      gl.RG, gl.HALF_FLOAT, 'RG16F motion'),
    reactive: readTarget(renderer, frame.reactive, rw, rh, 1, Uint8Array,
      gl.RED, gl.UNSIGNED_BYTE, 'R8 reactive mask'),
    control: readTarget(renderer, frame.control, ow, oh, 1, Uint8Array,
      gl.RED, gl.UNSIGNED_BYTE, 'R8 control mask'),
    renderWidth: rw,
    renderHeight: rh,
    outputWidth: ow,
    outputHeight: oh,
    jitter: { ...frame.jitter },
    previousJitter: { ...frame.previousJitter },
    resizeRevision: state.resizeRevision,
    reset: frame.reset,
    readbackMilliseconds: performance.now() - started,
  };
  validateSnapshot(snapshot);
  return snapshot;
}

function sampledRange(values, decoder = (value) => value) {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let finite = 0;
  const step = Math.max(1, Math.floor(values.length / 8192));
  for (let index = 0; index < values.length; index += step) {
    const value = decoder(values[index]);
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    finite += 1;
  }
  return { minimum, maximum, finite };
}

function sampledChannelRanges(values, channels, decoder = (value) => value) {
  return Array.from({ length: channels }, (_, channel) => {
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let finite = 0;
    const pixels = Math.floor(values.length / channels);
    const step = Math.max(1, Math.floor(pixels / 4096));
    for (let pixel = 0; pixel < pixels; pixel += step) {
      const value = decoder(values[pixel * channels + channel]);
      if (!Number.isFinite(value)) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      finite += 1;
    }
    return { minimum, maximum, finite };
  });
}

function publishDiagnostics(values) {
  state.diagnostics = { ...(state.diagnostics ?? {}), ...values };
  status.dataset.diagnostics = JSON.stringify(state.diagnostics);
  console.info('[dlss-webgpu] numeric diagnostics', state.diagnostics);
}

function validateSnapshot(snapshot) {
  const color = sampledRange(snapshot.color, (value) => HALF_TO_FLOAT[value]);
  const depth = sampledRange(snapshot.depth);
  const motion = sampledRange(snapshot.motion, (value) => HALF_TO_FLOAT[value]);
  const reactive = sampledRange(snapshot.reactive);
  const control = sampledRange(snapshot.control);
  if (!color.finite || color.maximum - color.minimum < 1e-6) {
    throw new Error('The exact pre-tonemap color target is blank');
  }
  if (!depth.finite || !motion.finite || !reactive.finite || !control.finite) {
    throw new Error('One or more exact WebGI auxiliary targets could not be read');
  }
  snapshot.ranges = { color, depth, motion, reactive, control };
}

function decodeWebGiSceneLinear(snapshot) {
  const { renderWidth: width, renderHeight: height, outputWidth, outputHeight } = snapshot;
  if (width !== outputWidth || height !== outputHeight) {
    throw new Error(`DLSS-NR-only mode requires a 1:1 WebGI render; received ${width}×${height} → ${outputWidth}×${outputHeight}`);
  }
  const data = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const sourceY = height - y - 1;
    for (let x = 0; x < width; x += 1) {
      const source = (sourceY * width + x) * 4;
      const target = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const value = HALF_TO_FLOAT[snapshot.color[source + channel]];
        data[target + channel] = Number.isFinite(value) ? Math.max(value, 0) : 0;
      }
      data[target + 3] = 1;
    }
  }
  return { data, width, height, implementation: 'WebGI RGBA16F pre-tonemap' };
}

function srgbEncode(value) {
  const bounded = Math.max(0, Math.min(1, value));
  return bounded <= 0.0031308 ? 12.92 * bounded
    : 1.055 * bounded ** (1 / 2.4) - 0.055;
}

function srgbDecode(value) {
  const bounded = Math.max(0, Math.min(1, value));
  return bounded <= 0.04045 ? bounded / 12.92
    : ((bounded + 0.055) / 1.055) ** 2.4;
}

function makeDisplayProxy(sceneLinear) {
  const proxy = new Float32Array(sceneLinear.data.length);
  const paper = Math.max(PAPER_WHITE_SCALE, 0.05);
  for (let index = 0; index < proxy.length; index += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      let value = Math.max(sceneLinear.data[index + channel], 0) / paper;
      if (value > 0.75) value = 0.75 + 0.25 * (1 - Math.exp(-5.770780 * (value - 0.75)));
      proxy[index + channel] = roundNativeF16(srgbEncode(value));
    }
    proxy[index + 3] = 1;
  }
  return { data: proxy, width: sceneLinear.width, height: sceneLinear.height };
}

function mat3Mul(matrix, value) {
  return [
    matrix[0] * value[0] + matrix[1] * value[1] + matrix[2] * value[2],
    matrix[3] * value[0] + matrix[4] * value[1] + matrix[5] * value[2],
    matrix[6] * value[0] + matrix[7] * value[1] + matrix[8] * value[2],
  ];
}

function luminance(color) {
  return color[0] * 0.212639 + color[1] * 0.715169 + color[2] * 0.072192;
}

function toOkLab(color) {
  const lms = mat3Mul([
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005,
  ], color).map((value) => Math.sign(value) * Math.abs(value) ** (1 / 3));
  return mat3Mul([
    0.2104542553, 0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, 0.4505937099,
    0.0259040371, 0.7827717662, -0.8086757660,
  ], lms);
}

function fromOkLab(lab) {
  const lms = mat3Mul([
    1, 0.3963377774, 0.2158037573,
    1, -0.1055613458, -0.0638541728,
    1, -0.0894841775, -1.2914855480,
  ], lab).map((value) => value * value * value);
  return mat3Mul([
    4.0767416621, -3.3077115913, 0.2309699292,
    -1.2684380046, 2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147, 1.7076147010,
  ], lms);
}

function clampAp1(color) {
  const ap1 = mat3Mul([
    0.613097, 0.339523, 0.047379,
    0.070194, 0.916354, 0.013452,
    0.020616, 0.109570, 0.869815,
  ], color).map((value) => Math.max(0, value));
  return mat3Mul([
    1.705051, -0.621792, -0.083259,
    -0.130256, 1.140805, -0.010548,
    -0.024003, -0.128969, 1.152972,
  ], ap1);
}

function hueOkLab(incorrect, correct) {
  const incorrectLab = toOkLab(incorrect);
  const correctLab = toOkLab(correct);
  const incorrectChroma = Math.hypot(incorrectLab[1], incorrectLab[2]);
  const correctChroma = Math.hypot(correctLab[1], correctLab[2]);
  const scale = correctChroma === 0 ? 1 : incorrectChroma / correctChroma;
  incorrectLab[1] = correctLab[1] * scale;
  incorrectLab[2] = correctLab[2] * scale;
  return clampAp1(fromOkLab(incorrectLab));
}

function upgradeToneMap(original, proxy, neural) {
  const originalY = luminance(original);
  const proxyY = luminance(proxy);
  const neuralY = luminance(neural);
  if (neuralY <= 1e-5) return original;
  const ratio = originalY < proxyY
    ? originalY / Math.max(proxyY, 1e-6)
    : (neuralY + Math.max(0, originalY - proxyY)) / neuralY;
  const scaled = hueOkLab(neural.map((value) => value * ratio), neural);
  return original.map((value, channel) => value
    + (scaled[channel] - value) * TRANSFER_STRENGTH);
}

function acesFit(value) {
  return (value * (value + 0.0245786) - 0.000090537)
    / (value * (0.983729 * value + 0.4329510) + 0.238081);
}

function webGiDisplay(sceneLinear) {
  let color = mat3Mul([
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777,
  ], sceneLinear.map((value) => value / 0.6)).map(acesFit);
  color = mat3Mul([
    1.60475, -0.53108, -0.07367,
    -0.10208, 1.10813, -0.00605,
    -0.00327, -0.07276, 1.07602,
  ], color).map((value) => Math.max(0, Math.min(1, value)));
  return color.map(srgbEncode);
}

function composeOutput(sceneLinear, proxy, neuralResult, controlMask) {
  const { width, height } = sceneLinear;
  const bytes = new Uint8ClampedArray(width * height * 4);
  const { features, featureChannels, width: featureWidth } = neuralResult;
  const paper = Math.max(PAPER_WHITE_SCALE, 0.05);
  for (let y = 0; y < height; y += 1) {
    const controlY = height - y - 1;
    for (let x = 0; x < width; x += 1) {
      const pixel = y * width + x;
      const rgba = pixel * 4;
      const feature = (y * featureWidth + x) * featureChannels;
      const preserve = USE_EXTERNAL_CONTROL_MASK
        ? controlMask[(controlY * width) + x] / 255
        : 0;
      const original = [
        Math.max(sceneLinear.data[rgba], 0) / paper,
        Math.max(sceneLinear.data[rgba + 1], 0) / paper,
        Math.max(sceneLinear.data[rgba + 2], 0) / paper,
      ];
      const proxyLinear = [0, 1, 2].map((channel) => srgbDecode(proxy.data[rgba + channel]));
      const neuralLinear = [0, 1, 2].map((channel) => {
        const source = roundNativeF16(proxy.data[rgba + channel]);
        const centered = Math.fround(Math.fround(source * 0.125) - 0.0625);
        const head = features[feature + channel];
        const residual = Number.isFinite(head) ? Math.fround(head * 0.03125) : 0;
        const mixed = Math.fround(centered + residual);
        const combined = Math.max(0, Math.min(1,
          Math.fround(Math.fround(mixed * 8) + 0.5)));
        const published = truncateOutputF16(combined);
        const controlled = published + (proxy.data[rgba + channel] - published) * preserve;
        return srgbDecode(controlled);
      });
      const upgraded = upgradeToneMap(original, proxyLinear, neuralLinear);
      const originalY = luminance(original);
      const upgradedY = luminance(upgraded);
      const ratio = originalY === 0 ? 1 : Math.max(0, Math.min(4, upgradedY / originalY));
      const result = upgraded.map((value, channel) => {
        const luminanceOnly = original[channel] * ratio;
        return (luminanceOnly + (value - luminanceOnly) * COLOR_STRENGTH) * paper;
      });
      const display = webGiDisplay(result);
      bytes[rgba] = Math.floor(display[0] * 255 + 0.5);
      bytes[rgba + 1] = Math.floor(display[1] * 255 + 0.5);
      bytes[rgba + 2] = Math.floor(display[2] * 255 + 0.5);
      bytes[rgba + 3] = 255;
    }
  }
  return new ImageData(bytes, width, height);
}

function modelProgress(message, error = false, download = null) {
  setStatus(`DLSS-NR WebGPU · ${message}`, error);
  // A resize keeps the last completed DLSS image onscreen. Progress remains in
  // the status log without bringing the full-page startup overlay back.
  if (!state.resizePending || !state.result) {
    globalThis.dlssLoading?.runtime(message, error, download);
  }
}

async function ensureInference(runtimeProfile) {
  const [fullWidth, fullHeight] = profileFullDimensions(runtimeProfile);
  if (state.model && (fullWidth > state.modelCapacity.width || fullHeight > state.modelCapacity.height)) {
    // Growth beyond the device limit is uncommon. The CPU weight cache makes
    // this a GPU-only rebuild rather than another 140 MiB transfer.
    destroyGpuRuntime();
  }
  if (!state.model) {
    state.model = await DlssNrWebGpuModel.create({
      production: PRODUCTION_RUNTIME,
      activationWidth: fullWidth,
      activationHeight: fullHeight,
      activationCapacityScale: DEVICE_CAPACITY_HEADROOM,
      onProgress: modelProgress,
    });
    state.modelCapacity = state.model.activationCapacity;
  }
  const [vitWidth, vitHeight] = runtimeProfile.dimensions.at(-1);
  const vitTokens = vitWidth * vitHeight;
  state.inference ??= await DlssNrBrowserInference.create(state.model,
    modelProgress,
    { maxVitTokens: vitTokens, production: PRODUCTION_RUNTIME });
  // Requesting resize headroom changes the physical device limit. Keep the
  // graph's logical FFN tiling limit identical to a fresh exact-size device so
  // memory headroom cannot select a different numerical execution path.
  state.inference.hierarchy.storageBindingLimit =
    state.model.storageBindingSizeFor(fullWidth, fullHeight);
  if (PRODUCTION_RUNTIME) await state.inference.ensureProductionTokens(vitTokens);
  return state.inference;
}

async function ensureProductionPipeline(runtimeProfile) {
  const inference = await ensureInference(runtimeProfile);
  state.productionPipeline ??= await DlssNrProductionPipeline.create(
    inference, outputCanvas, {
      paperWhite: PAPER_WHITE_SCALE,
      colorStrength: COLOR_STRENGTH,
    },
  );
  await state.productionPipeline.ensureGeometry(runtimeProfile);
  return state.productionPipeline;
}

async function ensureSrInference() {
  if (!state.model) throw new Error('DLSS-NR WebGPU device must be initialized before DLSS-SR');
  if (state.srModel && state.srModel.device !== state.model.device) {
    state.srModel.destroy();
    state.srModel = null;
    state.srInference = null;
  }
  state.srModel ??= await DlssSrWebGpuModel.create(state.model.device, {
    onProgress: (message, error = false) => setStatus(message, error),
  });
  state.srInference ??= await DlssSrBrowserInference.create(state.srModel,
    (message, error = false) => setStatus(message, error));
  return state.srInference;
}

async function captureNativeReference(srInference) {
  const enc0Tap = CAPTURE_NATIVE_ENC0_DOWN;
  const inputUrl = enc0Tap
    ? '/native-sr-enc0-tap/native-sr-input16.bin'
    : '/native-sr-contract/native-sr-input16.bin';
  const laterBoundaries = [
    'enc2-mixed', 'enc2-attn', 'enc2-mlp-hidden', 'enc2-skip', 'enc2-down',
    'enc3-mixed', 'enc3-attn', 'enc3-mlp-hidden', 'enc3-skip', 'enc3-down',
    'enc4-mixed', 'enc4-attn', 'enc4-mlp-hidden', 'enc4-skip', 'enc4-down',
    'dec5-mixed', 'dec5-attn', 'dec5-mlp-hidden', 'dec5-output',
    'dec4-output', 'dec3-output', 'dec2-output', 'dec1-output',
    'dec0-output',
  ];
  const boundaries = enc0Tap
    ? ['embedding', 'enc0-values', 'enc0-mixed', 'enc0-attn', 'enc0-mlp-hidden', 'enc0-skip', 'enc0-down',
      'enc1-mixed', 'enc1-attn', 'enc1-mlp-hidden', 'enc1-skip', 'enc1-down',
      ...laterBoundaries]
    : ['enc1-mixed', 'enc1-attn', 'enc1-mlp-hidden', 'enc1-skip', 'enc1-down',
      ...laterBoundaries];
  setStatus(`DLSS-SR diagnostic · loading immutable ${enc0Tap ? 'enc0 tap' : 'head'} input16`);
  const response = await fetch(inputUrl, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Native SR input16 load failed: HTTP ${response.status}`);
  const packed = new Uint16Array(await response.arrayBuffer());
  if (packed.length !== 256 * 256 * 16) {
    throw new Error(`Native SR input16 has ${packed.length} halves, expected ${256 * 256 * 16}`);
  }
  const features = new Float32Array(packed.length);
  for (let index = 0; index < packed.length; index += 1) features[index] = HALF_TO_FLOAT[packed[index]];
  setStatus('DLSS-SR diagnostic · evaluating fixed 512 reference through WebGPU');
  const result = await srInference.run({
    features,
    sceneColor: null,
    renderWidth: 512,
    renderHeight: 512,
    networkWidth: 256,
    networkHeight: 256,
    outputWidth: 1024,
    outputHeight: 1024,
    exposure: 1,
    captureHead: !enc0Tap,
    headOnly: true,
    captureBoundaries: boundaries,
  });
  const uploadCapture = async (name, values) => fetch(
    `/api/sr-capture?name=${name}&set=${SR_CAPTURE_SET}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: values,
  });
  let savedBytes = 0;
  if (!enc0Tap) {
    if (!result.head || result.head.length !== 256 * 256 * 40) {
      throw new Error(`WebGPU SR head capture has ${result.head?.length ?? 0} values`);
    }
    setStatus('DLSS-SR diagnostic · saving 40-lane browser head');
    const upload = await uploadCapture('head40', result.head);
    if (!upload.ok) throw new Error(`SR head capture upload failed: ${await upload.text()}`);
    savedBytes = (await upload.json()).bytes;
  }
  for (const [name, values] of Object.entries(result.boundaries)) {
    const boundaryUpload = await uploadCapture(name, values);
    if (!boundaryUpload.ok) {
      throw new Error(`SR ${name} capture upload failed: ${await boundaryUpload.text()}`);
    }
  }
  publishDiagnostics({ stage: enc0Tap ? 'sr-reference-enc0-down' : 'sr-reference-head40',
    milliseconds: result.milliseconds,
    values: enc0Tap ? result.boundaries['enc0-down']?.length : result.head.length,
    bytes: enc0Tap ? result.boundaries['enc0-down']?.byteLength : savedBytes });
  setStatus(`DLSS-SR diagnostic ${enc0Tap ? 'enc0-down' : 'head'} saved · ${result.milliseconds.toFixed(0)} ms`);
}

async function captureNativeDecoder5Input(srInference) {
  setStatus('DLSS-SR diagnostic · loading exact native encoder-4-down input');
  const response = await fetch('/native-sr-full-remap/native-sr-enc4-down.bin', {
    cache: 'no-store',
  });
  if (!response.ok) throw new Error(`Native encoder-4-down load failed: HTTP ${response.status}`);
  const packed = new Uint16Array(await response.arrayBuffer());
  const expected = 8 * 8 * 160;
  if (packed.length !== expected) {
    throw new Error(`Native encoder-4-down has ${packed.length} halves, expected ${expected}`);
  }
  const input = new Float32Array(expected);
  for (let index = 0; index < expected; index += 1) input[index] = HALF_TO_FLOAT[packed[index]];
  setStatus('DLSS-SR diagnostic · replaying decoder-5 from exact native input');
  const result = await srInference.runBlockDiagnostic({
    stageId: 'dec_5', input, width: 8, height: 8, precision: DEC5_PRECISION,
  });
  const uploads = [
    [`dec5-native-input-${DEC5_PRECISION}-qkv`, result.qkv],
    [`dec5-native-input-${DEC5_PRECISION}-mixed`, result.mixed],
    [`dec5-native-input-${DEC5_PRECISION}-attn`, result.attention],
    [`dec5-native-input-${DEC5_PRECISION}-output`, result.output],
  ];
  for (const [name, values] of uploads) {
    const upload = await fetch(`/api/sr-capture?name=${name}&set=${SR_CAPTURE_SET}`, {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: values,
    });
    if (!upload.ok) throw new Error(`SR ${name} capture upload failed: ${await upload.text()}`);
  }
  publishDiagnostics({
    stage: `sr-reference-dec5-native-input-${DEC5_PRECISION}`, milliseconds: result.milliseconds,
    values: result.output.length, bytes: result.output.byteLength,
  });
  setStatus(`DLSS-SR decoder-5 ${DEC5_PRECISION} native-input replay saved · ${result.milliseconds.toFixed(0)} ms`);
}

async function captureNativeDecoder4Input(srInference) {
  setStatus('DLSS-SR diagnostic · loading exact native decoder-4 block input');
  const response = await fetch('/native-sr-dec4-up-tap/native-sr-dec4-output.bin', {
    cache: 'no-store',
  });
  if (!response.ok) throw new Error(`Native decoder-4 input load failed: HTTP ${response.status}`);
  const packed = new Uint16Array(await response.arrayBuffer());
  const expected = 16 * 16 * 128;
  if (packed.length !== expected) {
    throw new Error(`Native decoder-4 input has ${packed.length} halves, expected ${expected}`);
  }
  const input = new Float32Array(expected);
  for (let index = 0; index < expected; index += 1) input[index] = HALF_TO_FLOAT[packed[index]];
  setStatus('DLSS-SR diagnostic · replaying decoder-4 from exact native input');
  const result = await srInference.runBlockDiagnostic({
    stageId: 'dec_4', input, width: 16, height: 16, precision: DEC4_PRECISION,
  });
  const uploads = [
    [`dec4-native-input-${DEC4_PRECISION}-qkv`, result.qkv],
    [`dec4-native-input-${DEC4_PRECISION}-mixed`, result.mixed],
    [`dec4-native-input-${DEC4_PRECISION}-attn`, result.attention],
    [`dec4-native-input-${DEC4_PRECISION}-mlp-hidden`, result.mlpHidden],
    [`dec4-native-input-${DEC4_PRECISION}-output`, result.output],
  ];
  for (const [name, values] of uploads) {
    const upload = await fetch(`/api/sr-capture?name=${name}&set=${SR_CAPTURE_SET}`, {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: values,
    });
    if (!upload.ok) throw new Error(`SR ${name} capture upload failed: ${await upload.text()}`);
  }
  publishDiagnostics({
    stage: `sr-reference-dec4-native-input-${DEC4_PRECISION}`, milliseconds: result.milliseconds,
    values: result.output.length, bytes: result.output.byteLength,
  });
  setStatus(`DLSS-SR decoder-4 ${DEC4_PRECISION} native-input replay saved · ${result.milliseconds.toFixed(0)} ms`);
}

async function captureNativeDecoder0Input(srInference) {
  setStatus('DLSS-SR diagnostic · loading exact native decoder-1 and encoder-0 inputs');
  const [decoderResponse, skipResponse] = await Promise.all([
    fetch('/native-sr-full-remap/native-sr-dec1-output.bin', { cache: 'no-store' }),
    fetch('/native-sr-full-remap/native-sr-enc0-skip.bin', { cache: 'no-store' }),
  ]);
  if (!decoderResponse.ok) {
    throw new Error(`Native decoder-1 output load failed: HTTP ${decoderResponse.status}`);
  }
  if (!skipResponse.ok) {
    throw new Error(`Native encoder-0 skip load failed: HTTP ${skipResponse.status}`);
  }
  const decoderPacked = new Uint16Array(await decoderResponse.arrayBuffer());
  const skipPacked = new Uint16Array(await skipResponse.arrayBuffer());
  const expectedDecoder = 128 * 128 * 64;
  const expectedSkip = 256 * 256 * 32;
  if (decoderPacked.length !== expectedDecoder || skipPacked.length !== expectedSkip) {
    throw new Error(`Native decoder-0 inputs have ${decoderPacked.length}/${skipPacked.length} halves, expected ${expectedDecoder}/${expectedSkip}`);
  }
  const input = new Float32Array(expectedDecoder);
  const skip = new Float32Array(expectedSkip);
  for (let index = 0; index < input.length; index += 1) input[index] = HALF_TO_FLOAT[decoderPacked[index]];
  for (let index = 0; index < skip.length; index += 1) skip[index] = HALF_TO_FLOAT[skipPacked[index]];
  const variants = DEC0_PRECISION_SWEEP
    ? ['f32', 'norm-f16', 'qkv-f16', 'projection-f16', 'mlp-f16', 'front-f16',
      'block-f16', 'head-f16', 'all-f16']
    : [DEC0_PRECISION];
  let totalMilliseconds = 0;
  let lastResult = null;
  for (const precision of variants) {
    setStatus(`DLSS-SR diagnostic · decoder-0 exact-input ${precision}`);
    const captureIntermediate = !DEC0_PRECISION_SWEEP;
    const result = await srInference.runDecoderDiagnostic({
      stageId: 'dec_0', input, skip, inputWidth: 128, inputHeight: 128,
      outputWidth: 256, outputHeight: 256, precision, captureIntermediate,
    });
    totalMilliseconds += result.milliseconds;
    lastResult = result;
    const uploads = DEC0_PRECISION_SWEEP
      ? [[`dec0-native-input-${precision}-head40`, result.head]]
      : [
        ['dec0-native-input-upsample', result.upsample],
        ['dec0-native-input-padded', result.padded],
        ['dec0-native-input-normalized', result.normalized],
        ['dec0-native-input-values', result.values],
        ['dec0-native-input-mixed', result.mixed],
        ['dec0-native-input-attention', result.attention],
        ['dec0-native-input-normalized2', result.normalized2],
        ['dec0-native-input-mlp-hidden', result.mlpHidden],
        ['dec0-native-input-output', result.output],
        ['dec0-native-input-head40', result.head],
      ];
    for (const [name, values] of uploads) {
      const upload = await fetch(`/api/sr-capture?name=${name}&set=${SR_CAPTURE_SET}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: values,
      });
      if (!upload.ok) throw new Error(`SR ${name} capture upload failed: ${await upload.text()}`);
    }
  }
  publishDiagnostics({
    stage: DEC0_PRECISION_SWEEP ? 'sr-reference-dec0-precision-sweep'
      : 'sr-reference-dec0-native-input', milliseconds: totalMilliseconds,
    variants, values: lastResult.head.length, bytes: lastResult.head.byteLength,
  });
  setStatus(`DLSS-SR decoder-0 ${DEC0_PRECISION_SWEEP ? 'precision sweep' : 'native-input replay'} saved · ${totalMilliseconds.toFixed(0)} ms`);
}

function setNeuralVisible(visible) {
  state.neuralVisible = state.settings.enabled && state.neuralReady && visible;
  outputCanvas.classList.toggle('visible', state.neuralVisible);
  demoUi.setPresented(state.neuralVisible);
  if (state.viewer) state.viewer.renderEnabled = state.live || !state.neuralVisible;
  document.querySelector("#nrCompare")?.setAttribute("aria-pressed", String(state.neuralVisible));
  if (!state.neuralReady) return;
  const performanceSummary = state.result?.timings
    ? ` · read ${state.result.timings.readbackMilliseconds.toFixed(0)} · prep ${formatMilliseconds(state.result.timings.uploadPreprocessMilliseconds)} · net ${formatMilliseconds(state.result.timings.networkMilliseconds)} · present ${formatMilliseconds(state.result.timings.presentationMilliseconds)} ms`
    : '';
  setStatus(state.neuralVisible
    ? `DLSS-NR WebGPU · ${state.result.outputWidth}×${state.result.outputHeight}${performanceSummary} · F6 source`
    : `WebGI source · ${describeFrame(state.frame)} · F6 neural`);
  status.dataset.state = 'ready';
}

async function runProductionFrame(frame, queuedCapture = null) {
  const settings = {...state.settings};
  const revision = state.settingsRevision;
  setStatus(`Reading WebGI RGBA16F color · ${describeFrame(frame)}`);
  const snapshot = queuedCapture ?? captureProductionColor(frame);
  state.snapshot = snapshot;
  state.activeSlot = snapshot.slotIndex;
  if (state.viewer) state.viewer.renderEnabled = state.live;

  const runtimeProfile = createRuntimeProfile(snapshot.renderWidth, snapshot.renderHeight);
  setStatus(`Color captured in ${snapshot.readbackMilliseconds.toFixed(0)} ms · preparing persistent WebGPU graph`);
  const pipeline = await ensureProductionPipeline(runtimeProfile);
  if (!snapshot.uploaded) {
    snapshot.uploadMilliseconds = pipeline.upload(snapshot.slotIndex, snapshot.color, snapshot.motion);
    snapshot.uploaded = true;
  }

  const reset = snapshot.reset || snapshot.sourceFrame !== state.lastNrSourceFrame + 1;
  globalThis.dlssLoading?.stage('Rendering first DLSS frame', 'Neural model ready · preparing your view');
  const {neural, timings: gpuTimings} = await pipeline.render(snapshot.slotIndex, runtimeProfile, settings, {reset});
  state.lastNrSourceFrame = snapshot.sourceFrame;
  const timings = {
    readbackMilliseconds: snapshot.readbackMilliseconds,
    uploadPreprocessMilliseconds: gpuTimings
      ? snapshot.uploadMilliseconds + gpuTimings.preprocessMilliseconds : null,
    networkMilliseconds: gpuTimings?.networkMilliseconds ?? null,
    presentationMilliseconds: gpuTimings?.presentationMilliseconds ?? null,
  };
  if (!state.startupGraphWarmed) {
    // The first production plan is discovered while the browser is compiling
    // the complete cold shader set. Never present that plan. A real resize used
    // to fix the intermittent Default/Cinematic aliasing because it discarded
    // these exact resolution-dependent resources after compilation was warm.
    // Perform that transition once at startup while retaining the GPU device,
    // uploaded model weights, hierarchy pipelines and ViT pipeline.
    const retainedDevice = state.model.device;
    const retainedModel = state.model;
    const retainedInference = state.inference;
    state.startupGraphWarmed = true;
    state.startupWarmupFrames++;
    state.productionPipeline.destroy();
    state.productionPipeline = null;
    state.inference.resetProductionGraph();
    state.lastNrSourceFrame = null;
    state.snapshot = null;
    state.neuralReady = false;
    state.result = null;
    state.rerunRequested = true;
    if (state.viewer) state.viewer.renderEnabled = true;
    setStatus('Cold WebGPU graph warmed · rebuilding stable presentation graph');
    globalThis.dlssLoading?.stage(
      'Preparing first frame',
      'WebGPU kernels ready · validating the presentation graph',
    );
    console.info('[dlss-webgpu] discarded cold startup plan', {
      dimensions: [snapshot.outputWidth, snapshot.outputHeight],
      retainedDevice: state.model.device === retainedDevice,
      retainedModel: state.model === retainedModel,
      retainedInference: state.inference === retainedInference,
      warmupFrames: state.startupWarmupFrames,
    });
    return;
  }
  state.result = {
    snapshot, sr: null, neural,
    outputWidth: snapshot.outputWidth, outputHeight: snapshot.outputHeight,
    runtimeProfile, pipeline: 'dlss-nr-production-gpu', maskMode: settings.autoMask ? 'automatic' : 'unmasked',
    settings, settingsRevision: revision,
    timings, timingSource: gpuTimings?.source ?? 'GPU timestamps unavailable',
  };
  // Live presents completed frames while the camera continues to move. A
  // newer camera revision must not starve presentation of every in-flight frame.
  const currentResize = !state.resizePending || snapshot.resizeRevision === state.resizeRevision;
  if (currentResize) {
    state.neuralReady = revision === state.settingsRevision &&
      (state.live || snapshot.viewRevision === state.viewRevision);
  }
  if (currentResize && state.neuralReady) {
    demoUi.frameComplete(timings, snapshot.outputWidth, snapshot.outputHeight);
  }
  globalThis.__dlssWebGpuExact = state;
  if (currentResize) {
    setNeuralVisible(!state.compareSource);
    if (state.resizePending && state.neuralReady) releasePresentedFrame();
  } else {
    state.rerunRequested = true;
  }
  console.info('[dlss-webgpu] production frame complete', {
    dimensions: [snapshot.outputWidth, snapshot.outputHeight],
    timings,
    resources: neural.resources,
    commandSubmissions: 1,
    completionWaits: 1,
    inputResources: ['RGBA16F color'],
    maskMode: state.result.maskMode,
    settings,
  });
}

async function runCapturedFrame(frame, queuedCapture = null) {
  if (!state.settings.enabled) return;
  if (state.running) {
    state.rerunRequested = true;
    return;
  }
  state.running = true;
  state.captureRequested = false;
  state.rerunRequested = false;
  if (!state.live && !state.resizePending) setNeuralVisible(false);
  try {
    if (PRODUCTION_RUNTIME) {
      await runProductionFrame(frame, queuedCapture);
      return;
    }
    setStatus(`Reading exact WebGI bridge · ${describeFrame(frame)} · five resources`);
    const snapshot = captureContract(frame);
    state.snapshot = snapshot;
    console.info('[dlss-webgpu] captured bridge ranges', snapshot.ranges);
    // Freeze the producer at the exact frame copied above while the offline
    // WebGPU graph evaluates it. Pointer/F6/R re-enable the unchanged viewer.
    if (state.viewer) state.viewer.renderEnabled = false;
    const srDiagnostic = CAPTURE_NATIVE_DEC0_INPUT || CAPTURE_NATIVE_DEC4_INPUT
      || CAPTURE_NATIVE_DEC5_INPUT || CAPTURE_NATIVE_REFERENCE_HEAD
      || CAPTURE_NATIVE_ENC0_DOWN;
    setStatus(`Bridge captured in ${snapshot.readbackMilliseconds.toFixed(0)} ms · preparing ${srDiagnostic ? 'SR diagnostic' : 'native-resolution NR input'}`);
    const runtimeProfile = createRuntimeProfile(
      srDiagnostic ? snapshot.outputWidth : snapshot.renderWidth,
      srDiagnostic ? snapshot.outputHeight : snapshot.renderHeight,
    );
    const inference = await ensureInference(runtimeProfile);
    if (srDiagnostic) {
      const srInference = await ensureSrInference();
      if (CAPTURE_NATIVE_DEC0_INPUT) await captureNativeDecoder0Input(srInference);
      else if (CAPTURE_NATIVE_DEC4_INPUT) await captureNativeDecoder4Input(srInference);
      else if (CAPTURE_NATIVE_DEC5_INPUT) await captureNativeDecoder5Input(srInference);
      else await captureNativeReference(srInference);
      if (state.viewer) state.viewer.renderEnabled = true;
      return;
    }
    const sceneLinear = decodeWebGiSceneLinear(snapshot);
    const bridgeColorChannels = sampledChannelRanges(snapshot.color, 4,
      (value) => HALF_TO_FLOAT[value]);
    const sceneRange = sampledRange(sceneLinear.data);
    const sceneInputChannels = sampledChannelRanges(sceneLinear.data, 4);
    publishDiagnostics({
      stage: 'nr-input-source',
      bridgeColorChannels,
      sceneRange,
      sceneInputChannels,
    });
    if (!sceneRange.finite || !Number.isFinite(sceneRange.minimum)
        || !Number.isFinite(sceneRange.maximum)) {
      throw new Error('WebGI produced no finite pre-tonemap color samples');
    }
    const proxy = makeDisplayProxy(sceneLinear);
    const proxyRange = sampledRange(proxy.data);
    publishDiagnostics({ stage: 'nr-input', sceneRange, proxyRange });
    if (CAPTURE_NR_REFERENCE) {
      const upload = await fetch(`/api/nr-capture?name=proxy&width=${proxy.width}&height=${proxy.height}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: proxy.data,
      });
      if (!upload.ok) throw new Error(`NR proxy capture failed: ${await upload.text()}`);
    }
    setStatus(`DLSS-NR only · automatic mask · ${sceneLinear.width}×${sceneLinear.height}`);
    const neural = await inference.run(proxy, {
      profile: 'runtime', runtimeProfile, autoMask: !USE_EXTERNAL_CONTROL_MASK,
    });
    const neuralRange = sampledRange(neural.features);
    if (CAPTURE_NR_REFERENCE) {
      const upload = await fetch(`/api/nr-capture?name=head&width=${neural.width}&height=${neural.height}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
        body: neural.features,
      });
      if (!upload.ok) throw new Error(`NR head capture failed: ${await upload.text()}`);
    }
    publishDiagnostics({
      stage: 'nr-output', sceneRange, proxyRange, neuralRange,
    });
    setStatus('DLSS-NR complete · automatic mask · applying WebGI display transform');
    const composed = composeOutput(sceneLinear, proxy, neural, snapshot.control);
    outputCanvas.width = sceneLinear.width;
    outputCanvas.height = sceneLinear.height;
    outputContext.putImageData(composed, 0, 0);
    const composedRange = sampledRange(composed.data);
    publishDiagnostics({
      stage: 'composed', sceneRange, proxyRange, neuralRange, composedRange,
    });
    state.result = {
      snapshot,
      sceneLinear,
      proxy,
      sr: null,
      neural,
      composed,
      outputWidth: sceneLinear.width,
      outputHeight: sceneLinear.height,
      runtimeProfile,
      pipeline: 'dlss-nr-only',
      maskMode: USE_EXTERNAL_CONTROL_MASK ? 'external-control' : 'automatic',
      unsupported: ['DLSS-SR intentionally disconnected pending parity closure',
        'feature-18 depth/motion temporal preprocessing'],
    };
    state.neuralReady = true;
    globalThis.__dlssWebGpuExact = state;
    setNeuralVisible(true);
    if (state.resizePending && snapshot.resizeRevision === state.resizeRevision) releasePresentedFrame();
    console.info('[dlss-webgpu] feature-18 frame complete', {
      dimensions: [sceneLinear.width, sceneLinear.height],
      graphMilliseconds: neural.milliseconds,
      pipeline: state.result.pipeline,
      readbackMilliseconds: snapshot.readbackMilliseconds,
      ranges: snapshot.ranges,
      runtimeProfile,
      maskMode: state.result.maskMode,
      contractGap: state.result.unsupported,
    });
  } catch (error) {
    console.error(error);
    setStatus(error.stack || error.message || String(error), true);
    state.live = false;
    state.pendingProductionFrame = null;
    state.rerunRequested = false;
    state.captureRequested = false;
    document.querySelector('#nrLive')?.setAttribute('aria-pressed', 'false');
    if (state.viewer) state.viewer.renderEnabled = true;
  } finally {
    state.running = false;
    state.activeSlot = null;
    const pending = state.pendingProductionFrame;
    state.pendingProductionFrame = null;
    if (pending) void runCapturedFrame(pending.frame, pending);
    else if (state.rerunRequested) requestNeuralFrame();
  }
}

function consumeFrame(frame) {
  if (!globalThis.dlssSceneReady) return;
  state.frame = frame;
  state.frameCount += 1;
  if (state.frameCount === 1) {
    setStatus(PRODUCTION_RUNTIME
      ? `WebGPU production connected · ${describeFrame(frame)} · color and NR motion`
      : `WebGPU diagnostic connected · ${describeFrame(frame)} · color/depth/MV/reactive/control`);
    console.info('[dlss-webgpu] bridge contract', {
      render: [frame.renderWidth, frame.renderHeight],
      output: [frame.outputWidth, frame.outputHeight],
      color: frame.color.texture.name,
      depth: frame.depth?.texture.name,
      motion: frame.motion?.texture.name,
      reactive: frame.reactive?.texture.name,
      control: frame.control?.texture.name,
      jitter: frame.jitter,
      previousJitter: frame.previousJitter,
      reset: frame.reset,
    });
  }
  if (state.live && state.initialRunScheduled && !state.pendingProductionFrame) state.captureRequested = true;
  // Only still captures wait for accumulation. Live consumes successive frames
  // during motion, with the existing two-slot backpressure and temporal history.
  if (state.captureRequested && !state.live && !viewerConverged(state.viewer)) return;
  if (state.captureRequested && state.running && PRODUCTION_RUNTIME) {
    try {
      const preferredSlot = state.pendingProductionFrame?.slotIndex ?? (state.activeSlot === 0 ? 1 : 0);
      const pending = captureProductionColor(frame, preferredSlot);
      if (state.productionPipeline
          && state.productionPipeline.geometry.validWidth === pending.renderWidth
          && state.productionPipeline.geometry.validHeight === pending.renderHeight) {
        pending.uploadMilliseconds = state.productionPipeline.upload(
          pending.slotIndex, pending.color, pending.motion);
        pending.uploaded = true;
      }
      state.pendingProductionFrame = pending;
      state.captureRequested = false;
      state.rerunRequested = false;
      // NR motion describes the preceding rendered frame. Once both slots are
      // occupied, wait for a slot instead of skipping source frames and applying
      // one-frame motion to older neural history. Upload still overlaps inference.
      if (state.viewer) state.viewer.renderEnabled = false;
      setStatus(`Queued fresh RGBA16F frame in slot ${pending.slotIndex + 1}/2 while WebGPU is busy`);
    } catch (error) {
      console.error(error);
      setStatus(error.message || String(error), true);
    }
  } else if (state.captureRequested && !state.running) {
    void runCapturedFrame(frame);
  }
  globalThis.__dlssWebGpuExact = state;
}

function requestNeuralFrame() {
  if (!globalThis.dlssSceneReady || !state.settings.enabled || !state.viewer || !state.frame) return;
  if (state.running) {
    state.rerunRequested = true;
  }
  state.captureRequested = true;
  state.viewer.renderEnabled = true;
  // The bridge pass is already dirty. setDirty() would restart accumulation.
  setStatus(`Requesting a fresh WebGI color frame · ${describeFrame(state.frame)}`);
}

function scheduleInitialRun() {
  if (state.initialRunScheduled || !globalThis.dlssSceneReady || !state.frame) return;
  state.initialRunScheduled = true;
  if (!state.settings.enabled) {
    setStatus('Neural rendering off · WebGI source');
    globalThis.dlssLoading?.finish();
    return;
  }
  setStatus(`${sceneLabel} loaded · resolving WebGI antialiasing`);
  globalThis.dlssLoading?.stage('Preparing first frame', `${sceneLabel} loaded · resolving antialiasing`);
  requestNeuralFrame();
}

function formatMilliseconds(value) {
  return value == null ? 'unavailable' : value.toFixed(1);
}

function toggleLive() {
  if (!globalThis.dlssSceneReady || !state.settings.enabled) return;
  state.live = !state.live;
  document.querySelector('#nrLive')?.setAttribute('aria-pressed', String(state.live));
  if (state.live) requestNeuralFrame();
  else {
    state.pendingProductionFrame = null;
    state.captureRequested = false;
    state.rerunRequested = false;
    // Finish with a converged still, rejecting any Live frame still in flight.
    state.viewRevision++;
    state.neuralReady = false;
    setNeuralVisible(false);
    requestNeuralFrame();
  }
}

function toggleComparison() {
  if (!state.settings.enabled) return;
  state.compareSource = !state.compareSource;
  if (state.compareSource) resizeHoldCanvas?.classList.remove('visible');
  else if (state.resizePending) resizeHoldCanvas?.classList.add('visible');
  setNeuralVisible(!state.compareSource);
}

function compareShortcut() {
  if (state.neuralReady) toggleComparison();
  else requestNeuralFrame();
}

let resizeTimer;
function scheduleResizeRebuild() {
  const firstEvent = !state.resizePending;
  state.resizePending = true;
  state.resizeSettled = false;
  state.resizeRevision++;
  state.viewportWidth = window.innerWidth;
  state.viewportHeight = window.innerHeight;
  if (firstEvent) {
    void preservePresentedFrame();
    if (state.settings.enabled && state.result && globalThis.dlssSceneReady) {
      globalThis.dlssLoading?.begin(
        'Resizing DLSS 5',
        'Preparing neural rendering for the new window size',
      );
    }
  }
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!globalThis.dlssSceneReady || !state.viewer) return;
    state.resizeSettled = true;
    state.pendingProductionFrame = null;
    state.captureRequested = false;
    state.rerunRequested = false;
    requestNeuralFrame();
  }, RESIZE_SETTLE_MILLISECONDS);
}

window.addEventListener('resize', scheduleResizeRebuild);

function attach(viewer) {
  if (!viewer || state.viewer === viewer) return;
  const bridge = viewer.getPlugin('DlssBridge');
  if (!bridge || typeof bridge.setFrameConsumer !== 'function') {
    setStatus('Exact WebGI scene loaded, but the rebuilt DLSS bridge consumer API is missing', true);
    return;
  }
  state.viewer = viewer;
  state.bridge = bridge;
  bridge.setFrameConsumer(consumeFrame);
  viewer.addEventListener('update', () => {
    if (!globalThis.dlssSceneReady || !state.settings.enabled) return;
    state.viewRevision++;
    if (window.innerWidth !== state.viewportWidth || window.innerHeight !== state.viewportHeight) {
      // WebGI may emit its resize update before the window listener registered
      // by this bundle. Detect it here so the old DLSS presentation is retained.
      scheduleResizeRebuild();
      return;
    }
    if (state.resizePending) {
      if (state.resizeSettled) {
        if (state.running) state.rerunRequested = true;
        else requestNeuralFrame();
      }
      return;
    }
    // Live already schedules its next frame. Do not discard its queued input,
    // hide its output, or resume a producer paused by two-slot backpressure.
    if (state.live) return;
    state.neuralReady = false;
    state.pendingProductionFrame = null;
    state.captureRequested = true;
    setNeuralVisible(false);
  });
  viewer.setDirty();
  setStatus(PRODUCTION_RUNTIME
    ? `${sceneLabel} scene · waiting for RGBA16F production frame`
    : `${sceneLabel} scene · waiting for five-texture diagnostic frame`);
}

window.addEventListener('dlss-viewer-ready', (event) => attach(event.detail));
if (globalThis.dlssViewer) attach(globalThis.dlssViewer);
else {
  const waitForViewer = setInterval(() => {
    if (!globalThis.dlssViewer) return;
    clearInterval(waitForViewer);
    attach(globalThis.dlssViewer);
  }, 50);
}

setInterval(scheduleInitialRun, 250);

window.addEventListener('keydown', (event) => {
  if (event.target instanceof Element && event.target.closest('input, select, textarea, [contenteditable=true]')) return;
  if (event.code === 'KeyL' && !event.repeat && !event.ctrlKey && !event.metaKey) {
    event.preventDefault();
    toggleLive();
  } else if (event.code === 'F6' && !event.repeat) {
    event.preventDefault();
    compareShortcut();
  } else if (event.code === 'KeyR' && !event.repeat && !event.ctrlKey && !event.metaKey) {
    event.preventDefault();
    requestNeuralFrame();
  }
}, true);

globalThis.__dlssWebGpuExact = state;
setStatus(`${sceneLabel} scene · waiting for WebGI viewer`);

let resumeSceneLive = false;
demoUi.mountRuntime(state, {
  render: requestNeuralFrame, toggleLive, compare: toggleComparison, compareShortcut,
  async beforeSceneChange() {
    clearTimeout(resizeTimer);
    releasePresentedFrame();
    resumeSceneLive = state.live;
    state.live = false;
    state.pendingProductionFrame = null;
    state.captureRequested = false;
    state.rerunRequested = false;
    if (state.viewer) state.viewer.renderEnabled = false;
    while (state.running) await new Promise(resolve => setTimeout(resolve, 16));
    state.frame = null;
    state.snapshot = null;
    state.result = null;
    state.neuralReady = false;
    state.neuralVisible = false;
    state.compareSource = false;
    state.lastNrSourceFrame = null;
    state.initialRunScheduled = false;
    state.frameCount = 0;
    outputCanvas.classList.remove('visible');
  },
  afterSceneChange() {
    sceneLabel = globalThis.dlssSceneLabel || 'Bistro';
    state.live = resumeSceneLive && state.settings.enabled;
    state.viewer?.setDirty();
  },
});

function updateNrSettings(changes) {
  const next = normalizeNrSettings(changes, state.settings);
  if (Object.keys(next).every(key => next[key] === state.settings[key])) return;
  state.settings = next;
  state.settingsRevision++;
  nrControls.sync(next);
  state.neuralReady = false;
  state.pendingProductionFrame = null;
  setNeuralVisible(false);
  if (!next.enabled) {
    clearTimeout(resizeTimer);
    releasePresentedFrame();
    globalThis.dlssLoading?.finish();
    state.live = false;
    state.captureRequested = false;
    state.rerunRequested = false;
    document.querySelector('#nrLive')?.setAttribute('aria-pressed', 'false');
    setStatus('Neural rendering off · WebGI source');
  } else {
    state.compareSource = false;
    requestNeuralFrame();
  }
}
const nrControls = PRODUCTION_RUNTIME ? mountNrControls(state.settings, updateNrSettings) : {sync() {}};
if (!PRODUCTION_RUNTIME && document.querySelector('#nrSettings')) document.querySelector('#nrSettings').hidden = true;
state.setSettings = updateNrSettings;
