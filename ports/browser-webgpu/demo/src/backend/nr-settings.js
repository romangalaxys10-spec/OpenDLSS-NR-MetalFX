// Feature-18 controls. Display/transport controls from the Electron wrapper
// intentionally do not belong to this contract.
export const NR_DEFAULTS = Object.freeze({enabled: true, intensity: 1, localTone: 1,
  localStructure: 1, skinStructure: -1, autoMask: true, style: 0, preset: 0,
  uiCorrection: false});

const ranges = {intensity: [0, 1], localTone: [0, 2], localStructure: [0, 2],
  skinStructure: [-1, 2], style: [0, 2], preset: [0, 3]};
export function normalizeNrSettings(changes = {}, previous = NR_DEFAULTS) {
  const result = {...previous};
  for (const [key, value] of Object.entries(changes)) {
    if (!Object.hasOwn(NR_DEFAULTS, key)) throw new Error(`Unknown DLSS-NR setting: ${key}`);
    if (typeof NR_DEFAULTS[key] === 'boolean') {
      if (![true, false, 0, 1].includes(value)) throw new Error(`${key} must be a boolean`);
      result[key] = Boolean(value);
    } else {
      if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${key} must be finite`);
      const [min, max] = ranges[key];
      result[key] = Math.min(max, Math.max(min, value));
      if (key === 'style' || key === 'preset') result[key] = Math.round(result[key]);
      if (key === 'skinStructure' && result[key] < 0) result[key] = -1;
    }
  }
  return result;
}

export function nrConditioning(settings) {
  const s = normalizeNrSettings(settings);
  return [s.style / 128, s.localTone, s.autoMask ? 1 : s.localStructure,
    s.autoMask ? (s.skinStructure < 0 ? s.localStructure : s.skinStructure) : -1,
    s.autoMask ? s.localStructure : -1];
}

// Native cg2r post-processing for Natural and Cinematic, after the first
// half-surface publication. Keep both HSL conversions and log2/exp2 stages:
// apparently identity stages still publish f32 rounding in the native kernel.
// Ada's PTX JIT contracts the curve subtraction, hue sector offset and upper
// HSL q expression. Spell out their FMAs; splitting them changes half ties.
export const NR_STYLE_WGSL = String.raw`
fn nr_publish(proxy: vec3<f32>, head: vec3<f32>, intensity: f32, style: f32, tone: f32) -> vec3<f32> {
  var neural: vec3<f32>;
  for (var c = 0u; c < 3u; c++) {
    let centered = proxy[c] * 0.125 - 0.0625;
    let residual = select(0.0, head[c] * 0.03125, head[c] == head[c] && abs(head[c]) < 3.402823e38);
    let mixed = centered + residual;
    let value = clamp(mixed * 8.0 + 0.5, 0.0, 1.0);
    if (style == 0.0 && intensity < 1.0) {
      neural[c] = truncate_half(clamp(fma(intensity, value - proxy[c], proxy[c]), 0.0, 1.0));
    } else { neural[c] = truncate_half(value); }
  }
  if (style != 0.0) {
    let styled = nr_style(neural, style, tone);
    for (var c = 0u; c < 3u; c++) {
      neural[c] = truncate_half(clamp(fma(intensity, styled[c] - proxy[c], proxy[c]), 0.0, 1.0));
    }
  }
  return neural;
}
fn nr_hsl(rgb: vec3<f32>) -> vec3<f32> {
  let high = max(max(rgb.r, rgb.g), rgb.b);
  let low = min(min(rgb.r, rgb.g), rgb.b);
  let sum = high + low;
  let light = sum * 0.5;
  var hue = 0.0; var saturation = 0.0;
  if (high > low) {
    let delta = high - low;
    if (light > 0.5) { saturation = delta / ((2.0 - high) - low); }
    else { saturation = delta / sum; }
    if (high == rgb.r) {
      hue = fma(rgb.g - rgb.b, 1.0 / delta, select(0.0, 6.0, rgb.g < rgb.b)) / 6.0;
    } else if (high == rgb.g) { hue = fma(rgb.b - rgb.r, 1.0 / delta, 2.0) / 6.0; }
    else { hue = fma(rgb.r - rgb.g, 1.0 / delta, 4.0) / 6.0; }
  }
  return vec3<f32>(hue, saturation, light);
}
fn nr_hue(p: f32, q: f32, hue: f32) -> f32 {
  var h = hue;
  if (h < 0.0) { h += 1.0; }
  if (h > 1.0) { h -= 1.0; }
  if (h < bitcast<f32>(0x3e2aaaabu)) { return fma(h, (q - p) * 6.0, p); }
  if (h < 0.5) { return q; }
  if (h < bitcast<f32>(0x3f2aaaabu)) {
    return fma((bitcast<f32>(0x3f2aaaabu) - h) * (q - p), 6.0, p);
  }
  return p;
}
fn nr_rgb(hsl: vec3<f32>) -> vec3<f32> {
  if (hsl.y <= 0.0) { return vec3<f32>(hsl.z); }
  var q: f32;
  if (hsl.z < 0.5) { q = hsl.z * (hsl.y + 1.0); }
  else { q = fma(-hsl.z, hsl.y, hsl.z + hsl.y); }
  let p = (hsl.z + hsl.z) - q;
  return vec3<f32>(nr_hue(p, q, hsl.x + bitcast<f32>(0x3eaaaaabu)),
    nr_hue(p, q, hsl.x), nr_hue(p, q, hsl.x - bitcast<f32>(0x3eaaaaabu)));
}
fn nr_style(neural: vec3<f32>, style: f32, tone: f32) -> vec3<f32> {
  // Native conditioning accepts tone up to two, but cg2r caps its adjustment at one.
  let style_tone = clamp(tone, 0.0, 1.0);
  let exposure = select(0.0, -0.1 * style_tone, style == 1.0);
  let contrast = select(0.0, -0.25 * style_tone, style == 1.0);
  let saturation = select(-0.15 * style_tone, -0.1 * style_tone, style == 1.0);
  var color: vec3<f32>;
  for (var c = 0u; c < 3u; c++) {
    let value = clamp(neural[c], 0.0, 1.0);
    let exposed = clamp(value * exp2(exposure), 0.0, 1.0);
    let square = exposed * exposed;
    let curve_delta = fma(square, 3.0 - (exposed + exposed), -exposed);
    let curved = clamp(fma(contrast, curve_delta, exposed), 0.0, 1.0);
    let first_gamma = exp2(log2(curved));
    color[c] = exp2(log2(max(0.0, first_gamma)));
  }
  let hsl = nr_hsl(color);
  let adjusted = nr_rgb(vec3<f32>(hsl.x, clamp(hsl.y * (saturation + 1.0), 0.0, 1.0), hsl.z));
  let again = nr_hsl(adjusted);
  return clamp(nr_rgb(vec3<f32>(again.x, clamp(exp2(log2(again.y)), 0.0, 1.0), again.z)),
    vec3<f32>(0.0), vec3<f32>(1.0));
}
`;
