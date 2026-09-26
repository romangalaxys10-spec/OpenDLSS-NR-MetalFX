// The cases web/fixtures/numerics.bin answers, shared by the Node check (tools/check_numerics.mjs) and the
// browser self-test (web/selftest.html). Each case names what the Vulkan reference produced, how the JS
// oracle in numerics.js reproduces it, and which WGSL entry point in shaders/selftest.wgsl reproduces it on
// the GPU. Keeping the three in one table is what stops a case from being checked on one side only.
//
// The fixture stores results and not inputs: both sides redraw them from the same 32-bit xorshift.

import * as num from './numerics.js';

/** The generator the dumper drew its inputs from. */
export class Xorshift {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    let x = this.state;
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    this.state = x;
    return x;
  }
}

/** Non-finite operands take separately listed paths in the kernels, so the random cases stay finite. */
export function finiteHalf(draw) {
  let bits = draw & 0xffff;
  if ((bits & 0x7c00) === 0x7c00) bits &= 0x7bff;
  return bits;
}

const isHalfNaN = (bits) => (bits & 0x7c00) === 0x7c00 && (bits & 0x03ff) !== 0;

/**
 * The three functions built on clamp() are compared over finite halves only.
 *
 * WGSL leaves min() and max() with a NaN operand unspecified, where the C++ reference's std::min and std::max
 * propagate it, so the two disagree on what clamp(NaN, lo, hi) is - and any WGSL written to match would be
 * relying on behaviour the specification does not give. Nothing reaches them with a NaN: a score is a dot
 * product of finite E4M3 values, and an activation that would have been a NaN publishes as +0 first.
 */
const nonFiniteHalf = (bits) => (bits & 0x7c00) === 0x7c00;

export function readFixture(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  if (magic !== 'NRNUM001') throw new Error(`not a numerics fixture: ${magic}`);
  const sections = new Map();
  let offset = 12;
  for (let i = 0; i < view.getUint32(8, true); ++i) {
    const tag = String.fromCharCode(...bytes.subarray(offset, offset + 4));
    const length = view.getUint32(offset + 4, true);
    sections.set(tag, bytes.subarray(offset + 8, offset + 8 + length));
    offset += 8 + length;
  }
  return sections;
}

/** Pack an array of f32 values as the u32 bit patterns a storage buffer hands the shader. */
function packF32(values) {
  const words = new Uint32Array(values.length);
  for (let i = 0; i < values.length; ++i) words[i] = num.f32Bits(values[i]);
  return words;
}

export function numericsCases(sections) {
  const halves = (tag) => {
    const s = sections.get(tag);
    return new Uint16Array(s.buffer, s.byteOffset, s.byteLength / 2);
  };
  const words = (tag) => {
    const s = sections.get(tag);
    return new Uint32Array(s.buffer, s.byteOffset, s.byteLength / 4);
  };
  const cases = [];

  {
    const expected = halves('F16B');
    const rng = new Xorshift(0x9e3779b9);
    const inputs = new Uint32Array(expected.length);
    for (let i = 0; i < inputs.length; ++i) inputs[i] = rng.next();
    cases.push({
      name: 'f16Bits over arbitrary f32 patterns',
      count: expected.length,
      expected,
      kind: 'half',
      actual: (i) => num.f16Bits(num.f32FromBits(inputs[i])),
      entryPoint: 'case_f16_bits',
      gpuInputs: inputs,
    });
  }

  cases.push({
    name: 'e4m3FromF16Bits over every half',
    count: 65536,
    expected: sections.get('E4EN'),
    kind: 'byte',
    actual: (i) => num.e4m3FromF16Bits(i),
    entryPoint: 'case_e4m3_encode',
  });

  cases.push({
    name: 'e4m3ToNumber over every byte',
    count: 256,
    expected: words('E4DE'),
    kind: 'word',
    actual: (i) => num.f32Bits(num.e4m3ToNumber(i)),
    entryPoint: 'case_e4m3_decode',
  });

  cases.push({
    name: 'mpCubicSilu over every half',
    count: 65536,
    expected: halves('SILU'),
    kind: 'half',
    actual: (i) => num.f16Bits(num.mpCubicSilu(num.f16ToNumber(i))),
    entryPoint: 'case_silu',
    skipInput: nonFiniteHalf,
  });

  cases.push({
    name: 'expWeight over every half',
    count: 65536,
    expected: halves('EXPW'),
    kind: 'half',
    actual: (i) => num.f16Bits(num.expWeight(num.f16ToNumber(i))),
    entryPoint: 'case_exp_weight',
    skipInput: nonFiniteHalf,
  });

  {
    const expected = halves('FDP8');
    const rng = new Xorshift(0x85ebca6b);
    const a = new Float64Array(expected.length * 16);
    const b = new Float64Array(expected.length * 16);
    const accumulators = new Float64Array(expected.length);
    const packed = new Uint32Array(expected.length * 33);
    for (let c = 0; c < expected.length; ++c) {
      for (let i = 0; i < 16; ++i) a[c * 16 + i] = num.e4m3ToNumber(rng.next() & 0xff);
      for (let i = 0; i < 16; ++i) b[c * 16 + i] = num.e4m3ToNumber(rng.next() & 0xff);
      accumulators[c] = num.f16ToNumber(finiteHalf(rng.next()));
      packed.set(packF32(a.subarray(c * 16, c * 16 + 16)), c * 33);
      packed.set(packF32(b.subarray(c * 16, c * 16 + 16)), c * 33 + 16);
      packed[c * 33 + 32] = num.f32Bits(accumulators[c]);
    }
    cases.push({
      name: 'adaFp8Fdpa16 over random E4M3 operands',
      count: expected.length,
      expected,
      kind: 'half',
      operands: { a, b, accumulators, width: 16 },
      actual: (c) => num.f16Bits(num.adaFp8Fdpa16(a.subarray(c * 16, c * 16 + 16),
                                                  b.subarray(c * 16, c * 16 + 16), 16, accumulators[c])),
      entryPoint: 'case_fdpa_fp8',
      gpuInputs: packed,
    });
  }

  {
    const expected = halves('FD16');
    const rng = new Xorshift(0xc2b2ae35);
    const a = new Float64Array(expected.length * 8);
    const b = new Float64Array(expected.length * 8);
    const accumulators = new Float64Array(expected.length);
    const packed = new Uint32Array(expected.length * 17);
    for (let c = 0; c < expected.length; ++c) {
      for (let i = 0; i < 8; ++i) a[c * 8 + i] = num.f16ToNumber(finiteHalf(rng.next()));
      for (let i = 0; i < 8; ++i) b[c * 8 + i] = num.f16ToNumber(finiteHalf(rng.next()));
      accumulators[c] = num.f16ToNumber(finiteHalf(rng.next()));
      packed.set(packF32(a.subarray(c * 8, c * 8 + 8)), c * 17);
      packed.set(packF32(b.subarray(c * 8, c * 8 + 8)), c * 17 + 8);
      packed[c * 17 + 16] = num.f32Bits(accumulators[c]);
    }
    cases.push({
      name: 'adaF16Fdpa8 over random half operands',
      count: expected.length,
      expected,
      kind: 'half',
      operands: { a, b, accumulators, width: 8 },
      actual: (c) => num.f16Bits(num.adaF16Fdpa8(a.subarray(c * 8, c * 8 + 8),
                                                 b.subarray(c * 8, c * 8 + 8), 8, accumulators[c])),
      entryPoint: 'case_fdpa_f16',
      gpuInputs: packed,
    });
  }

  // The ViT exponential has no entry in the fixture: the Vulkan tree computes it only in its shaders, and its
  // CPU reference never needed one. The JS oracle stands in, so this case checks that the WGSL and the JS
  // transcribed the same function - weaker than the others, and marked as such.
  {
    const expected = new Uint16Array(65536);
    for (let bits = 0; bits < 65536; ++bits) {
      expected[bits] = num.f16Bits(num.vitExpWeight(num.f16ToNumber(bits)));
    }
    cases.push({
      name: 'vitExpWeight over every half (against the JS oracle, not the reference)',
      count: 65536,
      expected,
      kind: 'half',
      oracle: 'js',
      actual: (i) => expected[i],
      entryPoint: 'case_vit_exp_weight',
    skipInput: nonFiniteHalf,
    });
  }

  return cases;
}

/**
 * Compare one case. NaN half patterns compare equal to each other and are counted apart: a JS number cannot
 * carry a NaN payload or sign, so the oracle publishes the canonical one where the reference keeps the
 * input's. Nothing in the graph feeds a NaN into these functions - a NaN activation publishes as +0.
 */
export function compare(entry, produced) {
  let mismatches = 0;
  let nanPairs = 0;
  let skipped = 0;
  let first = null;
  for (let i = 0; i < entry.count; ++i) {
    if (entry.skipInput && entry.skipInput(i)) { skipped += 1; continue; }
    const expected = entry.expected[i];
    const actual = produced(i);
    if (expected === actual) continue;
    if (entry.kind === 'half' && isHalfNaN(expected) && isHalfNaN(actual)) { nanPairs += 1; continue; }
    mismatches += 1;
    if (!first) first = { i, expected, actual };
  }
  return { mismatches, nanPairs, skipped, first };
}

/** The i32 accumulator in the WGSL is only valid while the aligned sums stay inside it; measure them. */
export function fixedPointBound(cases) {
  let widest = 0;
  for (const entry of cases) {
    if (!entry.operands) continue;
    const { a, b, accumulators, width } = entry.operands;
    const fractionBits = width === 16 ? 13 : 24;
    const exponentOf = width === 16 ? num.e4m3Exponent : num.f16Exponent;
    for (let c = 0; c < entry.count; ++c) {
      let maximumExponent = num.f13Start(accumulators[c]);
      for (let i = 0; i < width; ++i) {
        const x = a[c * width + i];
        const y = b[c * width + i];
        if (x !== 0 && y !== 0) maximumExponent = Math.max(maximumExponent, exponentOf(x) + exponentOf(y));
      }
      const scale = 2 ** (fractionBits - maximumExponent);
      let units = Math.trunc(Math.fround(accumulators[c] * scale));
      for (let i = 0; i < width; ++i) {
        units += Math.trunc(Math.fround(Math.fround(a[c * width + i] * b[c * width + i]) * scale));
      }
      widest = Math.max(widest, Math.abs(units));
    }
  }
  return widest;
}
