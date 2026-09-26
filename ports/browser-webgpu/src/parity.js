// The parity harness: run the port on a recorded frame and compare it, byte for byte, against what native
// produced for the same input. It is the browser twin of `dlss5vk parity`, with the same fixture contract.
//
// A fixture says what it exists to check. Its manifest lists the checks ("boundaries", "head", "output") and
// carries one reference for each; `planFixture` validates all of it before anything runs, and nothing is ever
// dropped from a count. A declared check that cannot run, a reference that is missing, short or unknown, or a
// reference that no declared check uses rejects the fixture - a truncated export must fail, not turn quietly
// into a smaller suite. Every verdict names the equality it proved: bit-exact, equal only up to the sign of zero
// (a failure: those are not the same bytes), or within a stated tolerance.
//
// A boundary fixture reads its input features rather than generating them, for a reason worth stating: the
// three noise lanes are Box-Muller over the GPU's *approximate* f32 transcendentals, so their last bits
// depend on which transcendental unit evaluated them. Everything downstream of the features is exactly
// reproducible; the features themselves are the one place a port cannot be, so that comparison starts after
// them. An end-to-end fixture records the proxy instead and generates the features from it, which puts those
// three lanes back inside the comparison - on the card this was measured on they reproduce, but that is a
// measurement rather than a guarantee.

const CODE_VALUES = (() => {
  const values = new Float32Array(256);
  for (let byte = 0; byte < 256; ++byte) {
    const sign = (byte & 0x80) ? -1 : 1;
    const exponent = (byte >>> 3) & 0xf;
    const mantissa = byte & 0x7;
    values[byte] = exponent === 0 ? sign * mantissa * 2 ** -9
                 : (exponent === 0xf && mantissa === 0x7) ? sign * 0
                 : sign * (1 + mantissa / 8) * 2 ** (exponent - 7);
  }
  return values;
})();

/**
 * The stored outputs a boundary fixture can hold references for, in graph order: blocks 0-69 and the five
 * encoder stage transitions. Block 70 feeds the head directly; the pooled-* captures have no reference.
 */
export const REFERENCE_BOUNDARIES = (() => {
  const names = [];
  for (let block = 0; block <= 69; ++block) {
    names.push(`block-${block}`);
    if ([0, 4, 8, 14, 22].includes(block)) names.push(`transition-${block}-${block + 1}`);
  }
  return names;
})();

/** One comparison's outcome: `verdict` is 'bit-exact', 'signed-zero', 'within-tolerance' or 'mismatch'. */
function tally(count) {
  return { count, mismatches: 0, signedZeros: 0, tolerated: 0, maxAbs: 0, sumSquares: 0, first: null, deltas: new Map() };
}

function finish(t) {
  t.verdict = t.mismatches ? 'mismatch' : t.signedZeros ? 'signed-zero' : t.tolerated ? 'within-tolerance' : 'bit-exact';
  t.rmse = Math.sqrt(t.sumSquares / Math.max(1, t.count));
  t.deltas = [...t.deltas.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  return t;
}

function miss(t, index, actual, expected, difference) {
  t.mismatches += 1;
  if (!t.first) t.first = { index, actual, expected };
  t.maxAbs = Math.max(t.maxAbs, Math.abs(difference));
  t.sumSquares += difference * difference;
}

function sameLength(actual, expected) {
  if (actual.length !== expected.length) {
    throw new Error(`cannot compare ${actual.length} values with a reference of ${expected.length}`);
  }
}

/** Byte for byte. A difference that is only the sign of a zero is counted apart: equivalent, not the same bytes. */
export function compareCodes(actual, expected) {
  sameLength(actual, expected);
  const t = tally(expected.length);
  for (let i = 0; i < expected.length; ++i) {
    const a = actual[i];
    const e = expected[i];
    if (a === e) continue;
    if ((a & 0x7f) === 0 && (e & 0x7f) === 0) { t.signedZeros += 1; continue; }
    miss(t, i, a, e, CODE_VALUES[a] - CODE_VALUES[e]);
    // A signed code delta says whether the port rounded one step up or down, which is a different problem
    // from a structural error; a histogram distinguishes the two at a glance.
    const delta = (a & 0x80 ? -(a & 0x7f) : a & 0x7f) - (e & 0x80 ? -(e & 0x7f) : e & 0x7f);
    t.deltas.set(delta, (t.deltas.get(delta) ?? 0) + 1);
  }
  return finish(t);
}

/** Bitwise, because these are the same numbers or they are not: +0 is not -0, and a NaN equals only itself. */
export function compareFloats(actual, expected) {
  sameLength(actual, expected);
  const a32 = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const e32 = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  const t = tally(expected.length);
  for (let i = 0; i < expected.length; ++i) {
    if (a32[i] === e32[i]) continue;
    if (((a32[i] | e32[i]) & 0x7fffffff) === 0) { t.signedZeros += 1; continue; }
    miss(t, i, actual[i], expected[i], actual[i] - expected[i]);
  }
  return finish(t);
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * The fixture's declared checks, validated against what it carries: every problem is collected, and any one
 * rejects the fixture before the network runs. Three checks exist, and they reach different parts of the port:
 *
 *   * "boundaries" compares the block outputs and stage transitions (from recorded input features, or from a
 *     proxy). Every name in REFERENCE_BOUNDARIES needs a reference or an entry in "omittedBoundaries" with the
 *     reason, so a shortened export cannot pass as a smaller suite;
 *   * "head" compares the f32 RGBA head against a recorded head, bitwise;
 *   * "output" compares the composed image: RGBA f32 halves from a proxy fixture, bitwise, or an older RGBA8
 *     capture, which can only be compared within one code (its own rounding is not known exactly).
 */
export async function loadFixture(directory) {
  const response = await fetch(`${directory}/manifest.json`);
  if (!response.ok) throw new Error(`${directory}/manifest.json: HTTP ${response.status}`);
  const manifest = await response.json();
  const problems = [];
  const plan = { directory, manifest, boundaries: [], omitted: [], checks: new Set() };
  const [validWidth, validHeight] = manifest.sourceDimensions ?? [];
  const [fullWidth, fullHeight] = manifest.fullDimensions ?? [];
  if (!(validWidth > 0 && validHeight > 0)) problems.push('"sourceDimensions" must be [width, height]');
  if (!(fullWidth > 0 && fullHeight > 0)) problems.push('"fullDimensions" must be [width, height]');
  const expectSize = (what, entry, bytes) => {
    if (!entry?.file) { problems.push(`${what}: no "file"`); return null; }
    return { what, url: `${directory}/${entry.file}`, bytes };
  };
  const reads = [];

  if (!!manifest.proxy === !!manifest.inputFeatures) problems.push('exactly one of "proxy" and "inputFeatures" is the input');
  if (manifest.proxy) {
    reads.push(['proxy', expectSize('proxy', manifest.proxy, manifest.proxy.width * manifest.proxy.height * 16)]);
  } else if (manifest.inputFeatures) {
    reads.push(['features', expectSize('inputFeatures', manifest.inputFeatures, fullWidth * fullHeight * 64)]);
  }

  if (!Array.isArray(manifest.checks) || manifest.checks.length === 0) {
    problems.push('"checks" must list what the fixture gates: "boundaries", "head", "output"');
  } else {
    for (const check of manifest.checks) {
      if (!['boundaries', 'head', 'output'].includes(check)) problems.push(`unknown check "${check}"`);
      else if (plan.checks.has(check)) problems.push(`check "${check}" listed twice`);
      plan.checks.add(check);
    }
  }

  const listed = new Map();
  for (const entry of manifest.blocks ?? []) listed.set(`block-${entry.block}`, entry);
  for (const entry of manifest.transitions ?? []) listed.set(`transition-${entry.id}`, entry);
  const omitted = manifest.omittedBoundaries ?? {};
  if (plan.checks.has('boundaries')) {
    if (listed.size !== (manifest.blocks ?? []).length + (manifest.transitions ?? []).length) {
      problems.push('a boundary is listed twice');
    }
    for (const name of listed.keys()) {
      if (!REFERENCE_BOUNDARIES.includes(name)) problems.push(`${name} is not a boundary of this graph`);
    }
    const unaccounted = [];
    for (const name of REFERENCE_BOUNDARIES) {
      const entry = listed.get(name);
      if (entry) {
        const shape = expectSize(name, entry, entry.width * entry.height * entry.channels);
        if (shape) plan.boundaries.push({ name, entry, ...shape });
        if (omitted[name]) problems.push(`${name} is both compared and declared omitted`);
      } else if (omitted[name]) {
        plan.omitted.push([name, omitted[name]]);
      } else {
        unaccounted.push(name);
      }
    }
    for (const [name, reason] of Object.entries(omitted)) {
      if (!REFERENCE_BOUNDARIES.includes(name)) problems.push(`omitted ${name} is not a boundary of this graph`);
      if (!reason) problems.push(`omitted boundary ${name} gives no reason`);
    }
    if (unaccounted.length) {
      problems.push(`${unaccounted.length} of ${REFERENCE_BOUNDARIES.length} graph boundaries have neither a reference ` +
                    `nor an omission reason (${unaccounted.slice(0, 8).join(', ')}${unaccounted.length > 8 ? ', ...' : ''})`);
    }
  } else if (listed.size || manifest.omittedBoundaries) {
    problems.push('the fixture carries boundary references but does not declare the "boundaries" check');
  }

  if (plan.checks.has('head') !== !!manifest.referenceHead) {
    problems.push(plan.checks.has('head') ? 'the "head" check needs "referenceHead"'
                                          : '"referenceHead" is carried but the "head" check is not declared');
  } else if (manifest.referenceHead) {
    reads.push(['referenceHead', expectSize('referenceHead', manifest.referenceHead, fullWidth * fullHeight * 16)]);
  }

  if (plan.checks.has('output') !== !!manifest.nativeOutput) {
    problems.push(plan.checks.has('output') ? 'the "output" check needs "nativeOutput"'
                                            : '"nativeOutput" is carried but the "output" check is not declared');
  } else if (manifest.nativeOutput) {
    const output = manifest.nativeOutput;
    if (output.width !== validWidth || output.height !== validHeight) problems.push('nativeOutput is not the source size');
    plan.outputBytes = output.dtype === 'u8';
    if (output.dtype === 'f32' && !manifest.proxy) problems.push('an f32 nativeOutput is composed from the proxy, and the fixture has none');
    if (output.dtype !== 'f32' && output.dtype !== 'u8') problems.push('nativeOutput "dtype" must be "f32" or "u8"');
    else reads.push(['nativeOutput', expectSize('nativeOutput', output, validWidth * validHeight * (plan.outputBytes ? 4 : 16))]);
  }

  // Every reference is fetched now (the boundaries only have their size checked), so a missing or short file
  // is found before the network runs rather than half way through the table.
  const loaded = await Promise.all(reads.filter(([, shape]) => shape).map(async ([key, shape]) => {
    try {
      const bytes = await fetchBytes(shape.url);
      if (bytes.length !== shape.bytes) problems.push(`${shape.what}: ${shape.url} holds ${bytes.length} bytes, its shape needs ${shape.bytes}`);
      return [key, bytes];
    } catch (error) {
      problems.push(`${shape.what}: ${error.message}`);
      return [key, null];
    }
  }));
  await Promise.all(plan.boundaries.map(async (boundary) => {
    const head = await fetch(boundary.url, { method: 'HEAD' });
    const length = Number(head.headers.get('content-length'));
    if (!head.ok) problems.push(`${boundary.name}: ${boundary.url}: HTTP ${head.status}`);
    else if (length !== boundary.bytes) problems.push(`${boundary.name}: ${boundary.url} holds ${length} bytes, its shape needs ${boundary.bytes}`);
  }));
  if (problems.length) throw new Error(`fixture rejected (${problems.length} problem${problems.length > 1 ? 's' : ''}); nothing was run:\n  ` +
                                       problems.join('\n  '));
  for (const [key, bytes] of loaded) {
    plan[key] = key === 'nativeOutput' && plan.outputBytes ? bytes : new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
  }
  return plan;
}

export async function runParity(network, fixture, report, { repeat = 1, profile = false } = {}) {
  const geometry = network.geometry;
  const [width, height] = fixture.manifest.sourceDimensions;
  if (width !== geometry.validWidth || height !== geometry.validHeight) {
    throw new Error(`fixture is ${width}x${height}, the network was built for ` +
                    `${geometry.validWidth}x${geometry.validHeight}`);
  }
  if (fixture.manifest.fullDimensions[0] !== geometry.fullWidth || fixture.manifest.fullDimensions[1] !== geometry.fullHeight) {
    throw new Error('fixture full dimensions disagree with the network geometry');
  }
  report.note(`checks: ${[...fixture.checks].join(', ')}` +
              (fixture.checks.has('boundaries') ? ` (${fixture.boundaries.length} boundary references, ${fixture.omitted.length} declared omitted)` : ''));
  for (const [name, reason] of fixture.omitted) report.note(`  ${name} omitted: ${reason}`);

  report.status('running the graph');
  if (profile) network.recorder.enableProfiling();
  if (fixture.features) network.writeFeatures(fixture.features);
  else await network.featuresFromProxy(fixture.proxy, fixture.manifest);
  // The first run pays for shader compilation and buffer residency, so a timing is only meaningful once the
  // graph has been through at least once. `repeat` reports the best of several, which is the number to quote.
  let elapsed = Infinity;
  for (let pass = 0; pass < Math.max(1, repeat); ++pass) {
    const started = performance.now();
    await network.run();
    const took = performance.now() - started;
    if (pass > 0 || repeat === 1) elapsed = Math.min(elapsed, took);
    report.status(`run ${pass + 1} of ${repeat}: ${took.toFixed(0)} ms`);
  }
  report.timing(elapsed, network.recorder.dispatchCount);
  const breakdown = profile ? await network.recorder.readProfile() : null;
  for (const kernel of breakdown ?? []) {
    report.note(`  ${kernel.name.padEnd(18)} ${kernel.milliseconds.toFixed(1)} ms`);
  }

  const summary = { 'bit-exact': 0, 'within-tolerance': 0, failed: 0 };
  const count = (result) => {
    if (result.verdict === 'bit-exact' || result.verdict === 'within-tolerance') summary[result.verdict] += 1;
    else summary.failed += 1;
  };
  const available = new Set(network.boundaryNames);
  for (const { name, entry, url } of fixture.boundaries) {
    const shape = `${entry.width}x${entry.height}x${entry.channels}`;
    if (!available.has(name)) { report.row(name, 'NOT CAPTURED by this graph', shape); summary.failed += 1; continue; }
    report.status(`comparing ${name}`);
    const expected = await fetchBytes(url);
    const actual = await network.readBoundary(name);
    if (actual.length !== expected.length) {
      report.row(name, `SHAPE: the graph has ${actual.length} bytes, the reference ${expected.length}`, shape);
      summary.failed += 1;
      continue;
    }
    const result = compareCodes(actual, expected);
    count(result);
    report.row(name, result, shape);
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  // The boundaries stop at block 69. Everything after it - the full-resolution post block, the head matrix,
  // and the composition - is only reached by these two.
  const head = fixture.referenceHead || fixture.nativeOutput ? await network.readHead() : null;

  if (fixture.referenceHead) {
    report.status('comparing the head');
    const result = compareFloats(head, fixture.referenceHead);
    count(result);
    report.row('head', result, `${geometry.fullWidth}x${geometry.fullHeight}x4`);
  }

  if (fixture.nativeOutput) {
    report.status('comparing the composed image');
    const result = compareComposed(head, fixture, geometry, width, height);
    count(result);
    report.row('composed rgb', result, `${width}x${height}x3`);
    if (result.verdict === 'within-tolerance') {
      report.note(`  composed rgb (RGBA8 capture): ${result.count - result.tolerated} exact, ${result.tolerated} one code off`);
    }
  }

  report.done(summary);
  return summary;
}

/** One f32 fused multiply-add: the product is exact in f64, so a single narrowing is the fma's own. */
const fma32 = (a, b, c) => Math.fround(a * b + c);

/** The centred proxy of one pixel, from whichever of the two the fixture carries. */
function centredAt(fixture, x, y, c, width, fullWidth) {
  // A proxy fixture stores the code value; a feature fixture stores the centring of it, which is the same
  // number the composition wants and is what native's own kernel computed.
  return fixture.proxy ? fma32(fixture.proxy[(y * width + x) * 4 + c], 0.125, -0.0625)
                       : fixture.features[(y * fullWidth + x) * 16 + 4 + c];
}

/**
 * The composition, exactly as the demo publishes it and as the native capture recorded it: the head's
 * residual on the centred proxy, back to a code value, truncated to the half grid. No history (the capture
 * is one frame), no style, no display transform - the fixture stops where the renderer would take over.
 * `inner * 8` is exact, so there is a single rounding whether or not a compiler contracts the last step.
 *
 * Two kinds of reference exist. A float capture is compared bit for bit; an older byte capture is the same
 * image quantized to eight bits by a rounding that is not known exactly, so it is compared within one code,
 * and the verdict says so.
 */
export function compareComposed(head, fixture, geometry, width, height) {
  const expected = fixture.nativeOutput;
  const bytes = expected instanceof Uint8Array;
  const t = tally(width * height * 3);
  const one = new Float32Array(2);
  const bits = new Uint32Array(one.buffer);
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      for (let c = 0; c < 3; ++c) {
        const h = head[(y * geometry.fullWidth + x) * 4 + c];
        const inner = fma32(h, 0.03125, centredAt(fixture, x, y, c, width, geometry.fullWidth));
        const value = truncateToHalf(Math.min(Math.max(Math.fround(inner * 8 + 0.5), 0), 1));
        const want = expected[(y * width + x) * 4 + c];
        if (bytes) {
          const published = quantizeByte(value);
          if (published === want) continue;
          if (Math.abs(published - want) === 1) { t.tolerated += 1; continue; }
          miss(t, (y * width + x) * 3 + c, published, want, published - want);
        } else {
          one[0] = value; one[1] = want;
          if (bits[0] === bits[1]) continue;
          if (((bits[0] | bits[1]) & 0x7fffffff) === 0) { t.signedZeros += 1; continue; }
          miss(t, (y * width + x) * 3 + c, value, want, value - want);
        }
      }
    }
  }
  return finish(t);
}

/** The eight-bit publication, evaluated in f32 the way the capture that produced the byte reference did. */
const quantizeByte = (value) => Math.min(255, Math.max(0, Math.floor(Math.fround(value * 255 + 0.5))));

/** Toward zero to the half grid: what the publication does, and not the same as rounding. */
export function truncateToHalf(value) {
  const scratch = truncateToHalf.scratch ?? (truncateToHalf.scratch = {
    floats: new Float32Array(1), bits: null, half: new Uint16Array(1), out: new Float32Array(1),
  });
  scratch.bits = scratch.bits ?? new Uint32Array(scratch.floats.buffer);
  scratch.floats[0] = value;
  const bits = scratch.bits[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  let half;
  if (exponent === 0xff) half = sign | (mantissa ? 0x7e00 : 0x7c00);
  else {
    const halfExponent = exponent - 112;
    if (halfExponent >= 31) half = sign | 0x7c00;
    else if (halfExponent <= 0) half = halfExponent < -10 ? sign : sign | ((mantissa | 0x800000) >>> (14 - halfExponent));
    else half = sign | (halfExponent << 10) | (mantissa >>> 13);
  }
  const s = (half & 0x8000) ? -1 : 1;
  const e = (half >>> 10) & 0x1f;
  const m = half & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 0x1f) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/**
 * The same composition as `compareComposed`, as a picture. It is the network's own output for the frame with
 * no history to blend and no display transform: a recorded frame has neither a previous frame nor the scene
 * the renderer would tone map back into.
 */
export function composeImage(head, fixture, geometry) {
  const { validWidth, validHeight, fullWidth } = geometry;
  const pixels = new Uint8ClampedArray(validWidth * validHeight * 4);
  for (let y = 0; y < validHeight; ++y) {
    for (let x = 0; x < validWidth; ++x) {
      const out = (y * validWidth + x) * 4;
      for (let c = 0; c < 3; ++c) {
        const inner = fma32(head[(y * fullWidth + x) * 4 + c] , 0.03125,
                            centredAt(fixture, x, y, c, validWidth, fullWidth));
        const value = truncateToHalf(Math.min(Math.max(Math.fround(inner * 8 + 0.5), 0), 1));
        pixels[out + c] = quantizeByte(value);
      }
      pixels[out + 3] = 255;
    }
  }
  return new ImageData(pixels, validWidth, validHeight);
}
