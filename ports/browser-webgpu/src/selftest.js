// Runs every case in numerics_cases.js twice - once through the JS oracle, once through the WGSL on the GPU -
// and reports both against web/fixtures/numerics.bin, which holds what the Vulkan implementation produced.
//
// This is the gate the rest of the port is built on. A kernel that gets the graph right and the rounding wrong
// produces an image that looks correct and is not, so the arithmetic is settled here, on its own, first.

import { requestDevice, storageFrom, storage, readBack, compile, dispatch, bindLayout } from './gpu.js';
import { readFixture, numericsCases, compare, fixedPointBound } from './numerics_cases.js';

const base = new URL('..', import.meta.url);
const fetchText = async (path) => (await fetch(new URL(path, base))).text();

/** The half bit patterns a case produced, widened to the u32 the shader wrote. */
function readAs(kind, bytes) {
  const words = new Uint32Array(bytes);
  return (i) => (kind === 'byte' ? words[i] & 0xff : kind === 'half' ? words[i] & 0xffff : words[i]);
}

export async function runSelfTest(report) {
  report.status('requesting a device');
  const { device, info } = await requestDevice();
  report.device(info);

  report.status('loading the fixture and the shaders');
  const [numerics, selftest] = await Promise.all([
    fetchText('shaders/numerics.wgsl'),
    fetchText('shaders/selftest.wgsl'),
  ]);
  const response = await fetch(new URL('web/fixtures/numerics.bin', base));
  if (!response.ok) throw new Error('web/fixtures/numerics.bin is missing; see tools/dump_numerics.cpp');
  const sections = readFixture(new Uint8Array(await response.arrayBuffer()));
  const cases = numericsCases(sections);

  report.status('compiling');
  const code = `${numerics}\n${selftest}`;
  const module = await compile(device, code, 'selftest.wgsl');

  // Every entry point declares the same two buffers, whether or not it reads the first.
  const layout = bindLayout(device, ['read-only-storage', 'storage'], 'selftest');
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

  let failures = 0;
  for (const entry of cases) {
    report.status(`running ${entry.name}`);

    const cpu = entry.oracle === 'js' ? null : compare(entry, entry.actual);

    const inputs = storageFrom(device, entry.gpuInputs ?? new Uint32Array(1), `${entry.entryPoint} inputs`);
    const results = storage(device, entry.count * 4, `${entry.entryPoint} results`);
    const pipeline = await device.createComputePipelineAsync({
      layout: pipelineLayout,
      compute: { module, entryPoint: entry.entryPoint },
    });
    dispatch(device, pipeline, layout, [inputs, results], Math.ceil(entry.count / 64), entry.entryPoint);
    const produced = readAs(entry.kind, await readBack(device, results, entry.count * 4));
    const gpu = compare(entry, produced);
    inputs.destroy();
    results.destroy();

    if ((cpu && cpu.mismatches) || gpu.mismatches) failures += 1;
    report.row(entry, cpu, gpu);
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  const bound = fixedPointBound(cases);
  report.note(`aligned fixed-point sums reached ${bound.toLocaleString()} (2^${Math.log2(bound).toFixed(1)}); ` +
              `the i32 the WGSL accumulates in holds ${(2 ** 31 - 1).toLocaleString()}`);
  report.done(failures);
  device.destroy();
  return failures;
}
