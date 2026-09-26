// `node serve.mjs [port]`, then open the printed link. See src/server.js.

import { startServer, root } from './src/server.js';

const port = Number(process.argv[2] ?? 8099);
await startServer({ port });
console.log(`serving ${root}`);
console.log(`  demo       http://localhost:${port}/demo/index.html`);
console.log(`  parity     http://localhost:${port}/web/parity.html`);
console.log(`  self-test  http://localhost:${port}/web/selftest.html`);
console.log('  weights /weights (NR_WEIGHTS) · fixtures /fixtures (NR_FIXTURES) · scenes /scenes (NR_SCENES)');
