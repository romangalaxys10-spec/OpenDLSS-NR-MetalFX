// Bundles the demo. Two outputs: the WebGI viewer half (TypeScript, resolving every `webgi/*` and `three`
// import to the CDN bundle's globals) and the WebGPU runtime half (plain modules, including the port itself).
//
// The port's own pages need no build at all; this exists only because WebGI ships as an IIFE bundle that
// publishes its API on `window`, and the viewer code is written against normal imports.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = path.dirname(fileURLToPath(import.meta.url));
const source = (...parts) => path.join(root, 'src', ...parts);
const production = process.argv.includes('--production');

// WebGI and three come from the CDN bundle at runtime; these four plugins are the demo's own forks of it.
const aliases = new Map([
  ['webgi/plugins/DlssBridgePlugin', source('webgi-addon', 'DlssBridgePlugin.ts')],
  ['webgi/plugins/VelocityBufferPlugin', source('webgi-addon', 'DlssVelocityBufferPlugin.ts')],
  ['webgi/plugins/shaders/ssVelocityVert.glsl', source('webgi-addon', 'shaders', 'ssVelocityVert.glsl')],
  ['webgi/plugins/shaders/ssVelocityFrag.glsl', source('webgi-addon', 'shaders', 'ssVelocityFrag.glsl')],
  ['three/examples/jsm/postprocessing/Pass', source('webgi-addon', 'passes', 'Pass.ts')],
  ['three/examples/jsm/postprocessing/RenderPass', source('webgi-addon', 'passes', 'RenderPass.ts')],
  ['three/examples/jsm/shaders/CopyShader', source('webgi-addon', 'passes', 'CopyShader.ts')],
  ['webgi/helpers/mathutil', source('webgi-addon', 'mathutil.ts')],
  ['webgi/plugins/GroundPlugin', source('webgi-addon', 'DlssGroundPlugin.ts')],
  ['webgi/helpers/threejs/Reflector2', source('webgi-addon', 'DlssReflector2.ts')],
  ['webgi/plugins/shaders/reflectorSample.glsl', source('webgi-addon', 'shaders', 'reflectorSample.glsl')],
  ['webgi/plugins/shaders/randomHelpers.glsl', source('webgi-addon', 'shaders', 'randomHelpers.glsl')],
  ['webgi/plugins/shaders/poissonDiskSamples.glsl', source('webgi-addon', 'shaders', 'poissonDiskSamples.glsl')],
]);

const cdnGlobals = {
  name: 'webgi-cdn-globals',
  setup(builder) {
    builder.onResolve({ filter: /.*/ }, (args) => {
      const exact = aliases.get(args.path);
      if (exact) return { path: exact };
      if (args.path === 'three' || args.path === 'ts-browser-helpers' ||
          args.path.startsWith('webgi/') || args.path.startsWith('three/examples/jsm/')) {
        return { path: source('webgi-global.ts') };
      }
      return null;
    });
  },
};

// demo-ui.js and nr-controls.js import the inspector and the settings module by the absolute paths the
// server serves them at; map those here rather than rewriting the imports.
//
// The inspector is three.js's own (`examples/jsm/inspector/`), taken from the installed package rather than
// copied in. Two of its imports need redirecting to use it on its own:
//
//   * it reads and writes panel state through `Inspector.js`, which also pulls in the seven tabs this demo
//     does not show. inspector-storage.js is the same two functions against this demo's own storage key.
//   * it imports EventDispatcher from `three`, which this build aliases to the viewer's global bundle so
//     that a second copy of three.js is not shipped. The class itself comes from the package.
const threePackage = path.join(root, '..', 'node_modules', 'three');
const inspector = path.join(threePackage, 'examples', 'jsm', 'inspector');

const browserPaths = {
  name: 'browser-paths',
  setup(builder) {
    builder.onResolve({ filter: /^\/three\// }, (args) => ({
      path: path.join(threePackage, args.path.slice('/three/'.length)),
    }));
    builder.onResolve({ filter: /^\.\.\/Inspector\.js$/ }, (args) => (
      args.importer.startsWith(inspector) ? { path: source('runtime', 'inspector-storage.js') } : null
    ));
    builder.onResolve({ filter: /^three$/ }, (args) => (
      args.importer.startsWith(inspector)
        ? { path: path.join(threePackage, 'src', 'core', 'EventDispatcher.js') } : null
    ));
    builder.onResolve({ filter: /^\/backend\// }, (args) => ({
      path: source('backend', args.path.slice('/backend/'.length)),
    }));
    builder.onResolve({ filter: /^\/app\// }, (args) => ({
      path: source('runtime', args.path.slice('/app/'.length)),
    }));
  },
};

const common = {
  absWorkingDir: root,
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2022'],
  minify: production,
  sourcemap: !production,
  treeShaking: true,
  logLevel: 'info',
};

fs.mkdirSync(path.join(root, 'dist'), { recursive: true });

await build({
  ...common,
  entryPoints: [source('main.ts')],
  outfile: path.join(root, 'dist', 'viewer.js'),
  loader: { '.glsl': 'text' },
  plugins: [cdnGlobals],
  tsconfig: path.join(root, 'tsconfig.json'),
});

// Bundled as a module because the port's sources resolve their shaders against import.meta.
await build({
  ...common,
  format: 'esm',
  entryPoints: [source('runtime', 'viewer-runtime.js')],
  outfile: path.join(root, 'dist', 'runtime.js'),
  plugins: [browserPaths],
});

await build({
  ...common,
  entryPoints: [source('runtime', 'loading-state.js')],
  outfile: path.join(root, 'dist', 'loading-state.js'),
});

console.log('built demo/dist');
