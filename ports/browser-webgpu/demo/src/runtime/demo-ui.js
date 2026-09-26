import {Profiler} from '/three/examples/jsm/inspector/ui/Profiler.js';
import {Parameters} from '/three/examples/jsm/inspector/tabs/Parameters.js';

// Actual three.js Inspector UI, driven by our existing WebGI / WebGPU runtime.
const query = new URLSearchParams(location.search);
const sr = query.get('sr') === '1';
const chain = sr && query.get('srChain') === '1';
if (query.has('scene')) {
  const url = new URL(location.href);
  url.searchParams.delete('scene');
  history.replaceState(history.state, '', url);
}
const profiler = new Profiler(null);
const parameters = new Parameters({builtin: true, icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 6h8m4 0h4M4 12h2m4 0h10M4 18h10m4 0h2"/><circle cx="14" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="16" cy="18" r="2"/></svg>'});
const performanceTab = new Parameters({name: 'Performance', allowDetach: false});
profiler.addTab(parameters);
profiler.addTab(performanceTab);
profiler.setActiveTab(performanceTab.id);
document.body.append(profiler.domElement);
const narrowScreen = matchMedia('(max-width: 700px)');
if (!narrowScreen.matches) profiler.show(parameters);
narrowScreen.addEventListener('change', event => { if (event.matches) profiler.hide(); });
profiler.toggleButton.title = 'Rendered frames per second · Open performance';
profiler.toggleButton.setAttribute('aria-label', 'Open performance');
parameters.builtinButton.setAttribute('aria-label', 'Toggle settings');
const fpsText = profiler.toggleButton.querySelector('.fps-counter');
const presentedIndicator = document.querySelector('#dlssPresentedIndicator');
function setPresented(active) {
  const presented = !!active;
  presentedIndicator.dataset.state = presented ? 'on' : 'off';
  presentedIndicator.querySelector('strong').textContent = `DLSS 5 ${presented ? 'ON' : 'OFF'}`;
  presentedIndicator.setAttribute('aria-label', `DLSS 5 is ${presented ? '' : 'not '}applied to the presented frame`);
  presentedIndicator.setAttribute('aria-pressed', String(presented));
  document.body.dataset.dlssPresented = String(presented);
}

// The upstream setter dispatches change events; cancel its deferred user
// callback when reflecting state to avoid triggering another render or action.
export function reflect(editor, value) {
  if (editor.getValue() === value) return;
  editor.setValue(value);
  clearTimeout(editor._changeTimeout);
}
export function enableControl(editor, enabled) {
  editor.domElement.dataset.disabled = String(!enabled);
  const row = editor.domElement.closest('.list-item-row');
  if (row) row.inert = !enabled;
  editor.domElement.querySelectorAll('input,select,button').forEach(input => { input.disabled = !enabled; });
}
export function labelControl(editor, label, id) {
  editor.name(label);
  const inputs = editor.domElement.querySelectorAll('input,select,button');
  inputs.forEach(input => input.setAttribute('aria-label', label));
  if (id && inputs[0]) inputs[0].id = id;
  return editor;
}

let runtime = null;
let actions = null;
let loadingScene = false;
let localEnvironmentEditor = null;
const frameTimes = {dlss: [], source: []};
let trackedViewer = null;
presentedIndicator.addEventListener('click', () => actions?.compareShortcut?.());
const sourceFrame = () => { if (!showingDlss()) recordFrame('source'); };
function trackViewer(viewer) {
  if (viewer === trackedViewer) return;
  trackedViewer?.removeEventListener('postRender', sourceFrame);
  trackedViewer = viewer;
  trackedViewer?.addEventListener('postRender', sourceFrame);
}
function recordFrame(kind) {
  const times = frameTimes[kind], now = performance.now();
  times.push(now);
  // Retain two samples even below 1 FPS; the old 1.5s cutoff lost slow frames.
  while (times.length > 2 && times[0] < now - 1500) times.shift();
}
function showingDlss() { return sr ? runtime?.visible : runtime?.neuralVisible; }
const controls = {scene: 'selection-five', mode: sr ? (chain ? 'srnr' : 'sr') : 'nr',
  live: false, output: 'dlss', render: () => actions?.render()};
const demo = parameters.createGroup('Demo');
const scenes = {'Cowboy Gramps': 'selection-five'};
const sceneControl = labelControl(demo.add(controls, 'scene', scenes), 'scene', 'demoScene').onChange(async value => {
  if (loadingScene || !globalThis.dlssChangeScene) return;
  if (value === 'local-file') {
    globalThis.dlssPromptForFile?.();
    reflect(sceneControl, globalThis.dlssSceneId || 'selection-five');
    return;
  }
  loadingScene = true;
  syncRuntime();
  try { await globalThis.dlssChangeScene(value); }
  catch (error) {
    globalThis.dlssLoading?.fail(error);
    const status = document.querySelector('#dlssWebGpuStatus');
    status.dataset.state = 'error'; status.textContent = `Scene loading failed: ${error.message}`;
    reflect(sceneControl, globalThis.dlssSceneId || 'selection-five');
  } finally { loadingScene = false; syncRuntime(); }
});
const mode = labelControl(demo.add(controls, 'mode', {'Neural rendering': 'nr'}), 'mode', 'dlssMode').onChange(value => {
  const url = new URL(location.href);
  if (value === 'nr') url.searchParams.delete('sr'); else url.searchParams.set('sr', '1');
  if (value === 'srnr') url.searchParams.set('srChain', '1'); else url.searchParams.delete('srChain');
  // Retain the selected scene across a runtime-mode reload without URL routing.
  try { sessionStorage.setItem('dlss-demo-scene', globalThis.dlssSceneId || 'selection-five'); } catch {}
  location.assign(url);
});
const modeRow = mode.domElement.closest('.list-item-wrapper') || mode.domElement;
modeRow.style.setProperty('display', 'none', 'important');
const live = labelControl(demo.add(controls, 'live'), 'live · L', 'nrLive').onChange(value => {
  if (runtime && value !== runtime.live) actions.toggleLive();
});
const output = labelControl(demo.add(controls, 'output', {DLSS: 'dlss', Original: 'source'}), 'output · F6', 'nrCompare').onChange(value => {
  if (runtime && (value === 'source') !== isComparing()) actions.compare();
});
const render = labelControl(demo.add(controls, 'render'), 'Render · R', 'nrRender');
const stats = {resolution: '—', readback: '—', preprocessing: '—', network: '—', presentation: '—'};
const metrics = performanceTab.createGroup('Production timings');
const statEditors = new Map();
for (const [key, label] of Object.entries({resolution: 'output', readback: 'WebGL readback', preprocessing: 'upload / preprocessing', network: 'network execution', presentation: 'presentation'})) {
  const editor = labelControl(metrics.add(stats, key), label);
  editor.input.readOnly = true;
  statEditors.set(key, editor);
}
const statusText = document.createElement('p');
statusText.className = 'demo-status';
statusText.textContent = 'FPS counts new frames in the visible output: DLSS completions or WebGI renders. 0 means paused. Timings update after each DLSS render.';
performanceTab.content.append(statusText);

function isComparing() { return sr ? runtime?.compare : runtime?.compareSource; }
function syncRuntime() {
  const enabled = !!runtime && !loadingScene && globalThis.dlssSceneReady && (sr || runtime.settings.enabled);
  enableControl(sceneControl, !!globalThis.dlssChangeScene && !loadingScene && !globalThis.dlssSceneLoading);
  enableControl(live, enabled);
  enableControl(output, enabled && !!runtime?.result);
  enableControl(render, enabled && !runtime?.running);
  reflect(live, !!runtime?.live);
  reflect(output, isComparing() ? 'source' : 'dlss');
  if (localEnvironmentEditor) {
    const localScene = globalThis.dlssSceneId === 'local-file';
    const row = localEnvironmentEditor.domElement.closest('.list-item-wrapper') || localEnvironmentEditor.domElement;
    // Inspector gives its rows an explicit display mode, which overrides the
    // browser's generic [hidden] rule. Hide the complete parameter wrapper.
    row.style.setProperty('display', localScene ? '' : 'none', localScene ? '' : 'important');
    const embedded = localEnvironmentEditor.domElement.querySelector('option[value="embedded"]');
    if (embedded) embedded.disabled = !globalThis.dlssEmbeddedEnvironment;
    if (localScene) reflect(localEnvironmentEditor, globalThis.dlssLocalEnvironment || 'studio-small-08');
  }
  const now = performance.now();
  const neural = showingDlss(), times = frameTimes[neural ? 'dlss' : 'source'];
  let active = !document.hidden && !globalThis.dlssSceneLoading &&
    (neural ? runtime?.live || runtime?.running : runtime?.viewer?.renderEnabled);
  const elapsed = times.length > 1 ? (times.at(-1) - times[0]) / (times.length - 1) : 0;
  if (!neural && now - times.at(-1) > Math.max(1500, elapsed * 2)) active = false;
  const fps = active && elapsed > 0 ? 1000 / Math.max(elapsed, now - times.at(-1)) : 0;
  fpsText.textContent = fps > 0 && fps < 10 ? fps.toFixed(1) : fps.toFixed(0);
  profiler.toggleButton.title = `${neural ? 'DLSS completed' : 'WebGI rendered'} frames per second${active ? '' : ' · Paused'} · Open performance`;
  if (!active) times.length = 0;
  profiler.toggleGraph.addPoint('fps', fps);
  profiler.toggleGraph.update();
}

export const demoUi = {
  profiler,
  setPresented,
  mountFileImport(promptForFile) {
    if (document.querySelector('#openLocalModel')) return;
    const files = parameters.createGroup('Local 3D model');
    const local = {open: () => promptForFile?.(), environment: globalThis.dlssLocalEnvironment || 'studio-small-08'};
    labelControl(files.add(local, 'open'), 'Load your 3D file', 'openLocalModel')
      .info('GLB, GLTF, DRC, OBJ/MTL, FBX, STL, Rhino 3DM and ZIP. Select dependent files together. You can also drag files or a folder onto the viewer.');
    localEnvironmentEditor = labelControl(files.add(local, 'environment', globalThis.dlssLocalEnvironments),
      'HDR environment', 'localEnvironment').onChange(async id => {
      if (id === globalThis.dlssLocalEnvironment) return;
      try { await globalThis.dlssSetLocalEnvironment?.(id); }
      catch { reflect(localEnvironmentEditor, globalThis.dlssLocalEnvironment || 'studio-small-08'); }
    });
    syncRuntime();
  },
  createParameters(name) { return parameters.createGroup(name); },
  mountRuntime(state, callbacks) { runtime = state; actions = callbacks; syncRuntime(); },
  async beforeSceneChange() {
    await actions?.beforeSceneChange?.();
    setPresented(false);
    frameTimes.dlss.length = frameTimes.source.length = 0;
    delete document.body.dataset.dlssReady;
    for (const editor of statEditors.values()) reflect(editor, '—');
  },
  afterSceneChange() { actions?.afterSceneChange?.(); },
  sync: syncRuntime,
  frameComplete(timings, width, height) {
    document.body.dataset.dlssReady = 'true';
    recordFrame('dlss');
    if (!globalThis.dlssSceneLoading) globalThis.dlssLoading?.finish();
    const ms = value => Number.isFinite(value) ? `${value.toFixed(1)} ms` : 'unavailable';
    Object.assign(stats, {resolution: `${width} × ${height}`, readback: ms(timings.readbackMilliseconds),
      preprocessing: ms(timings.uploadPreprocessMilliseconds), network: ms(timings.networkMilliseconds),
      presentation: ms(timings.presentationMilliseconds)});
    for (const [key, editor] of statEditors) reflect(editor, stats[key]);
  },
  mountSr(initialSize, setResolution, reset) {
    const settings = {width: initialSize[0], height: initialSize[1],
      apply() {
        try { setResolution(Number(settings.width), Number(settings.height)); sizeHint.textContent = 'Performance · preset J · 2×'; }
        catch (error) { sizeHint.textContent = error.message; }
      }, reset};
    const group = parameters.createGroup('Super resolution');
    const width = labelControl(group.add(settings, 'width', 1, 4096, 1), 'input width', 'srWidth');
    const height = labelControl(group.add(settings, 'height', 1, 4096, 1), 'input height', 'srHeight');
    labelControl(group.add(settings, 'apply'), 'Apply resolution');
    labelControl(group.add(settings, 'reset'), 'Reset temporal history', 'srReset');
    const sizeHint = document.createElement('p');
    sizeHint.className = 'demo-status'; sizeHint.textContent = 'Performance · preset J · 2×';
    group.paramList.domElement.append(sizeHint);
    return {sync(w, h) { reflect(width, w); reflect(height, h); }};
  },
};
globalThis.dlssDemoUi = demoUi;
const timer = setInterval(syncRuntime, 250);
addEventListener('dlss-viewer-ready', event => trackViewer(event.detail));
if (globalThis.dlssViewer) trackViewer(globalThis.dlssViewer);
addEventListener('pagehide', () => { clearInterval(timer); trackViewer(null); profiler.dispose(); }, {once: true});

const attributions = {
  'simple-lighting': ['Simple Lighting', 'https://www.blendkit.com/asset-gallery-detail/2d3edff0-47f6-4bd6-9d1b-cbde69378c65/', ' · Ryder Booth · Mustang by AIR3D · BlenderKit'],
  'vege-packshot': ['Vege packshot', 'https://www.blendkit.com/asset-gallery-detail/ed54839b-fc24-4651-8fae-da3ac8f547a0/', ' · Bart Papis · BlenderKit'],
  arunthayan: ['Arunthayan', 'https://www.blendkit.com/asset-gallery-detail/7d65df92-91fc-47ad-b967-086378a87707/', ' · Muhammed Ismayil · BlenderKit'],
  'selection-five': ['Cowboy Gramps', 'https://www.blendkit.com/asset-gallery-detail/96dce188-9c9c-4699-a45a-48663fbbbcb7/', ' · Muhammed Ismayil · CC0'],
  'selection-six': ['Village in the Highlands', 'https://www.blendkit.com/asset-gallery-detail/bf6f87e6-71f6-4443-9073-4e005a05d293/', ' · Ibrohim Toxirov · BlenderKit'],
  bistro: ['Amazon Lumberyard Bistro', 'https://developer.nvidia.com/orca/amazon-lumberyard-bistro', ' · Amazon Lumberyard · CC BY 4.0'],
  'lone-monk': ['Lone Monk', 'https://blenderartists.org/t/lone-monk-cc0-scene-and-assets/1287621', ' · Carlo Bergonzini / Monorender · CC0'],
};
function updateScene() {
  const id = globalThis.dlssSceneId || 'selection-five';
  reflect(sceneControl, id);
  const attribution = document.querySelector('#sceneAttribution');
  attribution.replaceChildren();
  if (attributions[id]) {
    const [title, href, credit] = attributions[id];
    const link = document.createElement('a');
    link.textContent = title; link.href = href; link.target = '_blank'; link.rel = 'noopener';
    attribution.append(link, credit);
  }
}
addEventListener('dlss-scene-ready', updateScene);
addEventListener('dlss-local-environment-changed', event => {
  if (localEnvironmentEditor) reflect(localEnvironmentEditor, event.detail.id);
  syncRuntime();
});
updateScene();
syncRuntime();
