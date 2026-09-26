import {NR_DEFAULTS, normalizeNrSettings} from '/backend/nr-settings.js';
import {demoUi, labelControl, enableControl, reflect} from './demo-ui.js';

const queryKeys = {enabled: 'nrEnabled', intensity: 'nrIntensity', localTone: 'nrTone',
  localStructure: 'nrStructure', skinStructure: 'nrSkin', autoMask: 'nrAutoMask',
  style: 'nrStyle', preset: 'nrPreset', uiCorrection: 'nrUiCorrection'};

export function readNrSettings() {
  const query = new URLSearchParams(location.search);
  let settings = {...NR_DEFAULTS};
  for (const [key, param] of Object.entries(queryKeys)) {
    if (!query.has(param)) continue;
    const text = query.get(param);
    if (text.trim() === '') continue;
    try { settings = normalizeNrSettings({[key]: Number(text)}, settings); }
    catch { /* Ignore malformed links; retain the documented default. */ }
  }
  return settings;
}

export function mountNrControls(initial, onChange) {
  const panel = demoUi.createParameters('Neural rendering');
  panel.paramList.domElement.id = 'nrSettings';
  let current = initial;
  // Keep the cowboy's approved skin structure at 1, independently of
  // general structure, and retain manual adjustments per scene.
  const defaultSkin = scene => scene === 'selection-five' ? 1 : NR_DEFAULTS.skinStructure;
  const skinByScene = new Map();
  const initialSkinOverride = new URLSearchParams(location.search).has('nrSkin') ? initial.skinStructure : undefined;
  let activeScene = null;
  const values = {...initial, skinAuto: initial.skinStructure < 0, reset: () => onChange({...NR_DEFAULTS})};
  const editors = new Map();
  function add(group, key, label, ...options) {
    const editor = labelControl(group.add(values, key, ...options), label);
    editor.onChange(value => {
      if (key === 'skinAuto') {
        const skinStructure = value ? -1 : current.localStructure;
        if (skinStructure !== current.skinStructure) onChange({skinStructure});
      } else if (value !== current[key]) onChange({[key]: value});
    }).debounce(typeof initial[key] === 'number' ? 80 : 0);
    editors.set(key, editor);
    return editor;
  }
  add(panel, 'enabled', 'enabled');
  add(panel, 'style', 'style', {Default: 0, Natural: 1, Cinematic: 2});
  add(panel, 'intensity', 'intensity', 0, 1, .01);
  add(panel, 'localTone', 'local tone', 0, 2, .01);
  add(panel, 'localStructure', 'local structure', 0, 2, .01);
  add(panel, 'skinStructure', 'skin structure', 0, 2, .01);
  add(panel, 'skinAuto', 'skin follows structure').info('Automatically follows local structure.');
  add(panel, 'autoMask', 'automatic masking').info('Apply skin and general structure separately.');
  const advanced = panel.addFolder('Additional options').close();
  add(advanced, 'preset', 'render preset', {Default: 0, 'Preset 1': 1, 'Preset 2': 2, 'Preset 3': 3})
    .info('These presets select the same model in the current implementation.');
  add(advanced, 'uiCorrection', 'UI correction').info('No effect on this scene-only input.');
  labelControl(panel.add(values, 'reset'), 'Reset NR settings', 'nrReset');
  function sync(settings) {
    current = settings;
    for (const [key, editor] of editors) {
      const value = key === 'skinAuto' ? settings.skinStructure < 0
        : key === 'skinStructure' && settings.skinStructure < 0 ? settings.localStructure : settings[key];
      reflect(editor, value);
    }
    enableControl(editors.get('skinStructure'), settings.skinStructure >= 0 && settings.autoMask);
    enableControl(editors.get('skinAuto'), settings.autoMask);
    const url = new URL(location.href);
    for (const [key, param] of Object.entries(queryKeys)) {
      const defaultValue = key === 'skinStructure' ? defaultSkin(activeScene) : NR_DEFAULTS[key];
      if (settings[key] === defaultValue) url.searchParams.delete(param);
      else url.searchParams.set(param, String(typeof settings[key] === 'boolean' ? Number(settings[key]) : settings[key]));
    }
    history.replaceState(history.state, '', url);
    demoUi.sync();
  }
  function applySceneSkin() {
    if (!globalThis.dlssSceneReady) return;
    const scene = globalThis.dlssSceneId;
    if (!scene || scene === activeScene) return;
    if (activeScene) skinByScene.set(activeScene, current.skinStructure);
    const skinStructure = skinByScene.get(scene) ??
      (activeScene === null && initialSkinOverride !== undefined ? initialSkinOverride : defaultSkin(scene));
    activeScene = scene;
    if (skinStructure !== current.skinStructure) onChange({skinStructure});
    else sync(current);
  }
  addEventListener('dlss-scene-ready', applySceneSkin);
  addEventListener('pagehide', () => removeEventListener('dlss-scene-ready', applySceneSkin), {once: true});
  sync(initial);
  // Let the runtime finish assigning its controls before invoking onChange.
  queueMicrotask(applySceneSkin);
  return {sync};
}
