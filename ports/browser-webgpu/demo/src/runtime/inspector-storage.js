// App-specific storage adapter for the upstream Inspector UI.
const key = 'ijewel-dlss-inspector';
export function getItem(id) {
  try { return JSON.parse(localStorage.getItem(key) || '{}')[id] || {}; }
  catch { return {}; }
}
export function setItem(id, value) {
  try {
    const data = JSON.parse(localStorage.getItem(key) || '{}');
    data[id] = value;
    localStorage.setItem(key, JSON.stringify(data));
  } catch { /* The controls also work when storage is unavailable. */ }
}
