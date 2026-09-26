const ASSET_PATH = /^\/(?:scenes|environments|generated)(?:\/|$)/;

export function assetUrl(path) {
  const value = String(path);
  const base = String(globalThis.__DLSS5_ASSET_BASE__ ?? '').replace(/\/$/, '');
  return base && ASSET_PATH.test(value) ? `${base}${value}` : value;
}
