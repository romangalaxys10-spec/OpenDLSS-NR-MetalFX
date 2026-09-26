export function updateBit(value: number, bit: number, enabled: number | boolean) {
  const mask = 1 << bit;
  return enabled ? value | mask : value & ~mask;
}
