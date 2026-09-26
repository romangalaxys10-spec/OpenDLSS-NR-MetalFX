// Bridge consumers run inside the composer, after the progressive pass but
// before BaseRenderer increments frameCount: use isConverged(false) here.
export function viewerConverged(viewer) {
  // A transition may still contain the previous camera/scene, even with AA off.
  if (viewer?.getPlugin('FrameFade')?.dirty) return false;
  const progressive = viewer?.getPlugin('Progressive');
  return !progressive?.enabled || progressive.isConverged(false);
}
