import './demo-ui.js';

import('./webgpu-bridge.js').catch(error => {
  globalThis.dlssLoading?.fail(error);
  console.error(error);
  const status = document.querySelector('#dlssWebGpuStatus');
  status.dataset.state = 'error';
  status.textContent = `Unable to initialize DLSS: ${error.message}`;
});
