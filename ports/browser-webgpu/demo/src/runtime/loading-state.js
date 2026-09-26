// Loaded before the viewer bundle, so slow scene/runtime downloads have a UI.
(() => {
  const panel = document.querySelector('#demoLoading');
  const title = document.querySelector('#loadingTitle');
  const detail = document.querySelector('#loadingDetail');
  const progress = document.querySelector('#loadingProgress');
  const transfer = document.querySelector('#loadingTransfer');
  const retry = document.querySelector('#loadingRetry');
  const mib = bytes => `${(bytes / 1048576).toFixed(1)} MB`;
  let active = true;
  let downloadInFlight = false;
  let finishPending = false;

  function hide() {
    finishPending = false;
    active = false;
    panel.hidden = true;
    document.querySelector('#canvasContainer').setAttribute('aria-busy', 'false');
  }

  function show(heading, description = '', download = null) {
    panel.hidden = false;
    panel.dataset.state = 'loading';
    document.querySelector('#canvasContainer').setAttribute('aria-busy', 'true');
    title.textContent = heading;
    detail.textContent = description;
    retry.hidden = true;
    progress.hidden = false;
    progress.removeAttribute('value');
    transfer.textContent = '';
    if (download) {
      const {loaded = 0, total = 0} = download;
      if (total > 0) {
        progress.value = Math.min(1, loaded / total);
        transfer.textContent = `${mib(loaded)} / ${mib(total)} · ${Math.min(100, Math.floor(loaded / total * 100))}%`;
      } else transfer.textContent = `${mib(loaded)} downloaded`;
    }
  }
  const loading = globalThis.dlssLoading = {
    begin(heading, description) {
      active = true;
      downloadInFlight = false;
      finishPending = false;
      show(heading, description);
    },
    stage(heading, description, download) {
      if (active && !downloadInFlight) show(heading, description, download);
    },
    runtime(message, error = false, download = null) {
      if (error) { this.fail(message); return; }
      // Scene loading owns the overlay until the imported scene is ready. This
      // prevents a previous in-flight neural frame from replacing or hiding it.
      if (document.body.dataset.sceneLoading === 'true') return;
      active = true;
      if (download) {
        const {loaded = 0, total = 0} = download;
        downloadInFlight = total <= 0 || loaded < total;
        show(`Downloading ${download.label}`, 'Downloading the neural model', download);
      } else {
        // Weight transfers and GPU pipeline creation intentionally overlap.
        // Keep one stable download view until the complete model has arrived,
        // rather than alternating these concurrent progress producers.
        if (downloadInFlight) return;
        if (/^Downloading\b/i.test(message)) {
          downloadInFlight = true;
          show('Downloading DLSS model', 'Starting the neural model download');
          return;
        }
        const compiling = /kernel|compil|neural rendering ·|super resolution ·/i.test(message);
        const count = compiling ? message.match(/(\d+)\/(\d+)/)?.[0] : null;
        show('Preparing DLSS for your GPU', compiling
          ? `Preparing GPU programs${count ? ` · ${count}` : ''}`
          : /requesting|adapter/i.test(message) ? 'Connecting to your GPU'
          : 'Loading neural models and preparing frame buffers');
      }
    },
    finish() {
      if (document.body.dataset.sceneLoading === 'true') {
        finishPending = true;
        return;
      }
      if (downloadInFlight) return;
      hide();
    },
    releaseScene() {
      if (finishPending && !downloadInFlight && document.body.dataset.sceneLoading !== 'true') hide();
    },
    fail(error) {
      active = false;
      downloadInFlight = false;
      finishPending = false;
      show('Unable to load the demo', error?.message || String(error));
      panel.dataset.state = 'error'; progress.hidden = true; retry.hidden = false;
      document.querySelector('#canvasContainer').setAttribute('aria-busy', 'false');
    },
  };
  retry.addEventListener('click', () => location.reload());
  addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement) loading.fail('A required script could not download. Check your connection and retry.');
  }, true);
})();
