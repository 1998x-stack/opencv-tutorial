/* v0.3: paired-image geometry, flow, and deterministic association visualisation. */
(() => {
  const el = id => document.getElementById(id);
  const state = { pairs: {}, images: [], result: null, kind: 'panorama',
    tracking: null, frame: 0, busy: false };
  const message = text => { el('intermediateStatus').textContent = text; };

  function showPair() {
    const pair = state.pairs[state.kind] || [];
    state.images = [...pair];
    el('pairFirst').src = pair[0] || '';
    el('pairSecond').src = pair[1] || '';
    el('intermediateOutput').removeAttribute('src');
    el('intermediatePreview').removeAttribute('src');
    el('intermediateMetrics').textContent = 'Choose Run to inspect real results.';
    el('pairRun').textContent = state.kind === 'panorama' ? 'Stitch panorama →' : 'Estimate optical flow →';
    message('Synthetic inputs loaded. You can replace them with local images.');
  }

  async function execute() {
    if (state.busy || state.images.length !== 2) return;
    state.busy = true;
    el('pairRun').disabled = true;
    message('Running ' + state.kind + '…');
    try {
      const res = await fetch('/api/intermediate/' + state.kind, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({first: state.images[0], second: state.images[1]})
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      state.result = data;
      el('intermediateOutput').src = data.image;
      el('intermediatePreview').src = data.preview || '';
      el('intermediateMetrics').textContent = JSON.stringify(data.metrics, null, 2);
      message('Computed from the two input images. Failures are rejected rather than hidden.');
    } catch (error) {
      message('Cannot compute: ' + error.message);
      el('intermediateMetrics').textContent = 'No validated result.';
      el('intermediateOutput').removeAttribute('src');
    } finally {
      state.busy = false;
      el('pairRun').disabled = false;
    }
  }

  async function loadUpload(input, index) {
    const file = input.files && input.files[0];
    if (!file) return;
    if (file.size > 4 * 1024 * 1024 || !file.type.startsWith('image/')) {
      message('Choose an image under 4 MB.');
      return;
    }
    const url = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    state.images[index] = url;
    el(index ? 'pairSecond' : 'pairFirst').src = url;
    message('Image updated; click Run to recompute.');
  }

  function renderTrack() {
    if (!state.tracking) return;
    el('trackingImage').src = state.tracking.frames[state.frame];
    const event = state.tracking.events[state.frame];
    el('trackingLabel').textContent = `Frame ${state.frame + 1} / ${state.tracking.frames.length}`;
    el('trackingEvents').textContent = JSON.stringify(event.tracks, null, 2);
  }

  async function main() {
    try {
      const res = await fetch('/api/intermediate/sample');
      if (!res.ok) throw new Error('Sample endpoint unavailable');
      state.pairs = await res.json();
      showPair();
    } catch (error) { message(error.message); }
    el('pairMode').addEventListener('change', event => {
      state.kind = event.target.value;
      showPair();
    });
    el('pairRun').addEventListener('click', execute);
    el('pairInputFirst').addEventListener('change', event => loadUpload(event.target, 0));
    el('pairInputSecond').addEventListener('change', event => loadUpload(event.target, 1));
    el('pairReset').addEventListener('click', showPair);
    el('trackingLoad').addEventListener('click', async () => {
      try {
        const res = await fetch('/api/intermediate/tracking-demo');
        if (!res.ok) throw new Error('Cannot load tracking demo');
        state.tracking = await res.json();
        state.frame = 0;
        el('trackingFrame').max = state.tracking.frames.length - 1;
        el('trackingFrame').value = 0;
        renderTrack();
      } catch (error) { el('trackingEvents').textContent = error.message; }
    });
    el('trackingFrame').addEventListener('input', event => {
      state.frame = Number(event.target.value);
      renderTrack();
    });
  }
  main();
})();
