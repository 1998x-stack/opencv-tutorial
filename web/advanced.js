/* Advanced lab: true StereoSGBM and ground-truth synthetic PnP/relative VO. */
(() => {
  const el = id => document.getElementById(id);
  const state = { pair: [], busy: false };
  const status = msg => { el('advancedStatus').textContent = msg; };

  function renderPair() {
    el('advancedLeft').src = state.pair[0] || '';
    el('advancedRight').src = state.pair[1] || '';
    el('advancedOutput').removeAttribute('src');
    el('advancedMetrics').textContent = 'Run stereo depth to inspect computed metrics.';
  }

  async function sample() {
    status('Loading deterministic rectified sample…');
    try {
      const response = await fetch('/api/advanced/sample');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      state.pair = data.stereo;
      el('advancedFx').value = data.calibration.fx_px;
      el('advancedBaseline').value = data.calibration.baseline_m;
      el('advancedExpected').textContent = 'Synthetic fixture: true disparity 8 px, true depth 6.3 m. ' +
        'Black output pixels indicate invalid disparity.';
      renderPair();
      status('Sample loaded. Calibrated, rectified cameras are ASSUMED for external uploads.');
    } catch (err) { status(`Sample error: ${err.message}`); }
  }

  async function upload(input, index) {
    const file = input.files && input.files[0];
    if (!file) return;
    if (file.size > 4 * 1024 * 1024 || !file.type.startsWith('image/')) {
      status('Select a valid image smaller than 4MB');
      return;
    }
    const encoded = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    state.pair[index] = encoded;
    el(index ? 'advancedRight' : 'advancedLeft').src = encoded;
    el('advancedExpected').textContent = 'External calibration and rectification are NOT checked.';
    status('Image updated; recompute explicitly.');
  }

  async function compute() {
    if (state.busy || state.pair.length !== 2) return;
    state.busy = true;
    el('advancedRun').disabled = true;
    const mode = el('advancedMode').value;
    status(`Running ${mode.toUpperCase()}…`);
    try {
      const payload = { first: state.pair[0], second: state.pair[1],
        fx_px: Number(el('advancedFx').value), baseline_m: Number(el('advancedBaseline').value) };
      const response = await fetch('/api/advanced/' + mode, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      el('advancedOutput').src = data.image;
      el('advancedMetrics').textContent = JSON.stringify({ ...data.metrics,
        ...(mode === 'vo' ? { translation_unit_direction: data.unit_translation } : {}) }, null, 2);
      status(mode === 'vo' ? 'Relative pose only: no absolute scale or SLAM.' :
        'Estimated disparity and derived depth for assumed rectified stereo pair.');
    } catch (err) {
      status('Computation rejected: ' + err.message);
      el('advancedOutput').removeAttribute('src');
      el('advancedMetrics').textContent = 'No validated result.';
    } finally { state.busy = false; el('advancedRun').disabled = false; }
  }

  async function geometry() {
    el('geometryMetrics').textContent = 'Estimating known synthetic 3D correspondences…';
    try {
      const response = await fetch('/api/advanced/geometry-demo');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      el('geometryOutput').src = data.image;
      el('geometryMetrics').textContent = JSON.stringify({ PnP: data.pnp, relative_VO: data.vo }, null, 2);
    } catch (err) { el('geometryMetrics').textContent = 'Demo failed: ' + err.message; }
  }

  async function main() {
    el('advancedRun').addEventListener('click', compute);
    el('advancedReset').addEventListener('click', sample);
    el('advancedUploadLeft').addEventListener('change', event => upload(event.target, 0));
    el('advancedUploadRight').addEventListener('change', event => upload(event.target, 1));
    el('advancedMode').addEventListener('change', event => {
      const vo = event.target.value === 'vo';
      el('advancedRun').textContent = vo ? 'Estimate relative pose →' : 'Compute stereo depth →';
      el('advancedExpected').textContent = vo ?
        'Requires calibrated camera, static 3D scene and nonzero parallax. Synthetic stereo shift may be degenerate.' :
        'Depth is meaningful ONLY if images are rectified and fx/baseline match physical cameras.';
    });
    el('geometryLoad').addEventListener('click', geometry);
    await sample();
  }
  main();
})();
