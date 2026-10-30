const $ = id => document.getElementById(id);
const state = { algorithm: 'gaussian', image: null, busy: false, sequence: 0,
  stages: [], stageIndex: 0, playTimer: null, studyToken: 0, studies: {} };
const sequence = ['gaussian', 'median', 'sobel', 'canny', 'otsu', 'adaptive', 'morphology', 'contours'];

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function setStatus(message, isError = false) {
  $('status').textContent = message;
  $('status').style.color = isError ? '#ff9b9b' : '#98a8c5';
}
function params() {
  return { algorithm: state.algorithm, kernel: Number($('kernel').value),
    sigma: Number($('sigma').value) / 10, threshold: Number($('threshold').value) };
}
function updateLabels() {
  $('kernelValue').textContent = `${$('kernel').value} × ${$('kernel').value}`;
  $('sigmaValue').textContent = (Number($('sigma').value) / 10).toFixed(1);
  $('thresholdValue').textContent = $('threshold').value;
}

function renderAlgorithms(catalog) {
  const root = $('algorithmList'); root.replaceChildren();
  sequence.forEach((name, index) => {
    const meta = catalog[name]; if (!meta) return;
    const btn = document.createElement('button');
    btn.className = `algorithm-item ${name === state.algorithm ? 'active' : ''}`;
    btn.setAttribute('aria-pressed', String(name === state.algorithm));
    const number = document.createElement('span'); number.className = 'number';
    number.textContent = String(index + 1).padStart(2, '0');
    const label = document.createElement('span'); label.textContent = meta.title;
    btn.append(number, label);
    btn.title = meta.description;
    btn.addEventListener('click', () => {
      state.algorithm = name;
      root.querySelectorAll('button').forEach(element => {
        const selected = element === btn;
        element.classList.toggle('active', selected);
        element.setAttribute('aria-pressed', String(selected));
      });
      document.querySelector('.sidebar-counter').textContent = `${String(index + 1).padStart(2, '0')} / 08`;
      stopPlayback(); updateStudyOptions(); runExperiment();
    });
    root.appendChild(btn);
  });
}

const explanations = {
  original: 'The unmodified 8-bit BGR source image.',
  grayscale: 'Convert 3 BGR channels into a single luminance-like intensity channel.',
  gaussian_smoothed: 'Convolve the input using normalized Gaussian weights; lower spatial frequencies remain.',
  smoothed: 'Gaussian smoothing suppresses noise before Canny gradient analysis.',
  horizontal_gradient: 'Sobel X: local horizontal intensity change.',
  vertical_gradient: 'Sobel Y: local vertical intensity change.',
  gradient_magnitude: 'Combine horizontal and vertical gradients into an edge-strength map.',
  histogram_equalized: 'CDF histogram transform, visualized as a diagnostic (not an Otsu preprocessing step).',
  binary: 'Threshold the grayscale source into foreground and background.',
  opening: 'Erode then dilate, reducing small foreground noise.',
  reference_rect_opening: 'Separate NumPy reference with a rectangular element (not the ellipse used by the output).',
  reference_mean_threshold: 'Separate local mean threshold diagnostic (output uses Gaussian-weighted neighborhoods).',
  result: 'Final result computed by the selected OpenCV algorithm.'
};
function stopPlayback() {
  if (state.playTimer !== null) { clearInterval(state.playTimer); state.playTimer = null; }
  $('playButton').textContent = '▶ Play'; $('playButton').setAttribute('aria-pressed', 'false');
}
function selectStage(index) {
  if (!state.stages.length) return;
  state.stageIndex = Math.max(0, Math.min(state.stages.length - 1, index));
  const [key, image] = state.stages[state.stageIndex];
  $('processedImage').src = image;
  $('stageSlider').value = String(state.stageIndex);
  $('playStageLabel').textContent = `${state.stageIndex + 1} / ${state.stages.length} · ${key.replaceAll('_',' ')}`;
  $('stageExplanation').textContent = explanations[key] || `Actual intermediate result: ${key}.`;
  document.querySelectorAll('.stage').forEach((el, i) => {
    el.classList.toggle('selected-stage', i === state.stageIndex);
    el.setAttribute('aria-pressed', String(i === state.stageIndex));
  });
}
function renderStages(images) {
  stopPlayback();
  state.stages = Object.entries(images);
  $('stageSlider').max = String(state.stages.length - 1);
  const root = $('stages'); root.replaceChildren();
  const entries = Object.entries(images);
  for (const [name, src] of entries) {
    const item = document.createElement('button'); item.type = 'button'; item.className = 'stage';
    const image = document.createElement('img'); image.src = src; image.alt = `${name} processing stage`;
    const label = document.createElement('span'); label.textContent = name.replaceAll('_', ' ').toUpperCase();
    item.append(image, label); root.append(item);
    item.addEventListener('click', () => { stopPlayback(); selectStage(state.stages.findIndex(pair => pair[0] === name)); });
  }
  $('stageCount').textContent = `${entries.length} STAGES · CLICK TO INSPECT`;
  selectStage(entries.length - 1);
}

function updateStudyOptions() {
  const supported = state.studies[state.algorithm] || {};
  const select = $('studySelect'); select.replaceChildren();
  for (const key of Object.keys(supported)) {
    const option = document.createElement('option'); option.value = key; option.textContent = key;
    select.append(option);
  }
  $('studyButton').disabled = !Object.keys(supported).length;
  $('studyStatus').textContent = Object.keys(supported).length ?
    'Four controlled parameter variants; all reuse one noisy image.' :
    'This algorithm has no supported parameter sweep in v0.2.';
  $('studyResults').replaceChildren();
}
async function runAblation() {
  if (!state.image || $('studyButton').disabled) return;
  const token = ++state.studyToken;
  $('studyButton').disabled = true;
  $('studyStatus').textContent = 'Evaluating controlled variants…';
  try {
    const response = await fetch('/api/ablate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({image: state.image, ...params(), study: $('studySelect').value,
        noise: $('noiseSelect').value, strength: Number($('noiseStrength').value), seed: 7})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (token !== state.studyToken) return;
    const root = $('studyResults'); root.replaceChildren();
    for (const candidate of data.candidates) {
      const article = document.createElement('article'); article.className = 'candidate';
      const image = document.createElement('img'); image.src = candidate.image;
      image.alt = `${data.study} value ${candidate.value}`;
      const title = document.createElement('strong'); title.textContent = `${data.study} = ${candidate.value}`;
      const details = document.createElement('span');
      const quality = candidate.quality;
      details.textContent = quality.quality_available ?
        `PSNR ${quality.after.psnr_db ?? '∞'} dB · SSIM ${quality.after.ssim}` :
        'No reference quality score for this operation';
      article.append(image, title, details); root.append(article);
    }
    const first = data.candidates[0].quality;
    $('studyStatus').textContent = `${data.study_resolution.join('×')} px · ${data.noise} noise` +
      (first.quality_available ? ` · Noisy baseline PSNR ${first.before.psnr_db} dB` :
        ' · Quality unavailable without a meaningful ground truth') +
      ' · Runtime is not a benchmark.';
  } catch (error) {
    if (token === state.studyToken) $('studyStatus').textContent = `Ablation failed: ${error.message}`;
  } finally {
    if (token === state.studyToken) updateStudyOptionsAfterRun();
  }
}
function updateStudyOptionsAfterRun() {
  $('studyButton').disabled = !Object.keys(state.studies[state.algorithm] || {}).length;
}

async function runExperiment() {
  if (!state.image) return;
  const thisRun = ++state.sequence;
  stopPlayback(); state.studyToken++; updateStudyOptionsAfterRun();
  state.busy = true; $('runButton').disabled = true;
  setStatus(`Processing ${state.algorithm}…`);
  try {
    const response = await fetch('/api/process', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: state.image, ...params() })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (thisRun !== state.sequence) return;
    $('processedImage').src = data.images.result;
    $('originalImage').src = data.images.original;
    $('emptyState').classList.add('hidden');
    renderStages(data.images);
    const m = data.metrics;
    $('executionTime').textContent = m.elapsed_ms.toFixed(2);
    $('resolution').textContent = `${m.width}×${m.height}`;
    $('imageDimensions').textContent = `${m.width} × ${m.height} px`;
    const ref = m.reference_mae ?? m.gradient_max_error ?? m.reference_pixel_difference ?? m.selected_threshold;
    $('referenceMetric').textContent = ref === undefined ? 'N/A' : String(ref);
    $('referenceLabel').textContent = m.reference_mae !== undefined ? 'mean abs. error' :
      m.gradient_max_error !== undefined ? 'maximum gradient error' :
      m.reference_pixel_difference !== undefined ? 'pixel disagreements' :
      m.selected_threshold !== undefined ? 'selected threshold' : 'not applicable';
    setStatus(`Finished · ${m.algorithm}`);
  } catch (error) {
    if (thisRun === state.sequence) setStatus(`Processing failed: ${error.message}`, true);
  } finally {
    if (thisRun === state.sequence) { state.busy = false; $('runButton').disabled = false; }
  }
}

async function useSample() {
  setStatus('Loading synthetic sample…');
  try {
    const data = await getJSON('/api/sample');
    state.image = data.image; await runExperiment();
  } catch (error) { setStatus(`Unable to load sample: ${error.message}`, true); }
}

function renderCurriculum(weeks, level) {
  const root = $('weekGrid'); root.replaceChildren();
  weeks.filter(w => level === 'All' || w.level === level).forEach(week => {
    const card = document.createElement('article'); card.className = 'week-card';
    const label = document.createElement('div'); label.className = 'week-label';
    const id = document.createElement('span'); id.textContent = week.id;
    const tier = document.createElement('span'); tier.textContent = week.level.toUpperCase();
    label.append(id, tier);
    const title = document.createElement('h3'); title.textContent = week.title;
    const body = document.createElement('p'); body.textContent = week.topics;
    const deliverable = document.createElement('div'); deliverable.className = 'week-deliverable';
    deliverable.textContent = `→ ${week.deliverable}`;
    card.append(label, title, body, deliverable); root.append(card);
  });
}

async function init() {
  updateLabels();
  for (const id of ['kernel', 'sigma', 'threshold']) $(id).addEventListener('input', updateLabels);
  for (const id of ['kernel', 'sigma', 'threshold']) $(id).addEventListener('change', runExperiment);
  $('runButton').addEventListener('click', runExperiment);
  $('playButton').addEventListener('click', () => {
    if (state.playTimer !== null) { stopPlayback(); return; }
    if (state.stages.length < 2) return;
    if (state.stageIndex === state.stages.length - 1) selectStage(0);
    $('playButton').textContent = 'Ⅱ Pause'; $('playButton').setAttribute('aria-pressed','true');
    state.playTimer = setInterval(() => {
      if (state.stageIndex >= state.stages.length - 1) { stopPlayback(); return; }
      selectStage(state.stageIndex + 1);
    }, 850);
  });
  $('prevButton').addEventListener('click', () => { stopPlayback(); selectStage(state.stageIndex - 1); });
  $('nextButton').addEventListener('click', () => { stopPlayback(); selectStage(state.stageIndex + 1); });
  $('stageSlider').addEventListener('input', event => { stopPlayback(); selectStage(Number(event.target.value)); });
  $('noiseStrength').addEventListener('input', event => $('strengthValue').textContent = event.target.value);
  $('studyButton').addEventListener('click', runAblation);
  $('sampleButton').addEventListener('click', useSample);
  $('compareSlider').addEventListener('input', event => {
    $('comparison').style.setProperty('--split', `${event.target.value}%`);
  });
  $('upload').addEventListener('change', event => {
    const file = event.target.files[0]; if (!file) return;
    if (!file.type.startsWith('image/')) { setStatus('Please select an image', true); return; }
    if (file.size > 5 * 1024 * 1024) { setStatus('Select an image smaller than 5 MB', true); return; }
    const reader = new FileReader();
    reader.onload = () => { state.image = reader.result; runExperiment(); };
    reader.onerror = () => setStatus('Could not read selected image', true);
    reader.readAsDataURL(file);
  });
  document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === tab));
    renderCurriculum(window.courseWeeks || [], tab.dataset.level);
  }));
  try {
    const [catalog, weeks, studies] = await Promise.all([getJSON('/api/catalog'), getJSON('/api/curriculum'), getJSON('/api/studies')]);
    state.studies = studies; updateStudyOptions();
    window.courseWeeks = weeks; renderAlgorithms(catalog); renderCurriculum(weeks, 'All');
    await useSample();
  } catch (error) { setStatus(`Laboratory initialization failed: ${error.message}`, true); }
}

document.addEventListener('DOMContentLoaded', init);
