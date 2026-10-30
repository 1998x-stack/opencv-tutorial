const $ = id => document.getElementById(id);
const state = { algorithm: 'gaussian', image: null, busy: false, sequence: 0 };
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
      runExperiment();
    });
    root.appendChild(btn);
  });
}

function renderStages(images) {
  const root = $('stages'); root.replaceChildren();
  const entries = Object.entries(images);
  for (const [name, src] of entries) {
    const item = document.createElement('div'); item.className = 'stage';
    const image = document.createElement('img'); image.src = src; image.alt = `${name} processing stage`;
    const label = document.createElement('span'); label.textContent = name.replaceAll('_', ' ').toUpperCase();
    item.append(image, label); root.append(item);
    item.addEventListener('click', () => { $('processedImage').src = src; });
    item.style.cursor = 'pointer';
  }
  $('stageCount').textContent = `${entries.length} STAGES · CLICK TO INSPECT`;
}

async function runExperiment() {
  if (!state.image) return;
  const thisRun = ++state.sequence;
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
    const [catalog, weeks] = await Promise.all([getJSON('/api/catalog'), getJSON('/api/curriculum')]);
    window.courseWeeks = weeks; renderAlgorithms(catalog); renderCurriculum(weeks, 'All');
    await useSample();
  } catch (error) { setStatus(`Laboratory initialization failed: ${error.message}`, true); }
}

document.addEventListener('DOMContentLoaded', init);
