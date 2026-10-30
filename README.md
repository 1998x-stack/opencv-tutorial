> **v0.3 intermediate observatory** — adds reproducible ORB/RANSAC panorama stitching, sparse LK flow and deterministic centroid tracking on top of the v0.2 beginner lab. This stacked branch depends on the v0.2 PR.

# Vision Lab · OpenCV from Scratch

A **24-week**, three-level computer-vision curriculum supported by executable NumPy reference algorithms, OpenCV comparisons, a self-contained interactive browser laboratory, CLI projects, testing, and performance benchmarks.

> **Status: v0.1 foundation.** The eight interactive classic algorithms, synthetic sample, API, 24-week curriculum navigator, document scanner, ORB matcher and motion detector are implemented. The high-level curriculum and advanced projects are a roadmap, **not** a claim that all 24 lessons or all eight projects are complete.

## Quickstart

Requires Python 3.10+, a supported OS/architecture and OpenCV 4.x.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m opencv_tutorial serve
# Open http://127.0.0.1:8765
```

The site serves **locally by default**, requires no web connectivity or cloud API, and includes a synthetic sample image. You can also upload a local image (processed locally on your own machine).

## CLI examples

```bash
python -m opencv_tutorial course
python -m opencv_tutorial sample --output results/sample.png
python -m opencv_tutorial process --input results/sample.png --output results/edges.png --algorithm sobel
python -m opencv_tutorial benchmark --rounds 30 --output results/benchmark.json
python -m opencv_tutorial scan --input document.jpg --output results/scanned.png
python -m opencv_tutorial motion --input input.mp4 --output results/motion.mp4
pytest -q
```

## Current modules

- **Algorithms:** explicitly implemented 2D correlation, Gaussian kernel/filter, Sobel gradients, Otsu segmentation; independently compared with OpenCV.
- **Pipeline:** Gaussian, median, Sobel, Canny, Otsu, adaptive threshold, morphology and contours; common parameter/metrics/intermediate-stage contract.
- **Geometry:** perspective transform and conservative quadrilateral document detection.
- **Features:** ORB features, Hamming matching, Lowe ratio, homography/RANSAC and inlier counts.
- **Video:** MOG2 background subtraction with shadow suppression, morphological cleaning, motion regions and video export.
- **Web lab:** interactive kernel/threshold controls, upload, before/after scrubber, intermediate-stage gallery, measured runtime, 24-week lesson navigator.
- **Quality:** deterministic inputs; unit, math parity and HTTP integration tests; GitHub Actions test matrix.

## 24-week roadmap

| Level | Weeks | Goal |
|---|---|---|
| Beginner | W01–W08 | Pixels, color, filtering, segmentation, contours, document scanning |
| Intermediate | W09–W16 | Features, matching, RANSAC, panorama, motion, optical flow, tracking |
| Advanced | W17–W24 | Multiview geometry, stereo depth, VO, DNN, C++, profiling, evaluation |

See [docs/architecture.md](docs/architecture.md), [docs/roadmap.md](docs/roadmap.md) and [docs/testing.md](docs/testing.md).

## Architecture

```text
opencv_tutorial/  # algorithms, geometry, features, motion, pipeline, CLI, HTTP server
web/              # native HTML/CSS/JS browser application
lessons/          # 24 lesson assignments and expected outputs
projects/         # project acceptance cards
benchmarks/       # benchmark runner notes
src/              # reserved for future optional C++ exercises
results/          # generated outputs; ignored by Git
```

## Important constraints

This is for **local/offline learning**, not a hardened internet-facing upload service. The server enforces request/image limits, validates parameters, serves only allowlisted assets, and does not persist uploaded images. Do not bind it to a public interface without authentication, rate limiting, isolation and operational review. External datasets are not redistributed; use their published licenses.

## Original work and references

Algorithm concepts follow classical computer vision texts and [OpenCV documentation](https://docs.opencv.org/4.x/). All source code and synthetic sample images in this repository are independently authored.

## v0.2: First-principles experiments

```bash
python -m opencv_tutorial sample --output results/sample.png
python -m opencv_tutorial ablate --input results/sample.png --algorithm gaussian \
  --study sigma --noise gaussian --strength 18 --seed 7 --output results/gaussian.json
python -m opencv_tutorial serve
```

Ablation JSON and PNG contact sheet use **one resized clean input and one deterministic noisy observation**, varying only the selected algorithm parameter. Denoising scores compare outputs with the clean grayscale image. No segmentation/edge quality scores are fabricated without ground-truth masks. Runtime from the study is **not** a benchmark; use `benchmark` for a timed comparison.

Browser: click a stage, drag the step slider, or press Play for a sequence of *real computed images*. The stage list contains diagnostics that are separately labeled (e.g. mean-threshold reference is **not** a stage of the Gaussian adaptive threshold result).

More information: [v0.2 experiment protocol](docs/v0.2-experiments.md).


## v0.3: intermediate feature & motion observatory

The intermediate lab introduces **real ORB/RANSAC panorama**, **pyramidal Lucas–Kanade flow**
with forward/backward consistency, and a deterministic nearest-centroid tracker with identity
lifecycle. Results include transparent intermediate imagery and inspectable metrics.

```bash
python -m opencv_tutorial evaluate-intermediate --output results/intermediate.json
python -m opencv_tutorial serve   # visit /#intermediate
python -m opencv_tutorial panorama --first a.png --second b.png --output pano.png
python -m opencv_tutorial flow --first frame1.png --second frame2.png --output flow.png
python -m opencv_tutorial track --input clip.mp4 --output annotated.mp4
```

**Known limitations:** Two-image planar panorama, sparse (not dense) optical flow, centroid-only
association vulnerable to crossing/occlusion; synthetic scores are sanity checks, not real-world
benchmarks. See [v0.3 methodology](docs/v0.3-intermediate.md).
