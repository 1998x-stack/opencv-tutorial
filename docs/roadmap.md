# Delivery plan and acceptance gates

## Stage 00 — Verified foundation (v0.1)

- [x] Executable project structure, 24-week syllabus metadata, Python CLI and local web server
- [x] Gaussian, median, Sobel, Canny, Otsu, adaptive threshold, morphology, contours
- [x] NumPy correlation, Gaussian, Sobel and Otsu references with OpenCV parity tests
- [x] Local before/after visualization, intermediate-stage gallery, runtime metrics, file input
- [x] Document detection/rectification and ORB + RANSAC baseline
- [x] MOG2 video foreground detection baseline
- [x] Deterministic samples, regression tests and CI configuration

## Stage 00.2 — Beginner methodology (v0.2)

- [x] Pure NumPy median/morphology/adaptive mean/histogram equalization teaching references
- [x] Stage-by-stage playback of actual intermediate images in the browser
- [x] Fixed-input/fixed-seed parameter ablations with reviewable contact sheets
- [x] MAE, PSNR, SSIM reference-based denoising scores, and IoU/Dice mask score utilities
- [x] Numerical parity, boundary-condition, and HTTP integration tests
- [ ] Complete all eight beginner notebooks (v0.3, substantial work remains)
- [ ] Golden real-world document scanner dataset and independent grading

## Stage 01 — W01–W08 lesson depth

- [ ] Eight lesson notebooks with derivations, graphical step-throughs and independent grading
- [ ] Hough geometry visualizations, CLAHE, bilateral filter and connected component analyses
- [ ] Difficult document benchmarks with ground-truth corner errors and failure categorization
- [ ] Golden image test suite with baseline versions and image-difference artifact output

## Stage 02 — W09–W16 research labs

- [ ] Advanced detector comparisons (FAST/SIFT/ORB), repeatability under transforms
- [ ] RANSAC numerical simulations, panorama blending, seam/ghosting ablation
- [ ] Lucas-Kanade and Farneback optical flow + quantitative errors
- [ ] Kalman-based persistent IDs, motion evaluation dataset and FPS/latency tests

## Stage 03 — W17–W24 advanced engineering

- [ ] Calibrated stereo disparity and Middlebury/KITTI evaluation adapters
- [ ] PnP / essential geometry / visual odometry and reprojection tests
- [ ] ONNX inference with optional model download and SHA256 verification
- [ ] Native C++ implementations, CPU profiling and optional accelerator backends
- [ ] Multi-run benchmarks with warmup, percentile latencies and memory metrics

## Stage 04 — Production-learning platform

- [ ] Lesson state tracking and reproducible experiment manifests
- [ ] Parameter sweeps, image comparisons, lesson completion reports
- [ ] Browser end-to-end automation and accessibility audit
- [ ] Isolated job runner for externally accessible hosting

Definition of done for **each** lesson: theory, code, deterministic example, alternative parameter cases, negative cases, tests, benchmark, and an explainable visualization. No unimplemented exercise should be labeled complete.


## v0.4 incremental achievement (synthetic foundation, NOT completed high-order production vision)

- [x] StereoSGBM depth conversion from **assumed rectified and calibrated** pair; invalid disparity masks.
- [x] PnP RANSAC + LM known-pose benchmarks and relative essential/recoverPose geometry.
- [x] Two-view LK-based relative motion function, synthetic VO trajectory with explicitly unknown scale.
- [x] Offline OpenCV DNN ONNX test fixture and CPU performance warmup/repeated latency probes; CUDA backend guarded.
- [x] Advanced browser lab, CLI, HTTP endpoint, and numerical/negative tests.
- [ ] True stereo calibration and rectification, external disparity/depth dataset scores.
- [ ] Long-sequence VO tracking, triangulation, BA, drift metrics, and absolute scale recovery.
- [ ] Trained object detection models and numeric cross-framework verification.
- [ ] Actual GPU speed benchmarks and CPU C++ native parity.
