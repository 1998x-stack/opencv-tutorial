# Reproducibility and test methodology

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check opencv_tutorial tests
python -m opencv_tutorial benchmark --rounds 30 --output results/bench.json
```

- **Parity:** NumPy correlation vs `cv2.filter2D`, Gaussian vs `cv2.GaussianBlur` (quantization tolerance), Sobel gradient vs `cv2.Sobel`, Otsu vs `cv2.threshold`.
- **Determinism:** seeded synthetic scenes; no random external web image inputs in test suite.
- **Integration:** local HTTP health, static UI, curriculum and algorithm processing.
- **Motion:** stationary background should produce zero foreground after warmup; introduced object produces a region.
- **Geometry:** corners, order and perspective warp on synthetic rectangle.
- **Performance:** median and P95 of repeated warmed runs; do not include network transfer or GUI display in algorithm timing.
- **Future:** optical flow benchmark EPE, tracking IDF1/HOTA, stereo EPE/bad-pixel rate, project-specific golden data.

Known limitations: histogram threshold ties can have multiple optimal values; camera/codec availability differs by machine; the UI currently requires manual browser validation in addition to HTTP integration tests. `elapsed_ms` for some algorithms includes optional NumPy reference comparison and must not be treated as optimized OpenCV-only time.
