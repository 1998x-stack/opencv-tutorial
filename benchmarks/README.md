# Performance experiments

Run `python -m opencv_tutorial benchmark --rounds 30 --output results/bench.json`.

Measurements are **per machine** and include both selected processing and reference-algorithm checks where applicable. Use the same OpenCV build, image dimensions, CPU/GPU backend and package versions for valid comparisons. A future iteration will add warmup controls, per-stage timing and multiple input sizes.
