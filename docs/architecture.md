# Architecture and contracts

## Goals

A single deterministic execution engine serves Python calls, a standard-library HTTP API, a browser UI and benchmarks. The teaching NumPy implementations are separate from optimized OpenCV routines to keep comparisons meaningful.

```text
Browser (vanilla JavaScript, local static assets)
    -> POST /api/process { image: data URL, algorithm, kernel, sigma, threshold }
        -> validation / decoding / pixel limits
            -> run_pipeline (one algorithm / intermediate stages / metrics)
                -> base64 PNG stages and JSON metrics
```

## Invariants

- Inputs: nonempty uint8 BGR arrays; images at most 8 million pixels.
- The HTTP payload must be JSON and no larger than 8 MiB; bytes are decoded in memory.
- Algorithms come from a strict allowlist; kernel must be odd 3..31; thresholds 0..255.
- Intermediate stages and output are encoded losslessly as PNG, never transformed by the browser.
- The NumPy `correlate2d` is **correlation** (not mathematically flipped convolution), with reflect-101 boundaries matching OpenCV `filter2D`.
- Benchmark output records environment-sensitive measurements; they are not portable throughput claims.
- Deterministic generated images avoid dataset copyright issues in the out-of-box experience.

## Errors and boundaries

Document detection can report that no quadrilateral is detected; it must never synthesize a false-positive result. ORB may have no descriptors, and RANSAC may fail to produce a homography. MOG2 is a scene foreground extractor, **not** a semantic object detector; target ID tracking remains future work. HTML UI is a local visualization and is not remotely authenticated.

## Extension interface

Add algorithm name to `ALGORITHMS`, implement its stage in `run_pipeline`, supply reference and synthetic regression tests, then add its selector entry to the web application. More expensive operations should be moved to an explicit job queue and resource isolation before public deployment.
