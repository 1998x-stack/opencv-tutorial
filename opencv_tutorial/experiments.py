"""Controlled v0.2 parameter sweeps and deterministic noise for educational ablations."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .evaluation import compare_images
from .pipeline import run_pipeline

STUDIES = {
    "gaussian": {"kernel": (3, 5, 7, 9), "sigma": (0.6, 1.2, 2.0, 3.0)},
    "median": {"kernel": (3, 5, 7, 9)},
    "canny": {"threshold": (30, 60, 100, 150)},
    "adaptive": {"kernel": (3, 5, 9, 15)},
    "morphology": {"kernel": (3, 5, 7, 9), "threshold": (60, 100, 140, 180)},
    "contours": {"threshold": (60, 100, 140, 180)},
}
NOISE_TYPES = ("none", "gaussian", "salt_pepper")


@dataclass(frozen=True)
class ExperimentInput:
    clean: np.ndarray
    observed: np.ndarray
    noise: str
    strength: float
    seed: int


def apply_noise(bgr: np.ndarray, noise: str = "none", strength: float = 18.0,
                seed: int = 7) -> ExperimentInput:
    """One immutable noisy observation shared by every parameter candidate."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[-1] != 3 or bgr.size == 0:
        raise ValueError("noise input must be a uint8 BGR image")
    if noise not in NOISE_TYPES:
        raise ValueError("noise must be none, gaussian or salt_pepper")
    if not math.isfinite(strength) or not 0 <= strength <= 60:
        raise ValueError("noise strength must be in [0,60]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("noise seed must be a uint32 integer")
    clean = bgr.copy()
    if noise == "none" or strength == 0:
        return ExperimentInput(clean, clean.copy(), noise, strength, seed)
    rng = np.random.default_rng(seed)
    if noise == "gaussian":
        disturbance = rng.normal(0, strength, bgr.shape)
        observed = np.clip(np.rint(clean.astype(np.float32) + disturbance), 0, 255).astype(np.uint8)
    else:
        # strength is the percentage of pixels replaced, not a probability from 0 to 1.
        observed = clean.copy()
        selected = rng.random(clean.shape[:2]) < strength / 100
        salt = rng.random(clean.shape[:2]) < .5
        observed[selected & salt] = 255
        observed[selected & ~salt] = 0
    return ExperimentInput(clean, observed, noise, strength, seed)


def _scores(clean: np.ndarray, observed: np.ndarray, processed: np.ndarray,
            algorithm: str, noise: str) -> dict:
    if algorithm not in ("gaussian", "median") or noise == "none":
        return {"quality_available": False, "quality_reason":
                "PSNR/SSIM require a pristine reference and a denoising operation"}
    reference = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    baseline = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    return {"quality_available": True, "before": compare_images(reference, baseline),
            "after": compare_images(reference, processed)}


def run_study(bgr: np.ndarray, algorithm: str, study: str,
              *, noise: str = "gaussian", strength: float = 18.0,
              seed: int = 7, kernel: int = 5, sigma: float = 1.2,
              threshold: int = 120) -> dict:
    if algorithm not in STUDIES or study not in STUDIES[algorithm]:
        raise ValueError("Unsupported algorithm/parameter study combination")
    if bgr.ndim != 3 or bgr.dtype != np.uint8 or bgr.shape[2] != 3:
        raise ValueError("Ablation expects uint8 BGR input")
    # Make studies bounded and responsive. Resize ONCE before adding identical noise.
    h, w = bgr.shape[:2]
    max_pixels = 110_000
    if h * w > max_pixels:
        scale = math.sqrt(max_pixels / (h * w))
        bgr = cv2.resize(bgr, (max(16, int(w * scale)), max(16, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    data = apply_noise(bgr, noise=noise, strength=strength, seed=seed)
    baseline = {"kernel": kernel, "sigma": sigma, "threshold": threshold}
    # Explicit validation of even the fixed parameters before starting the sweep.
    run_pipeline(data.observed, algorithm, **baseline)
    rows = []
    for value in STUDIES[algorithm][study]:
        params = {**baseline, study: value}
        result = run_pipeline(data.observed, algorithm, **params)
        quality = _scores(data.clean, data.observed, result.image, algorithm, noise)
        rows.append({"value": value, "parameters": params, "metrics": result.metrics,
                     "quality": quality, "image": result.image})
    return {"algorithm": algorithm, "study": study, "noise": noise, "strength": strength,
            "seed": seed, "clean": data.clean, "observed": data.observed,
            "baseline": baseline, "study_resolution": list(data.observed.shape[1::-1]),
            "candidates": rows,
            "method": "single fixed observation, fixed input order, one execution per candidate;"
                      " runtime is diagnostic, not a statistically valid benchmark"}
