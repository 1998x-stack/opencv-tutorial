"""One shared execution contract for CLI, API and benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .algorithms import binary_threshold, gaussian_manual, otsu_manual, sobel_manual
from .references import (adaptive_mean_manual, equalize_hist_manual,
                         median_manual, morphology_manual)

ALGORITHMS = {
    "gaussian": {"title": "Gaussian blur", "group": "Filtering", "description": "Spatial smoothing with Gaussian weights"},
    "median": {"title": "Median filter", "group": "Filtering", "description": "Remove impulse noise while preserving edges"},
    "sobel": {"title": "Sobel gradient", "group": "Edges", "description": "Directional image derivatives"},
    "canny": {"title": "Canny edges", "group": "Edges", "description": "Non-max suppression and hysteresis"},
    "otsu": {"title": "Otsu threshold", "group": "Segmentation", "description": "Variance-maximizing threshold selection"},
    "adaptive": {"title": "Adaptive threshold", "group": "Segmentation", "description": "Threshold using local neighborhoods"},
    "morphology": {"title": "Morphology", "group": "Segmentation", "description": "Opening followed by closing"},
    "contours": {"title": "Contours", "group": "Geometry", "description": "Extract and outline connected boundaries"},
}


@dataclass
class PipelineResult:
    image: np.ndarray
    stages: dict[str, np.ndarray]
    metrics: dict[str, float | int | str]


def _gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected 8-bit 3-channel BGR image")
    if bgr.size == 0 or bgr.shape[0] * bgr.shape[1] > 8_000_000:
        raise ValueError("Image must be nonempty and at most 8 megapixels")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def run_pipeline(
    bgr: np.ndarray, algorithm: str = "gaussian", kernel: int = 5,
    sigma: float = 1.2, threshold: int = 120,
) -> PipelineResult:
    """Run exactly one selected algorithm; capture consistent intermediate images."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if kernel not in range(3, 32, 2):
        raise ValueError("kernel must be an odd integer from 3 to 31")
    if not 0 < sigma <= 20 or not 0 <= threshold <= 255:
        raise ValueError("sigma must be in (0, 20] and threshold in [0, 255]")

    gray = _gray(bgr)
    stages = {"original": bgr, "grayscale": gray}
    start = perf_counter()
    metrics: dict[str, float | int | str] = {}
    if algorithm == "gaussian":
        image = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
        stages["gaussian_smoothed"] = image
        if gray.size <= 40_000 and kernel <= 9:  # bound reference cost
            reference = gaussian_manual(gray, kernel, sigma)
            metrics["reference_mae"] = round(float(np.mean(np.abs(image.astype(float) - reference))), 4)
    elif algorithm == "median":
        image = cv2.medianBlur(gray, kernel)
        if gray.size <= 40_000 and kernel <= 9:
            ref = median_manual(gray, kernel)
            metrics["reference_pixel_difference"] = int(np.count_nonzero(image != ref))
    elif algorithm == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        image = cv2.convertScaleAbs(magnitude, alpha=255 / max(float(magnitude.max()), 1))
        stages["gradient_magnitude"] = image
        stages["horizontal_gradient"] = cv2.convertScaleAbs(gx)
        stages["vertical_gradient"] = cv2.convertScaleAbs(gy)
        if gray.size <= 400_000:
            ref_x, _, _ = sobel_manual(gray)
            metrics["gradient_max_error"] = round(float(np.max(np.abs(gx - ref_x))), 6)
    elif algorithm == "canny":
        softened = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
        stages["smoothed"] = softened
        gx = cv2.Sobel(softened, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(softened, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        stages["gradient_magnitude"] = cv2.convertScaleAbs(
            mag, alpha=255 / max(float(mag.max()), 1))
        image = cv2.Canny(softened, threshold, min(255, 2 * threshold))
    elif algorithm == "otsu":
        selected = otsu_manual(gray)
        image = binary_threshold(gray, selected)
        _, cv_output = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        metrics["reference_pixel_difference"] = int(np.count_nonzero(cv_output != image))
        metrics["selected_threshold"] = selected
        stages["histogram_equalized"] = equalize_hist_manual(gray)
    elif algorithm == "adaptive":
        image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, kernel, 3)
        if gray.size <= 40_000 and kernel <= 9:
            stages["reference_mean_threshold"] = adaptive_mean_manual(gray, kernel, 3)
    elif algorithm == "morphology":
        binary = binary_threshold(gray, threshold)
        stages["binary"] = binary
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, elem)
        image = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, elem)
        stages["opening"] = opened
        if gray.size <= 40_000 and kernel <= 9:
            stages["reference_rect_opening"] = morphology_manual(binary, kernel, "open")
    else:  # contours
        binary = binary_threshold(gray, threshold)
        stages["binary"] = binary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = bgr.copy()
        cv2.drawContours(image, contours, -1, (70, 225, 160), 2)
        metrics["contour_count"] = len(contours)
    elapsed = (perf_counter() - start) * 1000
    stages["result"] = image
    metrics.update({"elapsed_ms": round(elapsed, 3), "width": int(gray.shape[1]),
                    "height": int(gray.shape[0]), "algorithm": algorithm})
    return PipelineResult(image=image, stages=stages, metrics=metrics)
