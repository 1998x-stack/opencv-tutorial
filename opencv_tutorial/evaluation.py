"""Reference-grounded image quality; never compare unrelated representations."""
from __future__ import annotations

import math

import cv2
import numpy as np


def _pair(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("reference and prediction must have matching 2D shapes")
    if reference.dtype != np.uint8 or prediction.dtype != np.uint8 or reference.size == 0:
        raise ValueError("quality metrics require nonempty uint8 grayscale images")
    return reference.astype(np.float64), prediction.astype(np.float64)


def mae(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref, pred = _pair(reference, prediction)
    return float(np.abs(ref - pred).mean())


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    """dB for uint8 signals, +inf when arrays are identical."""
    ref, pred = _pair(reference, prediction)
    mse = float(np.mean((ref - pred) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean 11x11 Gaussian-window SSIM using population local variances.

    This is the common luminance/contrast/structure form, not MS-SSIM.
    """
    ref, pred = _pair(reference, prediction)
    if min(ref.shape) < 11:
        raise ValueError("SSIM requires image height and width >= 11")
    mu_r = cv2.GaussianBlur(ref, (11, 11), 1.5)
    mu_p = cv2.GaussianBlur(pred, (11, 11), 1.5)
    r2, p2, rp = ref * ref, pred * pred, ref * pred
    sigma_r2 = np.maximum(cv2.GaussianBlur(r2, (11, 11), 1.5) - mu_r**2, 0)
    sigma_p2 = np.maximum(cv2.GaussianBlur(p2, (11, 11), 1.5) - mu_p**2, 0)
    sigma_rp = cv2.GaussianBlur(rp, (11, 11), 1.5) - mu_r * mu_p
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    numerator = (2 * mu_r * mu_p + c1) * (2 * sigma_rp + c2)
    denominator = (mu_r**2 + mu_p**2 + c1) * (sigma_r2 + sigma_p2 + c2)
    return float(np.mean(numerator / denominator))


def compare_images(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    score = psnr(reference, prediction)
    return {"mae": round(mae(reference, prediction), 4),
            "psnr_db": None if math.isinf(score) else round(score, 4),
            "ssim": round(ssim(reference, prediction), 6),
            "identical": bool(math.isinf(score))}


def mask_scores(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Foreground == nonzero; empty-empty masks get a perfect score."""
    _pair(reference, prediction)
    truth, predicted = reference > 0, prediction > 0
    intersection = int(np.count_nonzero(truth & predicted))
    union = int(np.count_nonzero(truth | predicted))
    denom = int(np.count_nonzero(truth)) + int(np.count_nonzero(predicted))
    return {"iou": round(intersection / union if union else 1.0, 6),
            "dice": round(2 * intersection / denom if denom else 1.0, 6)}
