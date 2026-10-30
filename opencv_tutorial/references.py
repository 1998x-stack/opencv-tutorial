"""Readable NumPy-only reference algorithms for W03–W06.

These are deliberately *not* fast and are restricted to small teaching images.
Pixel semantics, padding, tie-breaking and footprint choice are explicit.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .algorithms import require_gray

MAX_REFERENCE_PIXELS = 400_000


def _validate(image: np.ndarray, size: int) -> np.ndarray:
    require_gray(image)
    if image.size > MAX_REFERENCE_PIXELS:
        raise ValueError("Reference algorithms are limited to 400000 pixels")
    if not isinstance(size, int) or size < 3 or size > 15 or size % 2 != 1:
        raise ValueError("size must be an odd integer in [3,15]")
    return image


def median_manual(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Median over each replicated neighborhood, matching cv2.medianBlur."""
    image = _validate(image, size)
    padding = size // 2
    padded = np.pad(image, padding, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-1, -2)).astype(np.uint8)


def morphology_manual(image: np.ndarray, size: int = 3, operation: str = "erode") -> np.ndarray:
    """Erosion/dilation with a square, all-ones footprint and neutral outside border.

    Neutral padding avoids treating the image border as black foreground.
    The choice of rectangular footprint is intentional and documented.
    """
    image = _validate(image, size)
    if operation not in ("erode", "dilate", "open", "close"):
        raise ValueError("operation must be erode, dilate, open or close")

    def elementary(src: np.ndarray, op: str) -> np.ndarray:
        pad_value = 255 if op == "erode" else 0
        padded = np.pad(src, size // 2, mode="constant", constant_values=pad_value)
        windows = sliding_window_view(padded, (size, size))
        return (windows.min(axis=(-1, -2)) if op == "erode"
                else windows.max(axis=(-1, -2))).astype(np.uint8)

    if operation in ("erode", "dilate"):
        return elementary(image, operation)
    if operation == "open":
        return elementary(elementary(image, "erode"), "dilate")
    return elementary(elementary(image, "dilate"), "erode")


def adaptive_mean_manual(image: np.ndarray, size: int = 5, c: float = 3.0) -> np.ndarray:
    """Mean threshold with BORDER_REPLICATE; result 255 where src > local_mean-c."""
    image = _validate(image, size)
    if not np.isfinite(c) or not -255 <= c <= 255:
        raise ValueError("c must be finite and in [-255,255]")
    padded = np.pad(image.astype(np.float64), size // 2, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    mean = windows.mean(axis=(-1, -2))
    return np.where(image.astype(np.float64) > mean - c, 255, 0).astype(np.uint8)


def histogram_manual(image: np.ndarray) -> np.ndarray:
    """256-bin image histogram, including empty bins."""
    require_gray(image)
    return np.bincount(image.ravel(), minlength=256).astype(np.int64)


def equalize_hist_manual(image: np.ndarray) -> np.ndarray:
    """Histogram CDF equalization; constant-valued image stays unchanged."""
    require_gray(image)
    hist = histogram_manual(image)
    cdf = np.cumsum(hist)
    first = int(cdf[np.flatnonzero(hist)[0]])
    if first == image.size:
        return image.copy()
    lut = np.clip(np.rint((cdf - first) * 255 / (image.size - first)), 0, 255).astype(np.uint8)
    return lut[image]
