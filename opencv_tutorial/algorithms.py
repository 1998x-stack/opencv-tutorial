"""Reference implementations for teaching, deliberately explicit and independently testable."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def require_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty 2D uint8 grayscale image")
    return image


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D correlation with OpenCV's BORDER_REFLECT_101 boundary convention.

    Unlike mathematical convolution, kernel is not flipped. Output is float64.
    This is a teaching baseline, not a production-optimized implementation.
    """
    require_gray(image)
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or not all(v % 2 == 1 and v > 0 for v in k.shape):
        raise ValueError("Kernel must be nonempty, 2D, and odd sized")
    if not np.isfinite(k).all():
        raise ValueError("Kernel values must be finite")
    py, px = k.shape[0] // 2, k.shape[1] // 2
    padded = np.pad(image.astype(np.float64), ((py, py), (px, px)), mode="reflect")
    windows = sliding_window_view(padded, k.shape)
    return np.einsum("ijkl,kl->ij", windows, k, optimize=True)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size < 3 or size > 31 or size % 2 != 1 or not 0 < sigma <= 20:
        raise ValueError("size must be odd in [3,31]; sigma must be in (0,20]")
    ax = np.arange(size, dtype=np.float64) - size // 2
    weights = np.exp(-(ax**2) / (2 * sigma**2))
    weights /= weights.sum()
    return np.outer(weights, weights)


def gaussian_manual(image: np.ndarray, size: int = 5, sigma: float = 1.2) -> np.ndarray:
    return np.clip(np.rint(correlate2d(image, gaussian_kernel(size, sigma))), 0, 255).astype(
        np.uint8
    )


def sobel_manual(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    gx, gy = correlate2d(image, kx), correlate2d(image, ky)
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude


def otsu_manual(image: np.ndarray) -> int:
    """Determine the 8-bit threshold maximizing between-class variance.

    Ties select the first maximum, matching typical discrete Otsu conventions.
    """
    require_gray(image)
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    total = mu[-1]
    denominator = omega * (1 - omega)
    variance = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    variance[valid] = (total * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    return int(np.argmax(variance))


def binary_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    require_gray(image)
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in [0,255]")
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def make_sample(width: int = 768, height: int = 480, seed: int = 7) -> np.ndarray:
    """Deterministic synthetic photo for offline lessons; no external image rights."""
    if not 128 <= width <= 2048 or not 128 <= height <= 2048:
        raise ValueError("sample dimensions must be in [128,2048]")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    canvas[..., 0] = 65 + 55 * xx / width
    canvas[..., 1] = 69 + 36 * yy / height
    canvas[..., 2] = 82 + 20 * xx / width
    noise = rng.normal(0, 8, (height, width, 1))
    canvas += noise
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (int(width * .08), int(height * .18)),
                  (int(width * .45), int(height * .78)), (238, 195, 94), -1)
    cv2.circle(image, (int(width * .71), int(height * .50)),
               int(min(width, height) * .24), (75, 183, 252), -1)
    cv2.line(image, (int(width * .07), int(height * .91)),
             (int(width * .94), int(height * .1)), (245, 245, 245), 7)
    cv2.putText(image, "VISION", (int(width * .12), int(height * .53)),
                cv2.FONT_HERSHEY_SIMPLEX, max(.5, width / 1200), (26, 31, 47), 2, cv2.LINE_AA)
    return image
