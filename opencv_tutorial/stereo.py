"""Calibrated, rectified stereo-depth lab: explicit geometry and invalid disparity masks."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoResult:
    disparity: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray
    visualization: np.ndarray
    stats: dict


def stereo_depth(left: np.ndarray, right: np.ndarray, *, fx_px: float,
                 baseline_m: float, min_disparity: int = 0,
                 num_disparities: int = 64, block_size: int = 5) -> StereoResult:
    """Estimate depth Z=fx*baseline/disparity for *already rectified* stereo pairs.

    This does not calibrate/rectify cameras or establish that supplied calibration is true.
    SGBM disparity is fixed-point 1/16 px; nonpositive and sentinel values are invalid.
    """
    if (not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray)
            or left.shape != right.shape or left.ndim not in (2, 3)
            or left.dtype != np.uint8 or right.dtype != np.uint8
            or left.size == 0 or (left.ndim == 3 and left.shape[2] != 3)):
        raise ValueError("Stereo images must have matching nonempty uint8 grayscale/BGR shapes")
    if (not np.isfinite(fx_px) or fx_px <= 0 or not np.isfinite(baseline_m)
            or baseline_m <= 0):
        raise ValueError("fx_px and baseline_m must be finite positive calibrated values")
    if (not isinstance(min_disparity, int) or min_disparity < 0 or min_disparity > 128
            or num_disparities < 16 or num_disparities > 256
            or num_disparities % 16 or block_size < 3 or block_size > 21
            or block_size % 2 != 1):
        raise ValueError("Invalid SGBM parameters (disparities multiple of 16, odd block 3..21)")
    height, width = left.shape[:2]
    if min(height, width) < 32 or width <= min_disparity + num_disparities + block_size:
        raise ValueError("Image too small for stereo search range")
    a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    channels = 1
    sgbm = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                 numDisparities=num_disparities, blockSize=block_size,
                                 P1=8 * channels * block_size**2,
                                 P2=32 * channels * block_size**2,
                                 uniquenessRatio=7, speckleWindowSize=40,
                                 speckleRange=2, disp12MaxDiff=1)
    disparity = sgbm.compute(a, b).astype(np.float32) / 16.0
    valid = np.isfinite(disparity) & (disparity > max(0, min_disparity))
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (fx_px * baseline_m) / disparity[valid]
    scaled = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        scaled[valid] = np.clip(255 * disparity[valid] /
                                (min_disparity + num_disparities), 0, 255).astype(np.uint8)
    visualization = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    visualization[~valid] = 0
    stats = {"valid_fraction": float(np.mean(valid)),
             "valid_pixels": int(valid.sum()),
             "median_disparity_px": float(np.median(disparity[valid])) if valid.any() else None,
             "median_depth_m": float(np.median(depth[valid])) if valid.any() else None,
             "fx_px": float(fx_px), "baseline_m": float(baseline_m),
             "rectification": "assumed; not verified",
             "depth_units": "meters", "method": "StereoSGBM"}
    return StereoResult(disparity, depth, valid, visualization, stats)


def synthetic_stereo(seed: int = 17, *, disparity_px: int = 8,
                     width: int = 320, height: int = 220) -> tuple[np.ndarray, np.ndarray]:
    """Textured rectified flat plane, right image shifted left by known disparity."""
    if not 1 <= disparity_px <= 24 or min(width, height) < 128:
        raise ValueError("Invalid fixture dimensions or disparity")
    rng = np.random.default_rng(seed)
    canvas = rng.integers(20, 235, (height, width + disparity_px, 3), dtype=np.uint8)
    for _ in range(150):
        x = int(rng.integers(8, width - 8))
        y = int(rng.integers(8, height - 8))
        cv2.circle(canvas, (x, y), int(rng.integers(2, 5)),
                   tuple(int(v) for v in rng.integers(10, 245, 3)), -1)
    return canvas[:, :width].copy(), canvas[:, disparity_px:disparity_px + width].copy()


def stereo_fixture_report(seed: int = 17, *, disparity_px: int = 8) -> dict:
    left, right = synthetic_stereo(seed, disparity_px=disparity_px)
    fx, baseline = 420.0, 0.12
    result = stereo_depth(left, right, fx_px=fx, baseline_m=baseline)
    # Evaluate only a stable interior, not SGBM's known border occlusion/search region.
    roi = result.valid[35:-35, 90:-35]
    recovered = result.disparity[35:-35, 90:-35]
    median_error = float(np.median(np.abs(recovered[roi] - disparity_px))) if roi.any() else None
    return {"seed": seed, "expected_disparity_px": disparity_px,
            "expected_depth_m": fx * baseline / disparity_px,
            "median_abs_disparity_error_px": median_error, "stats": result.stats}
