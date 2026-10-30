"""Two-image ORB/RANSAC panorama with explicit geometric failure detection.

Works best on approximately planar scenes or pictures taken about the optical
centre. This is not a general multi-view bundle adjustment panorama engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .features import match_orb


@dataclass
class PanoramaResult:
    image: np.ndarray
    matches: np.ndarray
    warped_first: np.ndarray
    overlay_second: np.ndarray
    overlap: np.ndarray
    homography: np.ndarray
    stats: dict


def _validate_image(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or
            image.shape[0] < 32 or image.shape[1] < 32 or
            image.shape[0] * image.shape[1] > 4_000_000):
        raise ValueError("Expected uint8 BGR image of 32+ pixels per side, <= 4MP")


def stitch_pair(first: np.ndarray, second: np.ndarray, *,
                min_inliers: int = 10, max_canvas_pixels: int = 10_000_000,
                max_features: int = 1800) -> PanoramaResult:
    """Warp first into second's reference plane and blend only valid pixels.

    Reject weak/degenerate projective estimates and oversized canvas requests.
    """
    _validate_image(first)
    _validate_image(second)
    if min_inliers < 4 or max_canvas_pixels < 4096:
        raise ValueError("Invalid panorama limits")
    cv2.setRNGSeed(7)
    match = match_orb(first, second, max_features=max_features)
    H = match.homography
    if H is None or match.inliers < min_inliers or match.candidate_matches == 0:
        raise ValueError("Insufficient geometrically consistent feature matches")
    ratio = match.inliers / match.candidate_matches
    if ratio < .35:
        raise ValueError("Insufficient inlier ratio for a stable panorama")
    if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-9:
        raise ValueError("Degenerate homography")
    h1, w1 = first.shape[:2]
    h2, w2 = second.shape[:2]
    corners_first = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    projected = cv2.perspectiveTransform(corners_first.reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected.reshape(-1, 1, 2)):
        raise ValueError("Invalid projected image corners")
    poly_area = abs(float(cv2.contourArea(projected)))
    original_area = w1 * h1
    if not .1 <= poly_area / original_area <= 10:
        raise ValueError("Implausible projective scale change")
    all_pts = np.vstack([projected, [[0, 0], [w2, 0], [w2, h2], [0, h2]]])
    xmin, ymin = np.floor(all_pts.min(axis=0)).astype(np.int64)
    xmax, ymax = np.ceil(all_pts.max(axis=0)).astype(np.int64)
    width, height = int(xmax - xmin), int(ymax - ymin)
    if (width < 1 or height < 1 or width * height > max_canvas_pixels or
            width > 12000 or height > 12000):
        raise ValueError("Projected panorama exceeds canvas limit")
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    matrix = translation @ H
    warped = cv2.warpPerspective(first, matrix, (width, height))
    first_mask = cv2.warpPerspective(np.full((h1, w1), 255, np.uint8),
                                     matrix, (width, height), flags=cv2.INTER_NEAREST)
    placed = np.zeros_like(warped)
    placed_mask = np.zeros((height, width), np.uint8)
    x, y = int(-xmin), int(-ymin)
    placed[y:y+h2, x:x+w2] = second
    placed_mask[y:y+h2, x:x+w2] = 255
    valid_first, valid_second = first_mask > 0, placed_mask > 0
    both = valid_first & valid_second
    out = warped.copy()
    out[valid_second] = placed[valid_second]
    out[both] = ((warped[both].astype(np.uint16) + placed[both].astype(np.uint16)) // 2).astype(np.uint8)
    union = valid_first | valid_second
    yy, xx = np.nonzero(union)
    left, right = int(xx.min()), int(xx.max()) + 1
    top, bottom = int(yy.min()), int(yy.max()) + 1
    view = np.s_[top:bottom, left:right]
    stats = {"candidate_matches": match.candidate_matches, "inliers": match.inliers,
             "inlier_ratio": round(ratio, 4), "overlap_pixels": int(both.sum()),
             "canvas_size": [right-left, bottom-top], "keypoints_first": match.keypoints_a,
             "keypoints_second": match.keypoints_b}
    return PanoramaResult(out[view], match.matched, warped[view], placed[view],
                          (both[view].astype(np.uint8) * 255), H, stats)
