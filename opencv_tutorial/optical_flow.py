"""Sparse pyramidal Lucas-Kanade flow with forward/backward consistency gating."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FlowResult:
    annotated: np.ndarray
    points_first: np.ndarray
    points_second: np.ndarray
    forward_backward_error: np.ndarray
    stats: dict


def _validate_pair(first: np.ndarray, second: np.ndarray) -> None:
    for frame in (first, second):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim not in (2, 3) or frame.shape[0] < 32 or
                frame.shape[1] < 32 or frame.shape[0] * frame.shape[1] > 4_000_000):
            raise ValueError("Flow frames must be uint8 images 32px+, <=4MP")
        if frame.ndim == 3 and frame.shape[2] != 3:
            raise ValueError("Flow frames must be BGR or grayscale")
    if first.shape[:2] != second.shape[:2]:
        raise ValueError("Flow frames must share dimensions")


def sparse_lk(first: np.ndarray, second: np.ndarray, *, max_corners: int = 300,
              fb_max_error: float = 1.5, quality_level: float = .01,
              min_distance: float = 7.0) -> FlowResult:
    _validate_pair(first, second)
    if not (1 <= max_corners <= 2000 and 0 < fb_max_error <= 20 and
            0 < quality_level < 1 and 0 < min_distance <= 100):
        raise ValueError("Invalid optical flow parameter")
    gray_a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY) if first.ndim == 3 else first
    gray_b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY) if second.ndim == 3 else second
    points = cv2.goodFeaturesToTrack(gray_a, maxCorners=max_corners,
                                     qualityLevel=quality_level,
                                     minDistance=min_distance, blockSize=7)
    painted = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR) if second.ndim == 2 else second.copy()
    empty = np.empty((0, 2), np.float32)
    if points is None:
        return FlowResult(painted, empty, empty.copy(), np.empty(0),
                          {"detected": 0, "tracked": 0, "survival_ratio": 0.,
                           "median_dx": 0., "median_dy": 0., "median_fb_error": None})
    lk = {"winSize": (21, 21), "maxLevel": 3,
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
    next_pts, status_fw, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, **lk)
    if next_pts is None:
        good = np.zeros(len(points), dtype=bool)
        errors = np.full(len(points), np.inf)
    else:
        back_pts, status_bw, _ = cv2.calcOpticalFlowPyrLK(
            gray_b, gray_a, next_pts, None, **lk)
        if back_pts is None:
            good = np.zeros(len(points), dtype=bool)
            errors = np.full(len(points), np.inf)
        else:
            errors = np.linalg.norm(points.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
            good = ((status_fw.reshape(-1) == 1) & (status_bw.reshape(-1) == 1) &
                    np.isfinite(errors) & (errors <= fb_max_error) &
                    np.isfinite(next_pts.reshape(-1, 2)).all(axis=1))
            h, w = gray_b.shape
            coords = next_pts.reshape(-1, 2)
            good &= ((coords[:, 0] >= 0) & (coords[:, 0] < w) &
                     (coords[:, 1] >= 0) & (coords[:, 1] < h))
    origin = points.reshape(-1, 2)[good]
    dest = next_pts.reshape(-1, 2)[good] if next_pts is not None else empty.copy()
    valid_err = errors[good]
    for (x1, y1), (x2, y2) in zip(origin, dest):
        pt1, pt2 = (round(float(x1)), round(float(y1))), (round(float(x2)), round(float(y2)))
        cv2.arrowedLine(painted, pt1, pt2, (46, 232, 190), 1, tipLength=.35)
        cv2.circle(painted, pt2, 2, (30, 128, 255), -1)
    delta = dest - origin
    stats = {"detected": len(points), "tracked": len(origin),
             "survival_ratio": round(len(origin) / len(points), 4),
             "median_dx": round(float(np.median(delta[:, 0])), 3) if len(delta) else 0.,
             "median_dy": round(float(np.median(delta[:, 1])), 3) if len(delta) else 0.,
             "median_fb_error": round(float(np.median(valid_err)), 4) if len(valid_err) else None}
    return FlowResult(painted, origin, dest, valid_err, stats)


def flow_endpoint_error(result: FlowResult, dx: float, dy: float) -> dict:
    """EPE for synthetic translations only: not a general optical-flow benchmark."""
    if not len(result.points_first):
        return {"valid": False, "count": 0, "mean_epe": None, "median_epe": None}
    diff = result.points_second - result.points_first - np.array([dx, dy])
    epe = np.linalg.norm(diff, axis=1)
    return {"valid": True, "count": len(epe),
            "mean_epe": round(float(np.mean(epe)), 4),
            "median_epe": round(float(np.median(epe)), 4)}
