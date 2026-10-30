"""Offline synthetic fixtures and reproducible intermediate evaluation."""
from __future__ import annotations

import cv2
import numpy as np

from .optical_flow import flow_endpoint_error, sparse_lk
from .panorama import stitch_pair
from .tracking import CentroidTracker


def synthetic_panorama_pair(seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Two crops of one planar texture; true correspondence is translation."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    rng = np.random.default_rng(seed)
    world = rng.integers(20, 200, (300, 680, 3), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (3, 3), .6)
    for _ in range(75):
        x, y = int(rng.integers(10, 665)), int(rng.integers(10, 285))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(world, (x, y), int(rng.integers(2, 8)), color, -1)
    for i in range(12):
        cv2.putText(world, f"{i:02d}", (15 + 53 * i, 145 + (i % 3) * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (245, 245, 245), 1)
    return world[:, :420].copy(), world[:, 230:650].copy()


def synthetic_translation(seed: int = 19, *, dx: int = 7,
                          dy: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Globally translated textured image; uncovered pixels have zero information."""
    rng = np.random.default_rng(seed)
    first = rng.integers(0, 255, (220, 320), dtype=np.uint8)
    first = cv2.GaussianBlur(first, (5, 5), .7)
    affine = np.float32([[1, 0, dx], [0, 1, dy]])
    second = cv2.warpAffine(first, affine, (320, 220), borderMode=cv2.BORDER_CONSTANT)
    return cv2.cvtColor(first, cv2.COLOR_GRAY2BGR), cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)


def synthetic_tracking_frames(count: int = 14) -> list[np.ndarray]:
    if not 1 <= count <= 120:
        raise ValueError("count must be between 1 and 120")
    frames = []
    for t in range(count):
        frame = np.full((200, 360, 3), 23, np.uint8)
        cv2.rectangle(frame, (24+8*t, 35+3*t), (55+8*t, 66+3*t), (230, 230, 230), -1)
        cv2.rectangle(frame, (275-5*t, 120-2*t), (313-5*t, 153-2*t), (160, 220, 240), -1)
        frames.append(frame)
    return frames


def benchmark_synthetic(seed: int = 17, *, dx: int = 7, dy: int = 4) -> dict:
    """Deterministic synthetic measurements; do not generalise to natural imagery."""
    a, b = synthetic_panorama_pair(seed)
    panorama = stitch_pair(a, b)
    f1, f2 = synthetic_translation(seed, dx=dx, dy=dy)
    flow = sparse_lk(f1, f2)
    return {"seed": seed, "panorama": panorama.stats,
            "panorama_homography": panorama.homography.tolist(),
            "flow": flow.stats, "flow_ground_truth": {"dx": dx, "dy": dy},
            "flow_error": flow_endpoint_error(flow, dx, dy),
            "limitations": "synthetic two-crop planar panorama; synthetic global translation;"
                           " not indicative of wild-scene accuracy"}


def track_synthetic(count: int = 14) -> tuple[list[np.ndarray], list[dict]]:
    """Track known synthetic boxes to isolate association from foreground quality."""
    frames = synthetic_tracking_frames(count)
    tracker = CentroidTracker(max_distance=30, max_missed=2)
    outputs, steps = [], []
    for t, frame in enumerate(frames):
        boxes = [(24+8*t, 35+3*t, 32, 32), (275-5*t, 120-2*t, 39, 34)]
        tracks = tracker.update(boxes)
        outputs.append(tracker.annotate(frame))
        steps.append({"frame": t, "tracks": [{"id": r.id, "box": r.box,
                                               "missed": r.missed} for r in tracks]})
    return outputs, steps
