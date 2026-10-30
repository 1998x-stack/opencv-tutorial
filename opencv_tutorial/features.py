"""Local features and geometric verification without learned model downloads."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    matched: np.ndarray
    keypoints_a: int
    keypoints_b: int
    candidate_matches: int
    inliers: int
    homography: np.ndarray | None


def match_orb(image_a: np.ndarray, image_b: np.ndarray, max_features: int = 1000) -> MatchResult:
    if max_features < 50 or max_features > 10_000:
        raise ValueError("max_features must be in [50,10000]")
    def to_gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    ga, gb = to_gray(image_a), to_gray(image_b)
    orb = cv2.ORB_create(nfeatures=max_features)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None:
        return MatchResult(cv2.hconcat([image_a, image_b]), len(ka), len(kb), 0, 0, None)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .75 * n.distance]
    H, mask = None, None
    if len(good) >= 4:
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = int(np.sum(mask)) if mask is not None else 0
    preview = cv2.drawMatches(image_a, ka, image_b, kb, good[:80], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return MatchResult(preview, len(ka), len(kb), len(good), inliers, H)
