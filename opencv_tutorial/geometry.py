"""Projective geometry: robust corner ordering and document rectification."""
from __future__ import annotations

import cv2
import numpy as np


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("Expected exactly four finite 2D points")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    top_left = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -int(top_left), axis=0)
    # angles in image coordinates produce TL, TR, BR, BL on convex quads
    if cv2.contourArea(ordered.reshape(-1, 1, 2)) < 1:
        raise ValueError("Degenerate document corners")
    return ordered


def warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    pts = order_corners(corners)
    tl, tr, br, bl = pts
    width = max(1, int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))))
    height = max(1, int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))))
    if width * height > 12_000_000:
        raise ValueError("Warped document exceeds pixel limit")
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(pts, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def detect_document(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_min = image.shape[0] * image.shape[1] * .08
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < area_min:
            break
        candidate = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        if len(candidate) == 4 and cv2.isContourConvex(candidate):
            return order_corners(candidate.reshape(4, 2))
    return None
