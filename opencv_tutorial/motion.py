"""Stateful foreground extraction for fixed-camera video."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionFrame:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    annotated: np.ndarray


class MotionDetector:
    def __init__(self, history: int = 240, threshold: float = 24, min_area: int = 400):
        if history < 2 or threshold <= 0 or min_area < 1:
            raise ValueError("Invalid motion detector configuration")
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=threshold, detectShadows=True
        )
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def step(self, bgr: np.ndarray, learning_rate: float = -1) -> MotionFrame:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected BGR frame")
        if not (-1 <= learning_rate <= 1):
            raise ValueError("learning_rate must be -1 or between 0 and 1")
        raw = self.background.apply(bgr, learningRate=learning_rate)
        # MOG2: 0 background, 127 shadow, 255 foreground.
        _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        annotated = bgr.copy()
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            box = tuple(int(v) for v in cv2.boundingRect(contour))
            boxes.append(box)
            x, y, w, h = box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (60, 235, 150), 2)
        return MotionFrame(mask, boxes, annotated)
