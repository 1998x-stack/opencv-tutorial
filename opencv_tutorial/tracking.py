"""Deterministic nearest-centre multi-object tracker; no identity recovery guarantee."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite

import cv2
import numpy as np


@dataclass
class Track:
    id: int
    box: tuple[int, int, int, int]
    missed: int = 0
    age: int = 1
    path: list[tuple[int, int]] = field(default_factory=list)


def _center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = box
    return x + w // 2, y + h // 2


class CentroidTracker:
    """One-to-one greedy association; deterministic tie breaking; max disappearance TTL."""

    def __init__(self, max_distance: float = 45, max_missed: int = 4,
                 max_trail: int = 60):
        if (not isfinite(max_distance) or max_distance <= 0 or
                not 0 <= max_missed <= 120 or not 1 <= max_trail <= 1000):
            raise ValueError("Invalid tracker configuration")
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.max_trail = max_trail
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, boxes: list[tuple[int, int, int, int]]) -> list[Track]:
        """Return active tracks, including missed tracks until their TTL expires."""
        if len(boxes) > 500:
            raise ValueError("Too many detections")
        clean = []
        for box in boxes:
            if len(box) != 4 or any(not isinstance(v, (int, np.integer)) for v in box):
                raise ValueError("Boxes must be integer (x,y,w,h) tuples")
            x, y, w, h = map(int, box)
            if x < 0 or y < 0 or w < 1 or h < 1:
                raise ValueError("Invalid box dimensions")
            clean.append((x, y, w, h))
        # Prefer best global candidate pair, subject to uniqueness and distance gate.
        pairs = []
        for tid, track in self.tracks.items():
            px, py = _center(track.box)
            for idx, box in enumerate(clean):
                qx, qy = _center(box)
                distance = hypot(qx-px, qy-py)
                if distance <= self.max_distance:
                    pairs.append((distance, tid, idx))
        taken_tracks: set[int] = set()
        taken_boxes: set[int] = set()
        for _, tid, idx in sorted(pairs):
            if tid in taken_tracks or idx in taken_boxes:
                continue
            track = self.tracks[tid]
            track.box = clean[idx]
            track.missed = 0
            track.age += 1
            track.path.append(_center(track.box))
            track.path = track.path[-self.max_trail:]
            taken_tracks.add(tid)
            taken_boxes.add(idx)
        for tid, track in list(self.tracks.items()):
            if tid not in taken_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self.tracks[tid]
        for idx, box in enumerate(clean):
            if idx in taken_boxes:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, box, path=[_center(box)])
        return [self.tracks[tid] for tid in sorted(self.tracks)]

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        for track in self.tracks.values():
            if track.missed:
                continue
            x, y, w, h = track.box
            cv2.rectangle(out, (x, y), (x+w, y+h), (58, 220, 173), 2)
            cv2.putText(out, f"ID {track.id}", (x, max(y-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (58, 220, 173), 1)
            if len(track.path) > 1:
                cv2.polylines(out, [np.array(track.path, np.int32)], False,
                              (65, 175, 255), 2)
        return out
