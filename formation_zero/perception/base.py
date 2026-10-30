"""What every perception model promises (architecture 1.3 B): detections in, tracks out, teams on top.

Models sit behind these protocols, so the trained student replaces the COCO stand-in, or RF-DETR
replaces YOLO, without touching the stages around them. Boxes are always in pixels of the frame
the model was given; frames are RGB uint8, as `ingest.sources` yields them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detections:
    xyxy: np.ndarray                       # (n, 4) float, x1 y1 x2 y2
    confidence: np.ndarray                 # (n,)
    label: np.ndarray                      # (n,) str: "player", "referee"
    track_id: np.ndarray | None = None     # (n,) int, once a tracker has seen them

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, keep) -> Detections:
        return Detections(self.xyxy[keep], self.confidence[keep], self.label[keep],
                          None if self.track_id is None else self.track_id[keep])

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=object))


class Detector(Protocol):
    labels: tuple[str, ...]

    def __call__(self, images: list[np.ndarray]) -> list[Detections]: ...


class Tracker(Protocol):
    def update(self, detections: Detections) -> Detections: ...    # the same boxes with track_id set
    def reset(self) -> None: ...                                     # before each new take
    def cut(self) -> None: ...                                       # at a cut inside a take: tracks end, ids stay unique


class TeamClassifier(Protocol):
    def fit(self, crops: list[np.ndarray]) -> TeamClassifier: ...  # crops from across a whole game
    def predict(self, crops: list[np.ndarray]) -> np.ndarray: ...    # 0 or 1 per crop, -1 if unusable


__all__ = ["Detections", "Detector", "TeamClassifier", "Tracker"]
