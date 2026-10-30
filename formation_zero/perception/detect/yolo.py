"""YOLO detectors: the COCO-pretrained stand-in and the students trained on the teacher's labels.

COCO weights know only "person", so every person becomes a player, the crowd and the benches
included. On 60 snow-game frames, yolo11s at 1280 px found 80% of the teacher's players and
referees (75% from the sideline, 87% from the end zone) and added 5 to 9 other people a frame;
yolo11m found 82% at twice the time. A student (`labelling.train`) knows player and referee.

ultralytics is AGPL: prototype only (pyproject). Runs in the ML environment (.venv-sam3d).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from formation_zero.perception.base import Detections

COCO_PERSON = 0


class YoloDetector:
    def __init__(self, weights: str | Path, *, imgsz: int = 1280, conf: float = 0.25, device: str = "mps",
                 batch: int = 8):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        names = self.model.names
        coco = len(names) == 80 and names[COCO_PERSON] == "person"
        self.keep = {COCO_PERSON: "player"} if coco else {int(i): n for i, n in names.items()}
        self.labels = tuple(dict.fromkeys(self.keep.values()))
        self.imgsz, self.conf, self.device, self.batch = imgsz, conf, device, batch

    def __call__(self, images: list[np.ndarray]) -> list[Detections]:
        out = []
        for i in range(0, len(images), self.batch):
            # ultralytics reads numpy images as OpenCV BGR; ours are RGB
            chunk = [np.ascontiguousarray(im[..., ::-1]) for im in images[i:i + self.batch]]
            for r in self.model.predict(chunk, imgsz=self.imgsz, conf=self.conf, classes=list(self.keep),
                                        device=self.device, verbose=False):
                b = r.boxes
                out.append(Detections(b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float),
                                      np.array([self.keep[int(c)] for c in b.cls.cpu().numpy()], dtype=object)))
        return out


__all__ = ["YoloDetector"]
