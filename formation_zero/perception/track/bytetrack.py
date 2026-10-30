"""ByteTrack (supervision) over per-frame detections, and the per-track summary the router reads.

One tracker per label, so a referee track never takes over a player's. Association is on boxes
alone; a player hidden for longer than `lost_s` comes back under a new id, which `summarise`
counts as a fragment.

supervision dropped ByteTrack in 0.30 (pyproject pins below it). Runs in the ML environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from formation_zero.perception.base import Detections

LOST_S = 1.2            # how long a lost track waits to be matched again
FRAGMENT_GAP_S = 2.0    # a track starting this soon after another ended, and near it, continues it
FRAGMENT_DIST = 1.5     # "near": within this many box heights


class ByteTracker:
    def __init__(self, fps: float, *, activation: float = 0.25, lost_s: float = LOST_S, match: float = 0.8):
        self.fps, self.activation, self.lost_s, self.match = fps, activation, lost_s, match
        self.reset()

    def reset(self) -> None:
        self._by_label: dict[str, object] = {}
        self._ids: dict[tuple[str, int], int] = {}

    def _tracker(self, label: str):
        import supervision as sv

        if label not in self._by_label:
            self._by_label[label] = sv.ByteTrack(track_activation_threshold=self.activation,
                                                 lost_track_buffer=round(self.lost_s * self.fps),
                                                 minimum_matching_threshold=self.match,
                                                 frame_rate=round(self.fps))
        return self._by_label[label]

    def update(self, detections: Detections) -> Detections:
        """Tracked boxes this frame, with ids unique across labels; boxes ByteTrack holds back are dropped."""
        import supervision as sv

        parts = []
        for label in sorted({str(x) for x in detections.label} | set(self._by_label)):
            mine = detections[detections.label == label] if len(detections) else Detections.empty()
            sd = sv.Detections(xyxy=mine.xyxy.reshape(-1, 4), confidence=mine.confidence,
                               class_id=np.zeros(len(mine), dtype=int))
            out = self._tracker(label).update_with_detections(sd)
            ids = np.array([self._ids.setdefault((label, int(t)), len(self._ids) + 1) for t in out.tracker_id], dtype=int)
            parts.append(Detections(out.xyxy.astype(float), out.confidence.astype(float),
                                    np.full(len(out), label, dtype=object), ids))
        if not parts:
            return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=object), np.zeros(0, dtype=int))
        return Detections(np.concatenate([p.xyxy for p in parts]), np.concatenate([p.confidence for p in parts]),
                          np.concatenate([p.label for p in parts]), np.concatenate([p.track_id for p in parts]))


def summarise(rows: pd.DataFrame, fps: float) -> tuple[pd.DataFrame, dict]:
    """Per track: span, frames seen, coverage, gaps, mean confidence, and a confidence that is
    mean confidence times coverage. Per take: track counts and fragments (a track that starts
    within FRAGMENT_GAP_S of another of its label ending, within FRAGMENT_DIST box heights).

    `rows` has one row per tracked box: frame, track_id, label, conf, x1, y1, x2, y2.
    """
    tracks = []
    for tid, g in rows.sort_values("frame").groupby("track_id"):
        f = g["frame"].to_numpy()
        span = int(f[-1] - f[0] + 1)
        first, last = g.iloc[0], g.iloc[-1]
        tracks.append({"track_id": int(tid), "label": first["label"], "first": int(f[0]), "last": int(f[-1]),
                       "seen": len(f), "coverage": round(len(f) / span, 3), "gaps": int((np.diff(f) > 1).sum()),
                       "mean_conf": round(float(g["conf"].mean()), 3),
                       "confidence": round(float(g["conf"].mean()) * len(f) / span, 3),
                       "start_pt": ((first.x1 + first.x2) / 2, first.y2, first.y2 - first.y1),
                       "end_pt": ((last.x1 + last.x2) / 2, last.y2, last.y2 - last.y1)})
    t = pd.DataFrame(tracks)
    fragments = 0
    if len(t):
        for b in t.itertuples():
            for a in t[(t.label == b.label) & (t["last"] < b.first) & (b.first - t["last"] <= FRAGMENT_GAP_S * fps)].itertuples():
                (ax, ay, ah), (bx, by, _) = a.end_pt, b.start_pt
                if np.hypot(ax - bx, ay - by) <= FRAGMENT_DIST * ah:
                    fragments += 1
                    break
        t = t.drop(columns=["start_pt", "end_pt"])
    take = {"tracks": len(t), "fragments": fragments,
            "by_label": {str(k): int(v) for k, v in t["label"].value_counts().items()} if len(t) else {}}
    return t, take


__all__ = ["ByteTracker", "summarise"]
