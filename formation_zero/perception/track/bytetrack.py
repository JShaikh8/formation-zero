"""ByteTrack (supervision) over per-frame detections, and the per-track summary the router reads.

Players and referees share one tracker, so a player the detector calls a referee for a few frames
keeps his id; `vote_labels` then gives each track the label most of its boxes carry. The ball, if
detected, has its own. A player hidden for longer than `lost_s` comes back under a new id, which
`summarise` counts as a fragment.

Settings measured on five 49ers-Rams takes (2026-10-30, round-0 student): boxes down to 0.1 kept
for ByteTrack's second, low-score match and tracks started at 0.5 (the teacher's cut) halved the
track count against 0.25 for both; matching at IoU 0.1 instead of 0.2 cut restarts beside a just-
ended track by two thirds (small boxes in a moving camera); one tracker for both labels and a longer
lost buffer changed little. BoT-SORT's camera-motion compensation (ultralytics) did no better.
About half the remaining track starts and ends are people leaving the frame as the camera moves.

Rechecked with round-1 students, which score real players lower, on every third of plays 1-60
(2026-10-31): starting at 0.25 moved frames with 21-23 players before the snap from 82% to 86%, for
35% more sideline tracks and 49% more fragments. 0.5 stays.

The two jobs that one line does cannot be split usefully either (2026-11-01). Replayed on 20 takes of
Chargers-Chiefs with new tracks still starting at 0.6, letting boxes down to 0.25, 0.15 or 0.1 follow a
track left flickers about the same or worse (444 to 482, 515, 565) and restarts about the same (114 to
102, 97, 98). supervision multiplies IoU by score in its first match and asks IoU 0.5 in its second,
both hard-coded, so a weak box needs a tight fit whichever pass it lands in. Lost players are
rejoined after tracking instead (`track.stitch`).

supervision dropped ByteTrack in 0.30 (pyproject pins below it). Runs in the ML environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from formation_zero.perception.base import Detections

LOST_S = 2.5            # how long a lost track waits to be matched again
ACTIVATION = 0.5        # boxes above this match first and may start tracks; down to DETECT_CONF match second
MATCH = 0.9             # supervision's matching threshold is 1 - IoU: 0.9 accepts IoU of 0.1
DETECT_CONF = 0.1       # the detector confidence a tracking run asks for
GROUPS = {"player": "people", "referee": "people"}      # labels tracked together
FRAGMENT_GAP_S = 2.0    # a track starting this soon after another ended, and near it, continues it
FRAGMENT_DIST = 1.5     # "near": within this many box heights


class ByteTracker:
    def __init__(self, fps: float, *, activation: float = ACTIVATION, lost_s: float = LOST_S, match: float = MATCH,
                 groups: dict[str, str] = GROUPS):
        self.fps, self.activation, self.lost_s, self.match, self.groups = fps, activation, lost_s, match, groups
        self.reset()

    def reset(self) -> None:
        self._by_group: dict[str, object] = {}
        self._ids: dict[tuple[str, int], int] = {}
        self._next = 1

    def cut(self) -> None:
        """A cut inside the take: every track ends here, and later ids never repeat earlier ones."""
        self._by_group, self._ids = {}, {}

    def _tracker(self, group: str):
        import supervision as sv

        if group not in self._by_group:
            self._by_group[group] = sv.ByteTrack(track_activation_threshold=self.activation,
                                                 lost_track_buffer=round(self.lost_s * self.fps),
                                                 minimum_matching_threshold=self.match,
                                                 frame_rate=round(self.fps))
        return self._by_group[group]

    def update(self, detections: Detections) -> Detections:
        """Tracked boxes this frame, each with its detected label and an id unique across groups;
        boxes ByteTrack holds back are dropped."""
        import supervision as sv

        group_of = np.array([self.groups.get(str(x), str(x)) for x in detections.label], dtype=object)
        parts = []
        for group in sorted(set(group_of) | set(self._by_group)):
            mine = detections[group_of == group] if len(detections) else Detections.empty()
            sd = sv.Detections(xyxy=mine.xyxy.reshape(-1, 4), confidence=mine.confidence,
                               class_id=np.arange(len(mine)))          # class_id carries the row back
            out = self._tracker(group).update_with_detections(sd)
            ids = np.array([self._id(group, int(t)) for t in out.tracker_id], dtype=int)
            parts.append(Detections(out.xyxy.astype(float), out.confidence.astype(float),
                                    mine.label[out.class_id] if len(out) else np.zeros(0, dtype=object), ids))
        if not parts:
            return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=object), np.zeros(0, dtype=int))
        return Detections(np.concatenate([p.xyxy for p in parts]), np.concatenate([p.confidence for p in parts]),
                          np.concatenate([p.label for p in parts]), np.concatenate([p.track_id for p in parts]))


    def _id(self, group: str, t: int) -> int:
        if (group, t) not in self._ids:
            self._ids[group, t], self._next = self._next, self._next + 1
        return self._ids[group, t]


def vote_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """Each track's boxes all take the label with the most summed confidence among them."""
    if rows.empty:
        return rows
    label_of = rows.groupby(["track_id", "label"])["conf"].sum().reset_index().sort_values("conf") \
        .drop_duplicates("track_id", keep="last").set_index("track_id")["label"]
    return rows.assign(label=rows["track_id"].map(label_of))


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


__all__ = ["DETECT_CONF", "ByteTracker", "summarise", "vote_labels"]
