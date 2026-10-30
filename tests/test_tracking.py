"""Tracks from detections: the per-label ByteTrack wrapper and the per-track summary."""

import numpy as np
import pandas as pd
import pytest

from formation_zero.perception.base import Detections
from formation_zero.perception.track.bytetrack import summarise


def rows(track_id, frames, x, label="player", h=40.0, conf=0.8):
    return [{"frame": f, "track_id": track_id, "label": label, "conf": conf,
             "x1": x - 10, "y1": 100 - h, "x2": x + 10, "y2": 100.0} for f in frames]


def test_summary_counts_gaps_coverage_and_a_fragment():
    df = pd.DataFrame(rows(1, [0, 1, 2, 5, 6], 50)            # a gap of two frames
                      + rows(2, range(10, 20), 52)              # starts near where 1 ended: a fragment
                      + rows(3, range(20), 400)              # far away: not one
                      + rows(4, range(30, 40), 52, label="referee"))
    tracks, take = summarise(df, fps=25)
    one = tracks.set_index("track_id").loc[1]
    assert one["gaps"] == 1 and one["coverage"] == round(5 / 7, 3) and one["confidence"] == round(0.8 * 5 / 7, 3)
    assert take == {"tracks": 4, "fragments": 1, "by_label": {"player": 3, "referee": 1}}


def test_summary_of_nothing():
    tracks, take = summarise(pd.DataFrame(columns=["frame", "track_id", "label", "conf", "x1", "y1", "x2", "y2"]), 25)
    assert len(tracks) == 0 and take["tracks"] == 0


def test_detections_index_like_arrays():
    d = Detections(np.array([[0, 0, 1, 1], [2, 2, 3, 3]], float), np.array([0.9, 0.5]), np.array(["player", "referee"], dtype=object))
    assert len(d[d.label == "referee"]) == 1 and d[d.confidence > 0.6].label.tolist() == ["player"]


def test_tracker_keeps_ids_per_label_and_never_shares_them():
    pytest.importorskip("supervision")
    from formation_zero.perception.track.bytetrack import ByteTracker

    t = ByteTracker(fps=25)
    seen = []
    for i in range(5):
        d = Detections(np.array([[10 + i, 10, 30 + i, 60], [12 + i, 10, 32 + i, 60]], float), np.array([0.9, 0.9]),
                       np.array(["player", "referee"], dtype=object))
        out = t.update(d)
        seen.append(dict(zip(out.label, out.track_id)))
    assert seen[-1]["player"] != seen[-1]["referee"]
    assert all(s == seen[-1] for s in seen[1:])


class FakeSource:
    fps, width, height = 25.0, 64, 36

    def iter_frames(self, start, end):
        from formation_zero.ingest.sources import Frame

        for i in range(start, end):
            yield Frame(index=i, time_s=i / 25, image=np.zeros((36, 64, 3), np.uint8))


class OneBox:
    labels = ("player",)

    def __call__(self, images):
        return [Detections(np.array([[5.0, 5, 15, 30]]), np.array([0.9]), np.array(["player"], dtype=object)) for _ in images]


class CountingTracker:
    def reset(self):
        self.n = 0

    def update(self, d):
        self.n += 1
        return Detections(d.xyxy, d.confidence, d.label, np.full(len(d), 7))


def test_track_take_keeps_source_frame_numbers_across_batches():
    from formation_zero.perception.tracking import _plays, track_take

    boxes = track_take(FakeSource(), 100, 111, OneBox(), CountingTracker(), batch=4)
    assert boxes["frame"].tolist() == list(range(100, 111)) and set(boxes["track_id"]) == {7}
    assert _plays("1-3,7") == {1, 2, 3, 7}


def test_overlay_draws_without_changing_the_frame_size():
    from formation_zero.perception.overlay import draw

    img = np.zeros((36, 64, 3), np.uint8)
    out = draw(img, pd.DataFrame([{"track_id": 3, "label": "player", "x1": 5, "y1": 5, "x2": 15, "y2": 30}]), {3: 1})
    assert out.shape == img.shape and out.any()


def test_tracks_vote_their_team_and_doubtful_crops_abstain():
    from formation_zero.perception.tracking import vote_teams

    class Teams:      # margin by box width: wide boxes are team 1, middling ones too close to call
        def margin(self, crops):
            return np.array([{10: -1.0, 20: 0.1, 30: 1.0}[c.shape[1]] for c in crops])

    class Source(FakeSource):
        def iter_frames(self, start, end, step=1):
            from formation_zero.ingest.sources import Frame

            for i in range(start, end, step):
                yield Frame(index=i, time_s=i / 25, image=np.zeros((80, 120, 3), np.uint8))

    boxes = pd.DataFrame([{"frame": f, "track_id": t, "label": lab, "conf": 0.9, "x1": 0, "y1": 0, "x2": w, "y2": 40}
                          for f in range(10) for t, w, lab in ((1, 10, "player"), (2, 30, "player"), (3, 20, "player"),
                                                               (4, 30, "referee"))])
    assert vote_teams(Source(), 0, 10, boxes, Teams(), every=5) == {1: 0, 2: 1, 3: -1}
