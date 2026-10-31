"""Tracks from detections: the ByteTrack wrapper, the per-track label vote, and the per-track summary."""

import numpy as np
import pandas as pd
import pytest

from formation_zero.perception.base import Detections
from formation_zero.perception.track.bytetrack import summarise, vote_labels


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


def test_tracker_keeps_ids_and_never_shares_them():
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


def test_a_player_called_a_referee_for_a_frame_keeps_his_id():
    pytest.importorskip("supervision")
    from formation_zero.perception.track.bytetrack import ByteTracker

    t = ByteTracker(fps=25)
    ids = []
    for i, label in enumerate(["player"] * 4 + ["referee"] + ["player"] * 3):
        out = t.update(Detections(np.array([[10 + i, 10, 30 + i, 60]], float), np.array([0.9]), np.array([label], dtype=object)))
        ids.append((int(out.track_id[0]), out.label[0]) if len(out) else None)
    assert len({i for i, _ in ids[1:]}) == 1 and ids[4][1] == "referee"
    t.cut()
    out = t.update(Detections(np.array([[20, 10, 40, 60]], float), np.array([0.9]), np.array(["player"], dtype=object)))
    assert int(out.track_id[0]) not in {i for i, _ in ids}                 # after a cut, a new id


def test_tracks_take_the_label_most_of_their_boxes_carry():
    df = pd.DataFrame(rows(1, range(5), 50) + rows(1, [5], 50, label="referee", conf=0.9) + rows(2, range(3), 200, label="referee"))
    assert vote_labels(df).groupby("track_id")["label"].unique().map(list).to_dict() == {1: ["player"], 2: ["referee"]}


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

    def cut(self):
        pass

    def update(self, d):
        self.n += 1
        return Detections(d.xyxy, d.confidence, d.label, np.full(len(d), 7))


def test_track_take_keeps_source_frame_numbers_across_batches():
    from formation_zero.perception.tracking import _plays, track_take

    boxes, cuts = track_take(FakeSource(), 100, 111, OneBox(), CountingTracker(), batch=4)
    assert boxes["frame"].tolist() == list(range(100, 111)) and set(boxes["track_id"]) == {7} and cuts == []
    assert _plays("1-3,7") == {1, 2, 3, 7}


class TwoShots(FakeSource):     # a slowly brightening shot, then a cut to a bright one at frame 108
    def iter_frames(self, start, end):
        from formation_zero.ingest.sources import Frame

        for i in range(start, end):
            level = 200 if i >= 108 else 40 + i - start
            yield Frame(index=i, time_s=i / 25, image=np.full((72, 128, 3), level, np.uint8))


def test_a_cut_inside_a_take_ends_every_track_and_one_near_the_end_trims_it():
    from formation_zero.perception.tracking import track_take, trim_span

    tracker = CountingTracker()
    tracker.cut = lambda: setattr(tracker, "cuts", getattr(tracker, "cuts", 0) + 1)
    boxes, cuts = track_take(TwoShots(), 100, 111, OneBox(), tracker, batch=4)
    assert cuts == [108] and tracker.cuts == 1 and len(boxes) == 11
    assert trim_span(100, 111, cuts, fps=25) == (100, 108)                 # 3 frames from the end: trimmed
    assert trim_span(100, 200, cuts, fps=25) == (108, 200)                 # 8 frames from the start: trimmed
    assert trim_span(0, 300, cuts, fps=25) == (0, 300)                     # well inside: kept


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
    out, team_of = vote_teams(Source(), 0, 10, boxes, Teams(), every=5)
    assert team_of == {1: 0, 2: 1, 3: -1} and out["track_id"].equals(boxes["track_id"])


def test_a_lasting_change_of_team_is_an_id_swap_and_a_short_one_is_not():
    from formation_zero.perception.tracking import swap_cuts

    f = np.arange(0, 100, 5)
    one = np.r_[np.full(10, -1.0), np.full(10, 1.0)]
    assert swap_cuts(f, one) == [47]                                    # half-way between frames 45 and 50
    noisy = one.copy(); noisy[[2, 15]] *= -1                            # one doubtful crop each side: still cut
    assert swap_cuts(f, noisy) == [47]
    assert swap_cuts(f, np.r_[np.full(18, -1.0), np.full(2, 1.0)]) == []          # 2 crops is no swap
    assert swap_cuts(f, np.r_[np.full(10, -1.0), np.full(10, 0.4)]) == []         # too weak on one side
    assert swap_cuts(f, np.r_[np.full(4, -1.0), np.full(12, 1.0), np.full(4, -1.0)]) == [17, 77]  # and back
    assert swap_cuts(f, np.r_[np.full(6, -1.0), np.full(7, 1.0), np.full(7, -1.0)]) == []   # no one split fits


def test_vote_teams_gives_the_part_after_a_swap_a_new_id():
    from formation_zero.perception.tracking import vote_teams

    class Teams:      # track 1's box is narrow (team 0) until frame 50, then wide (team 1)
        def margin(self, crops):
            return np.array([-1.0 if c.shape[1] == 10 else 1.0 for c in crops])

    class Source(FakeSource):
        def iter_frames(self, start, end, step=1):
            from formation_zero.ingest.sources import Frame

            for i in range(start, end, step):
                yield Frame(index=i, time_s=i / 25, image=np.zeros((80, 120, 3), np.uint8))

    boxes = pd.DataFrame([{"frame": f, "track_id": t, "label": "player", "conf": 0.9, "x1": 0, "y1": 0,
                           "x2": (10 if f < 50 else 30) if t == 1 else 30, "y2": 40, "off_field": t == 7}
                          for f in range(100) for t in (1, 2, 7)])
    out, team_of = vote_teams(Source(), 0, 100, boxes, Teams(), every=5)
    assert team_of == {1: 0, 8: 1, 2: 1}                                # 7 is off the field: no vote, no cut
    assert set(out.loc[out["frame"] < 47, "track_id"]) == {1, 2, 7}
    assert set(out.loc[out["frame"] >= 47, "track_id"]) == {8, 2, 7}


def test_off_field_boxes_are_grey_and_have_no_id():
    from formation_zero.perception.overlay import OFF_FIELD, draw

    img = np.zeros((36, 64, 3), np.uint8)
    row = {"track_id": 3, "label": "player", "x1": 5, "y1": 5, "x2": 15, "y2": 30}
    out = draw(img, pd.DataFrame([{**row, "off_field": True}]), {3: 1})
    assert set(map(tuple, out.reshape(-1, 3))) == {(0, 0, 0), OFF_FIELD}


class FrameSource(FakeSource):
    def iter_frames(self, start, end, step=1):
        from formation_zero.ingest.sources import Frame

        for i in range(start, end, step):
            yield Frame(index=i, time_s=i / 25, image=np.zeros((80, 120, 3), np.uint8))


def standing(t, label, foot_y, x=10):
    return [{"frame": f, "track_id": t, "label": label, "conf": 0.9, "x1": x, "y1": foot_y - 15, "x2": x + 8, "y2": foot_y}
            for f in range(40)]


def fake_field(monkeypatch, mask, fit):
    import formation_zero.perception.field.homography as hg
    from formation_zero.perception.field import hashes

    monkeypatch.setattr(hashes, "find", lambda image, angle="sideline": {"det": {"mask": mask}})
    monkeypatch.setattr(hg, "register", lambda image, angle="sideline", found=None, prior=None: {"fit": fit})


def test_without_registration_players_off_the_mask_are_voted_off_and_officials_stay(monkeypatch):
    from formation_zero.perception.tracking import vote_field

    mask = np.zeros((80, 120), bool)
    mask[30:, :] = True                                   # the field is the lower part of the frame
    fake_field(monkeypatch, mask, None)
    boxes = pd.DataFrame(standing(1, "player", 20) + standing(2, "referee", 20, 30) + standing(3, "player", 60, 50)
                         + standing(4, "player", 60, 70) + standing(5, "player", 60, 90))
    assert vote_field(FrameSource(), 0, 40, boxes, "sideline", every=10) == {1}
    assert vote_field(FrameSource(), 0, 40, boxes, "endzone", every=10) == set()


def test_registered_frames_judge_feet_in_yards_even_where_the_mask_disagrees(monkeypatch):
    from formation_zero.perception.tracking import vote_field

    # image row v is field y 59 - 0.8 v: row 5 is 55 yd (past the far sideline), row 79.8 is -4.8 (the near bench)
    fit = {"easy": True, "G": np.array([[1.0, 0, 0], [0, -0.8, 59], [0, 0, 1]]),
           "camera": {"C": np.array([0.0, -87.0, 27.0]), "orth": 90.0, "f": 1000.0}}
    fake_field(monkeypatch, np.ones((80, 120), bool), fit)
    boxes = pd.DataFrame(standing(1, "player", 5) + standing(2, "referee", 5, 30) + standing(3, "player", 40, 50)
                         + standing(4, "player", 79.8, 70) + standing(5, "player", 72, 90))
    # 4 is cut off by the frame bottom, which is already past the near sideline; 5 stands at 1.4 yd, on
    assert vote_field(FrameSource(), 0, 40, boxes, "sideline", every=10) == {1, 4}
