"""Bird's-eye positions (milestone 4) on a synthetic panning camera: the 5-yd step between relative
frames, carrying and blending homographies through a take, the snap, the anchor, and the field
drawing's size. All numpy (PIL for the drawing); the camera-motion tracker needs OpenCV and is
exercised on film only."""

import numpy as np
import pandas as pd
import pytest

from formation_zero.configs.nfl_field import FIELD_WIDTH_YD
from formation_zero.perception.field import homography as hg
from formation_zero.perception.field import topdown as td

W, H = 1920, 1080
CAMERA = np.array([60.0, -87.3, 27.2])


def camera_h(target_x=52.0, f=4400.0, c=CAMERA):
    """Field (x, y, 1) -> pixels for the press-box camera looking at (target_x, 27, 0)."""
    fwd = np.array([target_x, 27.0, 0.0]) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    r = np.vstack([right, np.cross(fwd, right), fwd])
    k = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    p = k @ np.hstack([r, -(r @ c)[:, None]])
    return p[:, [0, 1, 3]] / np.linalg.norm(p[:, [0, 1, 3]])


def take(n=40, pan=0.3, zoom=1.002):
    """A take of n frames: the camera pans `pan` yd a frame and zooms in slowly. Returns the true
    image -> field homographies and the frame-to-frame motions (px of a frame -> px of the one before)."""
    hs = [camera_h(48.0 + pan * t, 4400.0 * zoom ** t) for t in range(n)]
    gs = [np.linalg.inv(h) for h in hs]
    motion = {t: hs[t - 1] @ np.linalg.inv(hs[t]) for t in range(1, n)}
    return gs, motion


def field_of(g, pts):
    return hg.apply(g, pts)


def test_step_offset_finds_the_whole_step_and_how_sure_it_is():
    g = np.linalg.inv(camera_h())
    pts = td.probe_points((H, W))
    st = td.step_offset(g, td.shift(g, -15.0), pts)
    assert st["dx"] == 15.0 and st["left"] < 1e-6 and st["dy"] < 1e-6
    st = td.step_offset(g, td.shift(g, -11.0), pts)          # 11 yd: nearest step 10, 1 yd left over
    assert st["dx"] == 10.0 and st["left"] == pytest.approx(1.0, abs=1e-6)


def test_carry_follows_the_camera_forward_and_back():
    gs, motion = take()
    pts = td.probe_points((H, W))
    fwd = td.carry(gs[0], [motion[t] for t in range(1, 20)])
    assert np.allclose(field_of(fwd[-1], pts), field_of(gs[19], pts), atol=1e-6)
    back = td.carry(gs[19], [motion[t] for t in range(1, 20)], forward=False)
    assert np.allclose(field_of(back[-1], pts), field_of(gs[0], pts), atol=1e-6)
    assert td.carry(gs[0], [motion[1], None]) is None


def test_blend_of_two_registrations_lands_near_the_frame_between():
    gs, _ = take(pan=0.4)
    pts = td.probe_points((H, W))
    mid = td.blend(gs[0], gs[10], 0.5, pts)
    err = np.hypot(*(field_of(mid, pts) - field_of(gs[5], pts)).T)
    assert err.max() < 0.15           # yd: a smooth pan is nearly linear in field position over 10 frames
    assert td.blend(gs[0], gs[10], 0.0, pts) is gs[0]


def test_bridge_puts_registrations_in_one_frame_and_fills_between():
    gs, motion = take(n=40)
    rng = np.random.default_rng(1)
    offsets = {t: 5.0 * rng.integers(-3, 4) for t in range(0, 40, 5)}
    offsets[10] = 2.0 + offsets[10]                    # this one is not a whole step off: dropped
    regs = {t: td.shift(gs[t], -o) for t, o in offsets.items() if t != 35}     # 35 did not register
    out = td.bridge(list(range(40)), regs, motion, (H, W), fps=25, extrapolate_s=0.2)
    assert out["dropped"] == [10]
    pts = td.probe_points((H, W))
    base = offsets[0]
    for t in range(36):
        assert np.allclose(field_of(out["G"][t], pts), field_of(gs[t], pts) - [base, 0], atol=1e-4), t
    assert out["source"][5] == "direct" and out["source"][7] == "bridged" and out["source"][10] == "bridged"
    assert out["source"][32] == "extrapolated" and 36 not in out["G"]        # 0.2 s past frame 30 = 5 frames
    assert all(s["left"] < 1e-6 for s in out["steps"])


def test_bridge_without_motion_interpolates():
    gs, motion = take(n=11, pan=0.2)
    motion[6] = None
    regs = {0: gs[0], 10: td.shift(gs[10], -5.0)}
    out = td.bridge(list(range(11)), regs, motion, (H, W), fps=25)
    assert out["steps"][0]["via"] == "still" and out["steps"][0]["dx"] == 5.0
    assert out["source"][5] == "interpolated"
    pts = td.probe_points((H, W))
    assert np.hypot(*(field_of(out["G"][5], pts) - field_of(gs[5], pts)).T).max() < 0.1


def test_take_camera_is_where_the_press_box_camera_stands():
    gs, _ = take(n=6)
    c = td.take_camera(dict(enumerate(gs)), {t: "direct" if t % 2 == 0 else "bridged" for t in range(6)}, (H, W))
    assert np.allclose(c, CAMERA, atol=0.05)
    assert td.take_camera(dict(enumerate(gs)), {}, (H, W)) is None


def positions_take(snap=30, n=60, fps=25.0):
    """Eleven a side lined up either side of x = 40 (offense team 0 to the left), still until the
    snap, then everyone runs at 5 yd/s."""
    rows = []
    rng = np.random.default_rng(0)
    xs = np.concatenate([40 - rng.uniform(0.8, 8, 11), 40 + rng.uniform(0.8, 8, 11)])
    ys = rng.uniform(10, 43, 22)
    for f in range(n):
        run = max(0, f - snap) / fps * 5.0
        for i in range(22):
            rows.append({"frame": f, "track_id": i, "label": "player", "team": int(i >= 11),
                         "x_yd": xs[i] + run, "y_yd": ys[i]})
    return pd.DataFrame(rows)


def test_snap_and_anchor():
    pos = td.smooth_tracks(positions_take())
    snap = td.snap_frame(pos, 25.0)
    assert 29 <= snap <= 32
    # relative frame 25 yd behind the absolute one; the offense (team 0) attacks +x from its own 30
    a = td.anchor_take(pos.assign(x_yd=pos["x_yd"] - 25), 0, 70.0, snap, 25.0)
    assert a["shift"] == 25.0 and a["agree"] == 1.0 and a["attack"] == 1 and abs(a["residual"]) < 0.5
    assert td.snap_frame(td.smooth_tracks(positions_take(snap=-10)), 25.0) is None     # moving from the start


def test_positions_leave_out_feet_below_the_frame_and_off_field_tracks():
    g = np.linalg.inv(camera_h())
    boxes = pd.DataFrame({"frame": [3, 3, 3], "track_id": [1, 2, 3], "label": ["player", "player", "referee"],
                          "x1": [900.0, 500.0, 1200.0], "y1": [500.0, 1000.0, 400.0], "x2": [940.0, 540.0, 1230.0],
                          "y2": [600.0, 1079.0, 480.0], "off_field": [False, False, False]})
    boxes.loc[3] = [3, 4, "player", 100.0, 300.0, 120.0, 350.0, True]
    out = td.positions(boxes, {3: g}, {3: "direct"}, (H, W), team_of={1: 0})
    assert list(out["track_id"]) == [1, 3]
    assert list(out["team"]) == [0, -1] and out["easy"].all()
    assert np.allclose(out[["x_yd", "y_yd"]].iloc[0], hg.apply(g, np.array([[920.0, 600.0]]))[0])


def test_field_drawing_is_to_scale():
    f = td.Field(ppy=10, margin=4)
    assert (f.width, f.height) == (1280, 613)
    assert f.px(0, FIELD_WIDTH_YD) == (40.0, 40.0)
    x, y = f.px(60, 0)
    assert (x, round(float(y), 6)) == (640.0, round(40 + FIELD_WIDTH_YD * 10, 6))
    pytest.importorskip("PIL")
    img = f.base()
    assert img.shape == (613, 1280, 3)
    col = img[300, int(f.px(60, 0)[0])]                     # the 50-yard line is painted white
    assert col.min() > 200
    one = f.draw(pd.DataFrame({"x_yd": [60.0], "y_yd": [26.7], "label": ["player"], "team": [0], "track_id": [7]}),
                 los_x=45.0)
    cx, cy = (round(v) for v in f.px(60.0, 26.7))
    assert tuple(one[cy, cx]) == td.TEAM_COLOURS[0]
    assert tuple(one[300, int(f.px(45.0, 0)[0])]) == td.LOS_COLOUR


def test_team_club_from_the_anchor_residual():
    # offense (team 0) lines up at absolute x 38 attacking +x from its own 28 (yardline_100 72); the
    # relative frame is 25 yd behind. With the clubs swapped the official line would be at 82.
    pos = td.smooth_tracks(positions_take()).assign(x_yd=lambda d: d["x_yd"] - 27)
    take = {"play_uid": "u1", "snap": td.snap_frame(pos, 25.0), "pos": pos, "fps": 25.0}
    pbp = pd.DataFrame({"posteam": ["AWY"], "yardline_100": [72.0], "play_type": ["pass"]}, index=["u1"])
    club, score = td.choose_team_club([take], pbp, "AWY", "HOM")
    assert club == {0: "AWY", 1: "HOM"}
    assert score["AWY"][0] < 0.5 < score["HOM"][0]


def test_smoothing_never_averages_across_a_gap():
    frames = list(range(10)) + list(range(12, 17))           # frames 10 and 11 missing
    x = [0.1 * f for f in range(10)] + [30 + 0.1 * (f - 12) for f in range(12, 17)]
    pos = pd.DataFrame({"frame": frames, "track_id": 1, "label": "player", "x_yd": x, "y_yd": 20.0})
    s = td.smooth_tracks(pos).set_index("frame")
    assert s.loc[9, "x_yd_s"] == pytest.approx(0.8) and s.loc[12, "x_yd_s"] == pytest.approx(30.1)
    assert np.isnan(s.loc[12, "speed"])                      # 3 frames after the last: no speed
    assert s.loc[5, "speed"] == pytest.approx(0.1 * 25)
