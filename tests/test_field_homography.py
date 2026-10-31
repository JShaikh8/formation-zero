"""Pixel-to-yards homography and the line-of-scrimmage anchor (GRD-37) on a synthetic camera. The
last two tests paint whole frames and need OpenCV, which the project venv lacks, so they skip there."""

import numpy as np
import pytest

from formation_zero.perception.field import homography as Hg

W, H = 960, 540
CAMERA = np.array([60.0, -86.0, 27.0])        # yd: where the clear game's press-box camera came out


def camera_h(c=CAMERA, target=(52.0, 27.0, 0.0), f=4400.0, w=W, h=H):
    """Field (x, y, 1) -> pixels for a pinhole camera at `c` looking at `target` (z up)."""
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    r = np.vstack([right, np.cross(fwd, right), fwd])
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    p = k @ np.hstack([r, -(r @ c)[:, None]])
    return p[:, [0, 1, 3]] / np.linalg.norm(p[:, [0, 1, 3]])


def test_dlt_recovers_a_homography_from_points_and_yard_line_points():
    hf = camera_h()
    fld = np.array([[x, y] for x in (45.0, 50.0, 55.0) for y in (Hg.HASH_NEAR_Y, Hg.HASH_FAR_Y)])
    img = Hg.apply(hf, fld)
    lp = Hg.apply(hf, np.array([[x, y] for x in (45.0, 50.0, 55.0, 60.0) for y in (5.0, 45.0)]))
    lx = np.repeat([45.0, 50.0, 55.0, 60.0], 2)
    g = Hg.dlt(img, fld, lp, lx)
    probe = np.array([[47.0, 1.0], [58.0, 50.0], [52.5, 26.0]])
    assert np.allclose(Hg.apply(g, Hg.apply(hf, probe)), probe, atol=1e-6)


def test_ransac_leaves_out_points_on_the_wrong_yard():
    hf = camera_h()
    xs = np.arange(44.0, 61.0)
    fld = np.array([[x, y] for x in xs for y in (Hg.HASH_NEAR_Y, Hg.HASH_FAR_Y)])
    img = Hg.apply(hf, fld) + np.random.default_rng(0).normal(0, 0.3, (len(fld), 2))
    wrong = [3, 10, 21]
    fld_bad = fld.copy()
    fld_bad[wrong, 0] += 1.0                                       # a dash put on the next yard
    lines = np.array([[x, y] for x in (45.0, 50.0, 55.0, 60.0) for y in (2.0, 27.0, 50.0)])
    g, pin, lin = Hg.ransac(img, fld_bad, Hg.apply(hf, lines), lines[:, 0], tol=2.5)
    assert np.flatnonzero(~pin).tolist() == wrong and lin.all()
    assert np.abs(Hg.point_errors(g, img, fld)).max() < 1.5


def test_the_camera_comes_back_from_its_homography():
    cam = Hg.camera(camera_h(), (H, W))
    assert abs(cam["f"] - 4400) < 1 and abs(cam["orth"] - 90) < 0.01
    assert np.allclose(cam["C"], CAMERA, atol=0.01) and Hg.plausible(cam)
    assert not Hg.plausible(cam, prior=(-70.0, 27.0))             # the game's camera is elsewhere


def row(hf, y, xs):
    return {"points": Hg.apply(hf, np.column_stack([xs, np.full(len(xs), y)])), "x": np.asarray(xs, float),
            "line": np.cross(*[np.append(p, 1.0) for p in Hg.apply(hf, np.array([[0.0, y], [100.0, y]]))])}


def test_register_names_the_rows_by_where_they_put_the_camera():
    """Far ticks, two hash rows and a junk row: only the true naming puts the camera behind the
    near sideline. x is relative (the leftmost yard line at 0), as `hashes.find` gives it."""
    hf = camera_h() @ np.array([[1, 0, 45.0], [0, 1, 0], [0, 0, 1]])  # relative x 0 = absolute 45
    xs = np.array([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
    rows = [row(hf, Hg.LATERAL[0][1], xs), row(hf, Hg.HASH_FAR_Y, xs[::2]), row(hf, Hg.HASH_NEAR_Y, xs[1:]),
            row(hf, 12.0, xs[:6])]                                 # dashes where no dashes are painted
    yard = np.vstack([np.concatenate(Hg.apply(hf, np.array([[x, 2.0], [x, 50.0]])))
                      for x in (0.0, 5.0, 10.0, 15.0)])
    alt = {"yard": yard, "x": np.array([0.0, 5.0, 10.0, 15.0]), "q": None, "rows": rows}
    reg = Hg.register(np.zeros((H, W, 3), np.uint8), found={"alternatives": [alt]})
    assert [n for _, n, _ in reg["rows"]] == ["tick_far", "hash_far", "hash_near"]
    assert reg["fit"]["easy"] and reg["fit"]["rms_px"] < 0.1
    assert np.allclose(reg["fit"]["camera"]["C"], CAMERA - [45, 0, 0], atol=0.05)


def test_the_line_of_scrimmage_lies_between_the_two_teams():
    rng = np.random.default_rng(2)
    los = 37.0
    off = np.concatenate([los - rng.uniform(0.8, 1.5, 7), los - rng.uniform(3, 7, 4)])     # line, backs
    dfn = np.concatenate([los + rng.uniform(0.8, 1.5, 4), los + rng.uniform(4, 15, 7)])   # line, the rest
    x = np.concatenate([off, dfn, [los + 6.0]])                   # one offensive shirt read as defense
    offense = np.concatenate([np.ones(11, bool), np.zeros(11, bool), [False]])
    offense[-1] = True
    for flip in (1, -1):
        r = Hg.line_of_scrimmage(flip * x, offense)
        assert r["attack"] == flip and r["misplaced"] == 1 and abs(r["x"] - flip * los) < 0.5


def test_the_anchor_shifts_by_whole_5_yard_steps():
    # offense facing +x at its own 23 (yardline_100 77): the line is at absolute 33
    a = Hg.anchor(los_rel=8.4, attack=1, yardline_100=77)
    assert a["los_x"] == 33 and a["shift"] == 25 and abs(a["residual"] + 0.4) < 1e-9
    # facing -x, 23 yd from the goal it attacks (the left one, at 10): absolute 33 again
    assert Hg.anchor(los_rel=-1.6, attack=-1, yardline_100=23)["shift"] == 35
    g = Hg.to_absolute(np.eye(3), 25.0)
    assert np.allclose(Hg.apply(g, np.array([[8.4, 20.0]])), [[33.4, 20.0]])


def paint_frame(hf):
    """Turf with yard lines, sideline bands, hash marks and sideline ticks drawn through hf."""
    import cv2

    rng = np.random.default_rng(3)
    img = np.clip(np.array([50, 110, 50], float) + rng.normal(0, 5, (H, W, 1)), 0, 255).astype(np.uint8)

    def quad(x0, x1, y0, y1):
        p = Hg.apply(hf, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        cv2.fillPoly(img, [np.round(p * 8).astype(np.int32)], (235, 235, 235), cv2.LINE_AA, shift=3)

    for x in np.arange(20.0, 101.0, 5.0):
        quad(x - 0.06, x + 0.06, 0, 53.333)
    quad(0, 120, 53.333, 55.333)
    quad(0, 120, -2, 0)
    for x in np.arange(20.0, 101.0):
        if x % 5:
            for y0, y1 in ((22.92, 23.58), (29.75, 30.42), (0.11, 0.78), (52.56, 53.22)):
                quad(x - 0.06, x + 0.06, y0, y1)
    return img


def test_a_painted_sideline_frame_registers_up_to_a_5_yard_shift():
    pytest.importorskip("cv2")
    hf = camera_h()
    reg = Hg.register(paint_frame(hf), "sideline")
    fit = reg["fit"]
    assert fit is not None and fit["easy"]
    assert {"hash_far", "hash_near"} <= {n for _, n, _ in reg["rows"]}
    probe = np.array([[40.0, 5.0], [52.0, 26.5], [63.0, 48.0]])
    rel = Hg.apply(fit["G"], Hg.apply(hf, probe))
    shift = probe[:, 0] - rel[:, 0]
    assert np.allclose(shift, shift[0], atol=0.15) and abs(shift[0] / 5 - round(shift[0] / 5)) < 0.03
    assert np.allclose(rel[:, 1], probe[:, 1], atol=0.5)            # 0.3 yd off at the far side
    assert abs(fit["camera"]["C"][1] - CAMERA[1]) < 3 and abs(fit["camera"]["C"][2] - CAMERA[2]) < 1.5


def test_a_painted_end_zone_frame_registers_too():
    """From behind an end zone the yard lines run across the picture and the hash rows are columns;
    either end will do, so the rows are named in both orders and the camera picks."""
    pytest.importorskip("cv2")
    c = np.array([-30.0, 26.7, 30.0])
    hf = camera_h(c=c, target=(32.0, 26.7, 0.0), f=1500.0)
    reg = Hg.register(paint_frame(hf), "endzone")
    fit = reg["fit"]
    assert reg["found"]["det"]["angle"] == "endzone" and fit is not None and fit["easy"]
    probe = np.array([[25.0, 10.0], [32.0, 26.0], [40.0, 45.0]])
    rel = Hg.apply(fit["G"], Hg.apply(hf, probe))
    # the relative frame may run either way along the field; across it, y may be mirrored
    sx = np.sign(np.polyfit(probe[:, 0], rel[:, 0], 1)[0])
    shift = rel[:, 0] - sx * probe[:, 0]
    assert np.allclose(shift, shift[0], atol=0.2) and abs(shift[0] / 5 - round(shift[0] / 5)) < 0.04
    y = rel[:, 1] if sx > 0 else 53.333 - rel[:, 1]
    assert np.allclose(y, probe[:, 1], atol=0.5)


def test_boxes_cut_off_by_the_bottom_of_the_frame_have_no_feet():
    boxes = np.array([[100, 500, 140, 600], [300, 980, 340, 1080], [500, 990, 540, 1077.5]])
    pts, ok = Hg.feet(boxes, (1080, 1920))
    assert np.allclose(pts[0], [120, 600]) and ok.tolist() == [True, False, False]


def project(pts3, c=CAMERA, target=(52.0, 27.0, 0.0), f=4400.0, w=W, h=H):
    """Field points (n, 3; z up) -> pixels, for the camera of `camera_h`."""
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    r = np.vstack([right, np.cross(fwd, right), fwd])
    p = (np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]) @ (r @ (np.asarray(pts3, float) - c).T)).T
    return p[:, :2] / p[:, 2:]


def test_with_the_camera_a_box_whose_head_needs_feet_below_the_frame_is_cut_off():
    g = np.linalg.inv(camera_h())
    near = Hg.apply(g, np.array([[480.0, H - 1.0]]))[0]                # the field at the frame's bottom
    out, inside = near + [0, -2.0], near + [0, 2.0]                    # feet 2 yd below it, and 2 yd above
    boxes = []
    for x, y in (out, inside):
        (hx, hy), (_, fy) = project([[x, y, 2.0], [x, y, 0.0]])
        boxes.append([hx - 10, hy, hx + 10, min(fy, H - 15.0)])        # the bench's box stops 15 px short
    assert Hg.feet(np.array(boxes), (H, W))[1].tolist() == [True, True]
    assert Hg.feet(np.array(boxes), (H, W), g, CAMERA)[1].tolist() == [False, True]


def test_an_anchor_near_the_rounding_edge_is_not_trusted():
    assert not Hg.anchor(los_rel=1.7, attack=1, yardline_100=26)["ok"]       # 2.3 yd off: could be a step out
    assert Hg.anchor(los_rel=3.6, attack=1, yardline_100=26)["ok"]
