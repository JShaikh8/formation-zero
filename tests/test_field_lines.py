"""Field lines (GRD-37) on synthetic frames. The geometry is numpy only; the rest needs OpenCV,
which the project venv lacks, so those tests skip there."""

import numpy as np
import pytest

from formation_zero.perception.field import lines as L

W, H = L.PROC_WIDTH, 540
TURF = (50, 110, 50)


def paint(img, x1, y1, x2, y2, width, colour=(235, 235, 235)):
    """Draw a straight line of `width` px into img, with numpy only."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.array([x2 - x1, y2 - y1], float)
    t = np.clip(((xx - x1) * d[0] + (yy - y1) * d[1]) / (d @ d), 0, 1)
    dist = np.hypot(xx - (x1 + t * d[0]), yy - (y1 + t * d[1]))
    img[dist <= width / 2] = colour
    return img


def turf(seed=0, colour=TURF, noise=6):
    rng = np.random.default_rng(seed)
    base = np.array(colour, float) + rng.normal(0, noise, (H, W, 1))
    return np.clip(base, 0, 255).astype(np.uint8)


def sideline_frame():
    """Seen from the sideline: five yard lines converging on a point above the frame, stopping at a
    far sideline band near the top, with a busy crowd beyond it."""
    img = turf()
    rng = np.random.default_rng(1)
    crowd = rng.integers(0, 255, (100 // 6 + 1, W // 6 + 1, 3)).repeat(6, 0).repeat(6, 1)[:100, :W]
    img[:100] = crowd
    vp = np.array([480.0, -900.0])
    for xb in (80, 260, 440, 620, 800):
        top = vp + (np.array([xb, H - 1.0]) - vp) * (120 - vp[1]) / (H - 1 - vp[1])
        paint(img, xb, H - 1, top[0], top[1], 4)
    paint(img, 0, 118, W - 1, 112, 9)
    return img, vp


def endzone_frame(n=8):
    img = turf(seed=2)
    for y in np.linspace(50, 500, n):
        paint(img, 0, y, W - 1, y + 4, 4)
    return img


# ---------------------------------------------------------------- geometry (numpy only)


def test_vanishing_point_of_converging_segments_ignores_strays():
    vp = np.array([500.0, -1000.0])
    segs = []
    for xb in (100, 300, 500, 700, 900):
        a = np.array([xb, 500.0])
        b = a + (vp - a) * 0.3
        segs.append([*a, *b])
    segs += [[0, 0, 200, 50], [600, 400, 700, 380]]
    v, inl = L.vanishing_point(np.array(segs))
    assert np.allclose(v[:2] / v[2], vp, atol=1.0)
    assert inl.tolist() == [True] * 5 + [False] * 2


def test_parallel_segments_meet_at_infinity():
    segs = np.array([[0, y, 400, y] for y in (50.0, 150.0, 250.0)])
    v, inl = L.vanishing_point(segs)
    assert abs(v[2]) < 1e-6 and inl.all()


def test_broken_pieces_of_one_line_become_one_line_and_a_neighbour_stays_apart():
    segs = np.array([[0, 100, 150, 110], [200, 113, 350, 123], [400, 127, 600, 140],   # one line, gaps
                     [0, 130, 300, 150]])                                                # 20+ px below
    out = L.collinear(segs, min_px=50)
    assert len(out) == 2
    first = out[0]
    assert min(first[0], first[2]) < 1 and max(first[0], first[2]) > 599


def test_the_two_edges_of_a_thick_line_merge():
    segs = np.array([[0, 100, 500, 100], [0, 104.5, 400, 104.5]])
    assert len(L.collinear(segs, tol=2.0, min_px=50)) == 1


# ---------------------------------------------------------------- on pixels (OpenCV)


def test_a_sideline_frame_gives_converging_yard_lines_the_far_sideline_and_a_mask():
    pytest.importorskip("cv2")
    img, vp = sideline_frame()
    res = L.detect(img, "sideline")
    assert res["angle"] == "sideline" and len(res["yard"]) == 5
    assert np.allclose(res["vanishing_point"][:2], vp, atol=40)
    far = res["boundary"]["far"]
    assert far is not None and abs((far[1] + far[3]) / 2 - 115) < 8
    assert res["boundary"]["near"] is None                  # the yard lines run off the bottom
    mask = res["mask"]
    assert mask[400, 480] and mask[150, 480] and not mask[60, 480]


def test_the_crowd_is_not_a_family_of_lines():
    pytest.importorskip("cv2")
    img, _ = sideline_frame()
    pix = L.line_pixels(img)
    fam = L.families(L.segments(pix), "sideline", pix)
    for y in fam["yard"]:
        assert min(y[1], y[3]) > 90                         # every yard line starts below the crowd


def test_end_zone_lines_are_flat_and_leave_the_whole_frame_in():
    pytest.importorskip("cv2")
    res = L.detect(endzone_frame(), "endzone")
    assert res["angle"] == "endzone" and len(res["yard"]) == 8
    assert all(v is None for v in res["boundary"].values())
    assert res["mask"].all()


def test_a_clear_end_zone_picture_overrules_a_sideline_label():
    pytest.importorskip("cv2")
    res = L.detect(endzone_frame(n=L.OVERRIDE_LINES + 1), "sideline")
    assert res["angle"] == "endzone"


def test_faint_lines_on_a_noisy_white_field_are_found():
    """Snow: lines a few grey levels above a speckled white field, which the plain top-hat misses."""
    cv2 = pytest.importorskip("cv2")
    img = turf(seed=3, colour=(185, 190, 180), noise=0)
    rng = np.random.default_rng(4)
    speckle = cv2.GaussianBlur(rng.normal(0, 18, (H, W)).astype(np.float32), (0, 0), 1.0)
    vp = np.array([480.0, -900.0])
    lines = np.zeros((H, W, 3), np.uint8)
    for xb in (100, 280, 460, 640, 820):
        top = vp + (np.array([xb, H - 1.0]) - vp) * (0 - vp[1]) / (H - 1 - vp[1])
        paint(lines, xb, H - 1, top[0], top[1], 5, colour=(14, 14, 14))
    img = np.clip(img.astype(float) + speckle[..., None] + lines, 0, 255).astype(np.uint8)
    res = L.detect(img, "sideline")
    assert len(res["yard"]) >= 4 and res["mask"] is not None


def test_bare_turf_has_no_mask():
    pytest.importorskip("cv2")
    assert L.field_mask(turf(seed=5), "sideline") is None


def test_masks_come_back_at_the_frame_size():
    pytest.importorskip("cv2")
    img, _ = sideline_frame()
    big = np.repeat(np.repeat(img, 2, 0), 2, 1)              # 1920 x 1080, like the snow game
    mask = L.field_mask(big, "sideline")
    assert mask.shape == big.shape[:2] and mask[800, 960] and not mask[120, 960]
