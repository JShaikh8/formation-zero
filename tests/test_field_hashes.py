"""Hash rows and the yard-line pencil (GRD-37) on synthetic geometry. Only `dash_candidates` needs
OpenCV, which the project venv lacks, so that test skips there."""

import numpy as np
import pytest

from formation_zero.perception.field import hashes as Hs

# field (x, y, 1) -> 960 x 540 pixels, roughly a zoomed sideline view: yard lines rake toward a
# point above the frame, lines along the field are near flat
H = np.array([[38.0, -9.0, 120.0], [0.4, -9.5, 520.0], [0.0004, 0.012, 1.0]])


def project(points):
    p = np.column_stack([points, np.ones(len(points))]) @ H.T
    return p[:, :2] / p[:, 2:]


def yard_line(x, y0=2.0, y1=50.0):
    return np.concatenate(project(np.array([[x, y0], [x, y1]])))


def test_the_pencil_gives_the_field_x_of_any_point():
    xs = np.array([0.0, 5.0, 10.0, 15.0])
    q = Hs.pencil(np.array([yard_line(x) for x in xs]), xs)
    pts = np.random.default_rng(0).uniform([0, 0], [15, 53], (50, 2))
    assert np.allclose(Hs.field_x(q, project(pts)), pts[:, 0], atol=1e-6)


def test_a_missing_yard_line_is_a_10_yard_gap_and_a_stray_line_is_left_out():
    xs = [0.0, 5.0, 15.0, 20.0, 25.0]                    # the 10 is hidden behind players
    yard = np.array([yard_line(x) for x in xs] + [yard_line(11.8, 10, 30)])
    best = Hs.yard_indexings(yard, 270.0)[0]
    assert sorted(best["keep"].tolist()) == [0, 1, 2, 3, 4]
    assert best["x"].tolist() == xs


def test_too_few_yard_lines_cannot_be_indexed():
    assert Hs.yard_indexings(np.array([yard_line(0.0), yard_line(5.0)]), 270.0) == []


def dashes(y, xs):
    return project(np.column_stack([xs, np.full(len(xs), y)]))


def test_rows_sit_on_whole_yards_and_skip_clutter():
    xs = np.array([0.0, 5.0, 10.0, 15.0])
    q = Hs.pencil(np.array([yard_line(x) for x in xs]), xs)
    far_x = np.array([1, 2, 3, 4, 6, 8, 9, 11, 12, 13])          # some hidden
    near_x = np.array([1, 3, 4, 6, 7, 8, 9, 12, 14])
    rng = np.random.default_rng(1)
    clutter = rng.uniform([100, 100], [900, 500], (60, 2))       # players, numbers
    halves = dashes(30.08, np.arange(1.5, 14, 1.0))              # on a row, but half-way between yards
    pts = np.vstack([dashes(30.08, far_x), dashes(23.25, near_x), clutter, halves])
    found = Hs.rows(pts, q)
    got = sorted((sorted(r["x"].tolist()) for r in found), key=len, reverse=True)
    assert got[:2] == [far_x.tolist(), near_x.tolist()]
    for r in found[:2]:
        assert r["fill"] >= Hs.MIN_FILL


def test_dash_candidates_are_small_blobs_off_the_long_lines():
    pytest.importorskip("cv2")
    pix = np.zeros((540, 960), bool)
    pix[100:104, 200:206] = True                                 # a dash
    pix[300:303, 600:604] = True                                 # another
    pix[50:500, 400:404] = True                                  # a yard line, erased below
    pix[400:460, 700:760] = True                                 # a number: too big
    out = Hs.dash_candidates(pix, np.array([[402.0, 50.0, 402.0, 500.0]]))
    assert sorted(np.round(out).astype(int).tolist()) == [[202, 102], [602, 301]]


def test_a_row_half_hidden_by_a_logo_keeps_its_visible_run():
    """A midfield logo hides yards 1-14 of the far hash row; one dash shows past it, at -1. The row
    is the run 16-22, not a 23-yd span too empty to count."""
    xs = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    q = Hs.pencil(np.array([yard_line(x) for x in xs]), xs)
    seen = np.array([-1, 16, 17, 18, 19, 22])
    found = Hs.rows(dashes(30.08, seen), q)
    assert len(found) == 1 and found[0]["x"].tolist() == [16, 17, 18, 19, 22]
