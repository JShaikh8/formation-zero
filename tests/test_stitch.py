"""Rejoining broken tracks and filling short gaps, on made-up boxes."""

import numpy as np
import pandas as pd

from formation_zero.perception.track.stitch import fill, links, stitch

FPS = 25.0


def run(track, frames, x0=100.0, vx=2.0, label="player", y=500.0):
    """A player walking right at `vx` px/frame, boxes 80 px tall."""
    f = np.asarray(frames)
    x = x0 + vx * f
    return pd.DataFrame({"frame": f, "track_id": track, "label": label, "conf": 0.8, "x1": x - 15, "y1": y - 80.0,
                         "x2": x + 15, "y2": y, "off_field": False})


def test_a_player_lost_for_a_moment_is_one_track_again():
    boxes = pd.concat([run(1, range(30)), run(2, range(40, 70))])
    out, teams, joined = stitch(boxes, {1: 0, 2: 0}, FPS)
    assert joined == 1
    assert set(out["track_id"]) == {1}
    assert teams == {1: 0}


def test_opposing_players_are_never_joined_however_well_they_line_up():
    boxes = pd.concat([run(1, range(30)), run(2, range(35, 70))])
    assert links(boxes, {1: 0, 2: 1}, FPS) == {}


def test_a_track_too_short_to_have_a_team_takes_the_one_it_joins():
    boxes = pd.concat([run(1, range(30)), run(2, range(35, 40))])
    _, teams, joined = stitch(boxes, {1: 1, 2: -1}, FPS)
    assert joined == 1
    assert teams == {1: 1}


def test_no_join_across_a_shot_change_or_past_the_longest_gap():
    boxes = pd.concat([run(1, range(30)), run(2, range(40, 70))])
    assert links(boxes, {1: 0, 2: 0}, FPS, cuts=[35]) == {}
    far = pd.concat([run(1, range(30)), run(2, range(60, 90))])
    assert links(far, {1: 0, 2: 0}, FPS) == {}


def test_each_end_joins_the_track_its_motion_points_at():
    """Two teammates run side by side and both are lost: each rejoins his own, not the nearer start."""
    boxes = pd.concat([run(1, range(30), y=500.0), run(2, range(30), y=550.0),
                       run(3, range(36, 60), y=500.0), run(4, range(36, 60), y=550.0)])
    assert links(boxes, {1: 0, 2: 0, 3: 0, 4: 0}, FPS) == {3: 1, 4: 2}


def test_a_chain_of_breaks_ends_up_under_its_first_id():
    boxes = pd.concat([run(1, range(20)), run(2, range(25, 45)), run(3, range(50, 70))])
    out, _, joined = stitch(boxes, {1: 0, 2: 0, 3: 0}, FPS)
    assert joined == 2
    assert set(out["track_id"]) == {1}


def test_bench_tracks_are_left_alone():
    boxes = pd.concat([run(1, range(30)), run(2, range(40, 70)).assign(off_field=True)])
    _, _, joined = stitch(boxes, {1: 0, 2: 0}, FPS)
    assert joined == 0


def test_a_track_on_the_painted_shield_joins_nobody():
    """Boxes far wider than tall: the NFL shield called a player. A real player must not inherit it."""
    shield = run(1, range(30), vx=0.0)
    shield = shield.assign(x1=shield["x1"] - 80, x2=shield["x2"] + 80)
    boxes = pd.concat([shield, run(2, range(35, 70), vx=0.0)])
    assert links(boxes, {1: 0, 2: 0}, FPS) == {}


def test_a_gap_is_filled_on_the_straight_line_and_marked():
    boxes = run(1, [*range(10), *range(15, 25)])
    out = fill(boxes, FPS)
    assert out["frame"].tolist() == list(range(25))
    made = out[out["filled"]]
    assert made["frame"].tolist() == [10, 11, 12, 13, 14]
    assert made["conf"].isna().all()
    assert np.allclose(made["x1"], 100 + 2.0 * made["frame"] - 15)


def test_long_gaps_and_gaps_across_a_shot_change_are_not_filled():
    long_gap = run(1, [*range(10), *range(60, 70)])
    assert not fill(long_gap, FPS)["filled"].any()
    assert not fill(run(1, [*range(10), *range(15, 25)]), FPS, cuts=[12])["filled"].any()
