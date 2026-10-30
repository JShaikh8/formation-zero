"""Frame sampling for the teacher: strata, spread, and staying inside the takes."""

from collections import Counter

import pandas as pd

from formation_zero.labelling.sample import PHASES, sample_frames


def index_and_pbp(n_plays=40):
    rows, pbp = [], []
    for i in range(n_plays):
        t = 100.0 + 30.0 * i
        rows.append({"play_uid": f"g_p{i:03d}", "film_play_index": i, "sideline_start_s": t, "sideline_end_s": t + 14,
                     "endzone_start_s": t + 14, "endzone_end_s": t + 28})
        pbp.append({"play_uid": f"g_p{i:03d}", "qtr": 1 + 4 * i // n_plays})
    return pd.DataFrame(rows), pd.DataFrame(pbp)


def test_every_stratum_gets_its_share():
    index, pbp = index_and_pbp()
    s = sample_frames(index, pbp, n=120)
    counts = Counter((x["angle"], x["phase_guess"], x["half"]) for x in s)
    assert len(counts) == 12 and set(counts.values()) == {10}


def test_frames_sit_inside_their_phase_window_of_their_take():
    index, pbp = index_and_pbp()
    by_uid = index.set_index("play_uid")
    for x in sample_frames(index, pbp, n=240):
        start, end = by_uid.loc[x["play_uid"], [f"{x['angle']}_start_s", f"{x['angle']}_end_s"]]
        lo, hi = PHASES[x["phase_guess"]]
        assert start + lo * (end - start) - 0.04 <= x["time_s"] <= start + hi * (end - start) + 0.04


def test_more_frames_than_plays_revisits_plays_at_new_offsets():
    index, pbp = index_and_pbp(n_plays=8)
    s = sample_frames(index, pbp, n=12 * 10)
    assert len(s) == 120 and len({x["frame"] for x in s}) == 120


def test_it_is_deterministic():
    index, pbp = index_and_pbp()
    assert sample_frames(index, pbp, n=100) == sample_frames(index, pbp, n=100)
