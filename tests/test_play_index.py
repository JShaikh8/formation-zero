"""Tests for building the play index from shots.

This is the join everything downstream depends on, and its dangerous failure is silent: an extra
or missing clip in the middle keeps the totals plausible while misattaching every later play. So
the tests pin the two things that prevent that — pairing starts at the first shot rather than
assuming which angle leads, and the join refuses to produce a table at all when the film and the
play list disagree on how many plays happened.
"""

import numpy as np
import pandas as pd
import pytest

from gridiron.perception.play_index import (
    FILMABLE_PLAY_TYPES, Play, emissions_from, filmable, join, kde_logpdf, label_views,
    merge_same_view, pair_plays,
)
from gridiron.perception.shots import Shot


def shots(n, duration=14.0):
    return [Shot(start=i * duration, duration=duration, tilt=None, samples=35) for i in range(n)]


def alternating(n, first="ENDZONE"):
    other = "SIDELINE" if first == "ENDZONE" else "ENDZONE"
    return [first if i % 2 == 0 else other for i in range(n)]


# ---- emissions -----------------------------------------------------------------------------


def test_kde_is_denser_near_its_training_points():
    points = [80.0, 84.0, 88.0]
    assert kde_logpdf(84.0, points) > kde_logpdf(40.0, points)


def test_kde_represents_a_bimodal_class():
    """Sideline reads near 0 or near 85; both modes must score above the gap between them."""
    points = [1.0, 3.0, 5.0, 82.0, 85.0, 88.0]
    assert kde_logpdf(3.0, points) > kde_logpdf(45.0, points)
    assert kde_logpdf(85.0, points) > kde_logpdf(45.0, points)


def test_kde_on_no_points_is_negative_infinity():
    assert kde_logpdf(50.0, []) == float("-inf")


def test_emissions_drop_unmeasurable_tilts():
    labels = [{"view": "SIDELINE", "tilt": 85.0}, {"view": "SIDELINE", "tilt": None},
              {"view": "ENDZONE", "tilt": float("nan")}, {"view": "ENDZONE", "tilt": 30.0}]
    assert emissions_from(labels) == {"SIDELINE": [85.0], "ENDZONE": [30.0]}


# ---- view labelling ------------------------------------------------------------------------


def test_viterbi_recovers_alternation_from_clean_signal():
    em = {"SIDELINE": [85.0], "ENDZONE": [30.0]}
    tilts = [85.0, 30.0, 85.0, 30.0, 85.0, 30.0]
    assert label_views(tilts, em) == ["SIDELINE", "ENDZONE"] * 3


# Emission sets shaped like the real ones: sideline is bimodal near 0 and near 85, endzone
# occupies the middle band. Single-point emissions would make the KDE absurdly confident and
# no realistic prior could move it, which is not the regime this model runs in.
REAL_EMISSIONS = {"SIDELINE": [1.0, 3.0, 5.0, 80.0, 84.0, 88.0],
                  "ENDZONE": [15.0, 25.0, 30.0, 35.0, 45.0, 50.0]}


def test_viterbi_overrides_one_ambiguous_shot_using_its_neighbours():
    """The whole reason for a sequence model: structure beats a single bad observation.

    Shot 3 reads 70, which on its own looks sideline. It sits between two confident sidelines,
    and alternation says it cannot be one.
    """
    tilts = [85.0, 30.0, 85.0, 70.0, 85.0, 30.0, 85.0]
    assert label_views(tilts, REAL_EMISSIONS, p_same=0.02)[3] == "ENDZONE"


def test_a_weak_alternation_prior_lets_the_observation_win():
    """The override is not unconditional — it is the prior's strength doing the work."""
    tilts = [85.0, 30.0, 85.0, 70.0, 85.0, 30.0, 85.0]
    assert label_views(tilts, REAL_EMISSIONS, p_same=0.45)[3] == "SIDELINE"


def test_viterbi_carries_a_shot_with_no_measurable_tilt():
    em = {"SIDELINE": [85.0], "ENDZONE": [30.0]}
    assert label_views([85.0, 30.0, 85.0, None, 85.0], em)[3] == "ENDZONE"


def test_viterbi_handles_nan_the_same_as_none():
    em = {"SIDELINE": [85.0], "ENDZONE": [30.0]}
    assert label_views([85.0, 30.0, 85.0, float("nan"), 85.0], em)[3] == "ENDZONE"


def test_viterbi_on_empty_input():
    assert label_views([], {"SIDELINE": [85.0], "ENDZONE": [30.0]}) == []


# ---- merging -------------------------------------------------------------------------------


def test_merge_collapses_a_same_view_run():
    """A spurious cut splits one take into two pieces of the same view."""
    s = shots(4)
    merged, views = merge_same_view(s, ["ENDZONE", "SIDELINE", "SIDELINE", "ENDZONE"])
    assert views == ["ENDZONE", "SIDELINE", "ENDZONE"]
    assert len(merged) == 3


def test_merged_shot_spans_the_whole_run():
    s = shots(3)
    merged, _ = merge_same_view(s, ["SIDELINE", "SIDELINE", "SIDELINE"])
    assert len(merged) == 1
    assert merged[0].start == s[0].start
    assert merged[0].end == pytest.approx(s[-1].end)


def test_merge_leaves_a_strictly_alternating_sequence_untouched():
    s = shots(6)
    merged, views = merge_same_view(s, alternating(6))
    assert len(merged) == 6 and views == alternating(6)


# ---- pairing -------------------------------------------------------------------------------


def test_pairing_starts_at_the_first_shot_when_endzone_leads():
    """The demo film leads with endzone; assuming sideline-first would drop the opening play."""
    plays = pair_plays(shots(6), alternating(6, first="ENDZONE"))
    assert len(plays) == 3
    assert plays[0].endzone.start == 0.0
    assert plays[0].sideline.start == 14.0


def test_pairing_handles_sideline_leading_film():
    plays = pair_plays(shots(6), alternating(6, first="SIDELINE"))
    assert len(plays) == 3
    assert plays[0].sideline.start == 0.0


def test_pairing_rejects_an_unmerged_same_view_pair():
    with pytest.raises(ValueError, match="merge_same_view"):
        pair_plays(shots(4), ["SIDELINE", "SIDELINE", "ENDZONE", "SIDELINE"])


def test_play_span_covers_both_angles():
    plays = pair_plays(shots(2), ["ENDZONE", "SIDELINE"])
    assert plays[0].start == 0.0
    assert plays[0].end == pytest.approx(28.0)


# ---- the join ------------------------------------------------------------------------------


def pbp_frame(types):
    return pd.DataFrame({
        "game_key": ["gk"] * len(types),
        "play_uid": [f"gk-{i}" for i in range(len(types))],
        "play_index": range(len(types)),
        "play_type": types,
        "desc": [f"play {i}" for i in range(len(types))],
    })


def test_filmable_drops_rows_that_put_nothing_on_film():
    frame = pbp_frame(["pass", "timeout", "run", "two_minute_warning", "penalty", "punt"])
    assert list(filmable(frame)["play_type"]) == ["pass", "run", "punt"]


def test_filmable_keeps_every_kicking_play():
    assert set(FILMABLE_PLAY_TYPES) >= {"punt", "field_goal", "extra_point", "kickoff"}


def test_join_matches_plays_to_rows_in_order():
    plays = pair_plays(shots(6), alternating(6))
    table = join(plays, pbp_frame(["pass", "run", "punt"]), fps=25.0)
    assert list(table["play_type"]) == ["pass", "run", "punt"]
    assert list(table["film_play_index"]) == [0, 1, 2]


def test_join_refuses_when_counts_disagree():
    """A length mismatch means an extra or missing clip, which shifts every later play."""
    plays = pair_plays(shots(6), alternating(6))
    with pytest.raises(ValueError, match="plays on film"):
        join(plays, pbp_frame(["pass", "run"]), fps=25.0)


def test_join_carries_frame_numbers_for_both_angles():
    plays = pair_plays(shots(2), ["ENDZONE", "SIDELINE"])
    row = join(plays, pbp_frame(["pass"]), fps=25.0).iloc[0]
    assert row["endzone_start_frame"] == 0
    assert row["sideline_start_frame"] == 350
