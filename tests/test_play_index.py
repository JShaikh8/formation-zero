"""Tests for building the play index from shots.

This is the join everything downstream depends on, and its dangerous failure is silent: an extra
or missing clip in the middle keeps the totals plausible while misattaching every later play. So
the tests pin the two things that prevent that — pairing starts at the first shot rather than
assuming which angle leads, and the join refuses to produce a table at all when the film and the
play list disagree on how many plays happened.
"""

import pandas as pd
import pytest

from formation_zero.data.playtypes import maybe_on_film, on_film, snapped
from formation_zero.perception.play_index import (
    FILMABLE_PLAY_TYPES,
    emissions_from,
    film_rows,
    filmable,
    join,
    kde_logpdf,
    label_views,
    merge_same_view,
    on_film_rows,
    pair_plays,
)
from formation_zero.perception.shots import Shot


def shots(n, duration=14.0):
    return [Shot(start=i * duration, duration=duration, tilt=None, samples=35) for i in range(n)]


def alternating(n, first="SIDELINE"):
    other = "ENDZONE" if first == "SIDELINE" else "SIDELINE"
    return [first if i % 2 == 0 else other for i in range(n)]


# ---- emissions -----------------------------------------------------------------------------


def test_kde_is_denser_near_its_training_points():
    points = [80.0, 84.0, 88.0]
    assert kde_logpdf(84.0, points) > kde_logpdf(40.0, points)


def test_kde_represents_a_bimodal_class():
    """The endzone camera reads near 0 or near 85; both modes must score above the gap between them."""
    points = [1.0, 3.0, 5.0, 82.0, 85.0, 88.0]
    assert kde_logpdf(3.0, points) > kde_logpdf(45.0, points)
    assert kde_logpdf(85.0, points) > kde_logpdf(45.0, points)


def test_kde_on_no_points_is_negative_infinity():
    assert kde_logpdf(50.0, []) == float("-inf")


def test_emissions_drop_unmeasurable_tilts():
    labels = [{"view": "ENDZONE", "tilt": 85.0}, {"view": "ENDZONE", "tilt": None},
              {"view": "SIDELINE", "tilt": float("nan")}, {"view": "SIDELINE", "tilt": 30.0}]
    assert emissions_from(labels) == {"ENDZONE": [85.0], "SIDELINE": [30.0]}


# ---- view labelling ------------------------------------------------------------------------


def test_viterbi_recovers_alternation_from_clean_signal():
    em = {"ENDZONE": [85.0], "SIDELINE": [30.0]}
    tilts = [85.0, 30.0, 85.0, 30.0, 85.0, 30.0]
    assert label_views(tilts, em) == ["ENDZONE", "SIDELINE"] * 3


# Emission sets shaped like the real ones: endzone is bimodal near 0 and near 85, sideline
# occupies the middle band. Single-point emissions would make the KDE absurdly confident and
# no realistic prior could move it, which is not the regime this model runs in.
REAL_EMISSIONS = {"ENDZONE": [1.0, 3.0, 5.0, 80.0, 84.0, 88.0],
                  "SIDELINE": [15.0, 25.0, 30.0, 35.0, 45.0, 50.0]}


def test_viterbi_overrides_one_ambiguous_shot_using_its_neighbours():
    """The whole reason for a sequence model: structure beats a single bad observation.

    Shot 3 reads 70, which on its own looks endzone. It sits between two confident endzones,
    and alternation says it cannot be one.
    """
    tilts = [85.0, 30.0, 85.0, 70.0, 85.0, 30.0, 85.0]
    assert label_views(tilts, REAL_EMISSIONS, p_same=0.02)[3] == "SIDELINE"


def test_a_weak_alternation_prior_lets_the_observation_win():
    """The override is not unconditional — it is the prior's strength doing the work."""
    tilts = [85.0, 30.0, 85.0, 70.0, 85.0, 30.0, 85.0]
    assert label_views(tilts, REAL_EMISSIONS, p_same=0.45)[3] == "ENDZONE"


def test_viterbi_carries_a_shot_with_no_measurable_tilt():
    em = {"ENDZONE": [85.0], "SIDELINE": [30.0]}
    assert label_views([85.0, 30.0, 85.0, None, 85.0], em)[3] == "SIDELINE"


def test_viterbi_handles_nan_the_same_as_none():
    em = {"ENDZONE": [85.0], "SIDELINE": [30.0]}
    assert label_views([85.0, 30.0, 85.0, float("nan"), 85.0], em)[3] == "SIDELINE"


def test_viterbi_on_empty_input():
    assert label_views([], {"ENDZONE": [85.0], "SIDELINE": [30.0]}) == []


# ---- merging -------------------------------------------------------------------------------


def test_merge_collapses_a_same_view_run():
    """A spurious cut splits one take into two pieces of the same view."""
    s = shots(4)
    merged, views = merge_same_view(s, ["SIDELINE", "ENDZONE", "ENDZONE", "SIDELINE"])
    assert views == ["SIDELINE", "ENDZONE", "SIDELINE"]
    assert len(merged) == 3


def test_merged_shot_spans_the_whole_run():
    s = shots(3)
    merged, _ = merge_same_view(s, ["ENDZONE", "ENDZONE", "ENDZONE"])
    assert len(merged) == 1
    assert merged[0].start == s[0].start
    assert merged[0].end == pytest.approx(s[-1].end)


def test_merge_leaves_a_strictly_alternating_sequence_untouched():
    s = shots(6)
    merged, views = merge_same_view(s, alternating(6))
    assert len(merged) == 6 and views == alternating(6)


# ---- pairing -------------------------------------------------------------------------------


def test_pairing_starts_at_the_first_shot_when_sideline_leads():
    """The demo film leads with sideline; assuming endzone-first would drop the opening play."""
    plays = pair_plays(shots(6), alternating(6, first="SIDELINE"))
    assert len(plays) == 3
    assert plays[0].sideline.start == 0.0
    assert plays[0].endzone.start == 14.0


def test_pairing_handles_endzone_leading_film():
    plays = pair_plays(shots(6), alternating(6, first="ENDZONE"))
    assert len(plays) == 3
    assert plays[0].endzone.start == 0.0


def test_pairing_rejects_an_unmerged_same_view_pair():
    with pytest.raises(ValueError, match="merge_same_view"):
        pair_plays(shots(4), ["ENDZONE", "ENDZONE", "SIDELINE", "ENDZONE"])


def test_play_span_covers_both_angles():
    plays = pair_plays(shots(2), ["SIDELINE", "ENDZONE"])
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
    plays = pair_plays(shots(2), ["SIDELINE", "ENDZONE"])
    row = join(plays, pbp_frame(["pass"]), fps=25.0).iloc[0]
    assert row["sideline_start_frame"] == 0
    assert row["endzone_start_frame"] == 350


# ---- what is on film ------------------------------------------------------------------------

NULLIFIED_PASS = ("(Shotgun) M.Stafford pass incomplete short middle to P.Nacua.\n"
                  "PENALTY on LAR-W.McClendon, Offensive Holding, 10 yards, enforced at LAR 30 - No Play.")
FALSE_START = "PENALTY on LAR-K.Dotson, False Start, 5 yards, enforced at SF 19 - No Play."


@pytest.mark.parametrize("desc,expected", [
    (NULLIFIED_PASS, True),
    (FALSE_START, False),
    ("(4:10) (Shotgun) PENALTY on CHI-15-R.Odunze, False Start, 5 yards, enforced at CHI 35 - No Play.", False),
    ("(4:22) 9-M.Stafford pass incomplete short right to 84-C.Parkinson. PENALTY on CHI-26-N.Wright, "
     "Illegal Use of Hands, 5 yards, enforced at CHI 41 - No Play.", True),
    ("", False),
])
def test_a_nullified_play_is_on_film_only_if_it_was_snapped(desc, expected):
    assert snapped(desc) is expected
    assert on_film("no_play", desc) is expected


def test_on_film_rows_keep_nullified_snaps_but_not_presnap_fouls():
    frame = pd.DataFrame({"play_type": ["pass", "no_play", "timeout", "no_play", "punt"],
                          "desc": ["a pass", NULLIFIED_PASS, "Timeout", FALSE_START, "a punt"]})
    assert list(on_film_rows(frame)["desc"]) == ["a pass", NULLIFIED_PASS, "a punt"]
    assert list(filmable(frame)["desc"]) == ["a pass", "a punt"]      # the counted plays are unchanged


@pytest.mark.parametrize("ptype,desc,expected", [
    ("no_play", FALSE_START, True),          # pre-snap foul: on film in some games
    ("qb_kneel", "M.Stafford kneels to LA 26 for -1 yards.", True),
    ("no_play", NULLIFIED_PASS, False),      # snapped: always on film
    ("pass", "a pass", False),
])
def test_presnap_fouls_and_kneels_may_or_may_not_be_on_film(ptype, desc, expected):
    assert maybe_on_film(ptype, desc) is expected


def test_film_rows_flag_the_optional_ones():
    frame = pd.DataFrame({"play_type": ["pass", "no_play", "timeout", "qb_kneel"],
                          "desc": ["a pass", FALSE_START, "Timeout", "kneels"]})
    rows = film_rows(frame)
    assert list(rows["desc"]) == ["a pass", FALSE_START, "kneels"]
    assert list(rows["film_optional"]) == [False, True, True]
