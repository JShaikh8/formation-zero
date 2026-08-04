"""Tests for the formation taxonomy and slot assignment."""

import pytest

from gridiron.intelligence.formations import (
    ALIASES,
    ALL_SLOTS,
    FORMATIONS,
    UNOBSERVABLE_PAIRS,
    Alignment,
    assign_slots,
    canonical_id,
    classify_alignment,
    slots_to_formation,
    to_snap_frame,
)


def A(lateral, depth, position=None):
    return Alignment(lateral=lateral, depth=depth, position=position)


# ---- taxonomy integrity --------------------------------------------------------------------


def test_every_formation_has_exactly_six_skill_slots():
    """The five-man line is assumed; formations are defined by the six skill slots."""
    bad = {fid: sorted(f.slots) for fid, f in FORMATIONS.items() if len(f.slots) != 6}
    assert bad == {}


def test_no_two_formations_share_an_alignment():
    """The whole lookup depends on slot sets being unique after de-aliasing."""
    seen = {}
    for fid, f in FORMATIONS.items():
        assert f.slots not in seen, f"#{fid} duplicates #{seen.get(f.slots)}"
        seen[f.slots] = fid


def test_aliases_point_at_surviving_formations():
    for alias, target in ALIASES.items():
        assert alias not in FORMATIONS, f"#{alias} is aliased but still defined"
        assert target in FORMATIONS


def test_canonical_id_resolves_the_offset_i_duplicates():
    assert canonical_id(69) == 4      # "Offset I (Strong)" == "2 RB Offset I"
    assert canonical_id(4) == 4       # already canonical


def test_fakes_collapse_to_their_observable_parent():
    """A fake is designed to look identical pre-snap, so it cannot be a formation label."""
    assert canonical_id(54) == 53     # Fake Punt      -> Punt
    assert canonical_id(55) == 52     # Fake Field Goal -> Field Goal


def test_unobservable_pairs_are_surfaced_on_match():
    match = slots_to_formation(FORMATIONS[53].slots)
    assert match.exact
    assert match.ambiguous_with == ("Punt Formation", "Fake Punt")


def test_ordinary_formations_are_not_flagged_ambiguous():
    assert slots_to_formation(FORMATIONS[1].slots).ambiguous_with == ()


def test_all_slots_are_known_names():
    unexpected = {s for s in ALL_SLOTS if not s[0].isupper()}
    assert unexpected == set()
    assert "WRL" in ALL_SLOTS and "FBR" in ALL_SLOTS and "QBS" in ALL_SLOTS


# ---- derived attributes --------------------------------------------------------------------


def test_qb_depth_attribute():
    assert FORMATIONS[3].qb_depth == "under_center"   # I-formation
    assert FORMATIONS[11].qb_depth == "shotgun"


def test_backfield_attribute_distinguishes_i_from_offset():
    assert FORMATIONS[3].backfield == "i_form"
    assert FORMATIONS[4].backfield == "offset_i_right"
    assert FORMATIONS[68].backfield == "offset_i_left"
    assert FORMATIONS[1].backfield == "split_backs"


def test_backfield_attribute_empty_and_special_teams():
    assert FORMATIONS[46].backfield == "empty"        # 5 WR
    assert FORMATIONS[53].backfield == "special_teams"


def test_personnel_grouping_matches_nflverse_style():
    assert FORMATIONS[11].personnel == "11"   # 1 RB, 1 TE, 3 WR
    assert FORMATIONS[46].personnel == "00"   # 5 WR, empty backfield
    assert FORMATIONS[2].personnel == "12"    # 1 RB, 2 TE
    assert FORMATIONS[3].personnel == "21"    # 2 backs, 1 TE
    assert FORMATIONS[53].personnel is None   # special teams has no grouping


def test_strength_attribute():
    assert FORMATIONS[13].strength == "left"    # 3 WR all left
    assert FORMATIONS[14].strength == "right"
    assert FORMATIONS[3].strength == "right"    # lone TE right


# ---- snap frame ----------------------------------------------------------------------------


def test_to_snap_frame_attacking_right():
    """Offense attacking +x: behind the LOS is smaller x; the offense's right is larger y."""
    lateral, depth = to_snap_frame(x=48, y=30, los_x=50, ball_y=26.67, attacking="right")
    assert depth == pytest.approx(2.0)
    assert lateral == pytest.approx(3.33)


def test_to_snap_frame_attacking_left_mirrors_lateral():
    lateral, depth = to_snap_frame(x=52, y=30, los_x=50, ball_y=26.67, attacking="left")
    assert depth == pytest.approx(2.0)
    assert lateral == pytest.approx(-3.33)


def test_to_snap_frame_on_the_line_is_zero_depth():
    _lateral, depth = to_snap_frame(x=50, y=26.67, los_x=50, ball_y=26.67)
    assert depth == pytest.approx(0.0)


def test_to_snap_frame_rejects_bad_direction():
    with pytest.raises(ValueError, match="attacking"):
        to_snap_frame(50, 26, 50, 26, attacking="sideways")


# ---- slot assignment -----------------------------------------------------------------------


def test_quarterback_depth_selects_qb_or_qbs():
    assert "QB" in assign_slots([A(0, 1.5, "QB")])
    assert "QBS" in assign_slots([A(0, 6.0, "QB")])


def test_single_back_is_rb():
    slots = assign_slots([A(0, 6.0, "QB"), A(-1.0, 7.0, "RB")])
    assert "RB" in slots


def test_two_backs_at_similar_depth_are_split_backs():
    slots = assign_slots([A(0, 1.5, "QB"), A(-2.5, 6.0, "RB"), A(2.5, 6.0, "RB")])
    assert "RBL" in slots and "RBR" in slots


def test_stacked_backs_are_i_formation():
    slots = assign_slots([A(0, 1.5, "QB"), A(0.2, 4.5, "FB"), A(0.0, 7.0, "RB")])
    assert "FB" in slots and "RB" in slots


def test_displaced_fullback_is_offset():
    slots = assign_slots([A(0, 1.5, "QB"), A(2.5, 4.5, "FB"), A(0.0, 7.0, "RB")])
    assert "FBR" in slots and "RB" in slots
    slots = assign_slots([A(0, 1.5, "QB"), A(-2.5, 4.5, "FB"), A(0.0, 7.0, "RB")])
    assert "FBL" in slots


def test_three_backs_are_fullback_plus_split_pair():
    slots = assign_slots([
        A(0, 1.5, "QB"), A(0.0, 4.5, "FB"), A(-2.5, 6.5, "RB"), A(2.5, 6.5, "RB"),
    ])
    assert set(slots) >= {"FB", "RBL", "RBR"}


def test_attached_tight_end_is_inline_not_a_receiver():
    slots = assign_slots([A(0, 1.5, "QB"), A(5.5, 0.5, "TE")])
    assert "TER" in slots


def test_detached_tight_end_reads_as_a_receiver_split():
    """Flexed out past the tight end zone, a TE occupies a receiver slot — as charters call it."""
    slots = assign_slots([A(0, 1.5, "QB"), A(11.0, 1.0, "TE")])
    assert "WRR" in slots and "TER" not in slots


def test_receivers_are_numbered_outside_in():
    slots = assign_slots([
        A(0, 5.5, "QB"), A(-20.0, 0.5, "WR"), A(-11.0, 2.0, "WR"), A(-8.5, 2.0, "WR"),
    ])
    assert slots["WRL"].lateral == pytest.approx(-20.0)
    assert slots["WSL"].lateral == pytest.approx(-11.0)
    assert slots["WS2L"].lateral == pytest.approx(-8.5)


def test_tight_ends_are_numbered_inside_out():
    slots = assign_slots([A(0, 1.5, "QB"), A(4.5, 0.5, "TE"), A(6.5, 0.5, "TE")])
    assert slots["TER"].lateral == pytest.approx(4.5)
    assert slots["TE2R"].lateral == pytest.approx(6.5)


def test_linemen_are_excluded_from_slots():
    slots = assign_slots([A(0, 1.5, "QB"), A(0.0, 0.3, "C"), A(2.7, 0.3, "RT")])
    assert set(slots) == {"QB"}


def test_linemen_inferred_without_position_labels():
    """Unlabelled players on the line inside the tackle box are treated as linemen."""
    slots = assign_slots([A(0.0, 0.3), A(1.2, 0.3), A(2.7, 0.3), A(0.0, 5.5), A(-20.0, 0.5)])
    assert "QBS" in slots and "WRL" in slots
    assert not {"TER", "TEL"} & set(slots)


# ---- lookup --------------------------------------------------------------------------------


def test_slots_to_formation_exact_match():
    match = slots_to_formation(["QB", "FB", "RB", "TER", "WRL", "WRR"])
    assert match.exact
    assert match.formation.id == 3
    assert match.formation.name == "2 RB I-Formation"


def test_slots_to_formation_falls_back_to_nearest():
    """A partially recovered alignment still yields a best guess plus what disagreed."""
    match = slots_to_formation(["QB", "FB", "RB", "TER", "WRL"])   # missing one receiver
    assert not match.exact
    assert match.formation.id == 3
    assert match.missing == frozenset({"WRR"})
    assert match.extra == frozenset()


def test_slots_to_formation_reports_extra_slots():
    match = slots_to_formation(["QB", "FB", "RB", "TER", "WRL", "WRR", "TE2R"])
    assert not match.exact
    assert "TE2R" in match.extra


def test_slots_to_formation_on_empty_input():
    match = slots_to_formation([])
    assert match.formation is None and not match.exact


def test_every_formation_round_trips_through_lookup():
    for fid, f in FORMATIONS.items():
        match = slots_to_formation(f.slots)
        assert match.exact and match.formation.id == fid


# ---- end to end ----------------------------------------------------------------------------


def test_classify_alignment_recognizes_i_formation():
    players = [
        A(0.0, 0.3, "C"), A(-1.2, 0.3, "LG"), A(1.2, 0.3, "RG"),
        A(-2.7, 0.3, "LT"), A(2.7, 0.3, "RT"),
        A(0.0, 1.5, "QB"), A(0.1, 4.5, "FB"), A(0.0, 7.0, "RB"),
        A(5.0, 0.4, "TE"), A(-20.0, 0.5, "WR"), A(20.0, 0.5, "WR"),
    ]
    match = classify_alignment(players)
    assert match.exact
    assert match.formation.id == 3
    assert match.formation.personnel == "21"


def test_classify_alignment_recognizes_shotgun_trips():
    players = [
        A(0.0, 0.3, "C"), A(-1.2, 0.3, "LG"), A(1.2, 0.3, "RG"),
        A(-2.7, 0.3, "LT"), A(2.7, 0.3, "RT"),
        A(0.0, 5.5, "QB"), A(1.5, 6.5, "RB"), A(5.0, 0.4, "TE"),
        A(-20.0, 0.5, "WR"), A(-12.0, 2.0, "WR"), A(20.0, 0.5, "WR"),
    ]
    match = classify_alignment(players)
    assert match.exact
    assert match.formation.id == 11          # 3 WR (2 left), 1 RB
    assert match.formation.qb_depth == "shotgun"
