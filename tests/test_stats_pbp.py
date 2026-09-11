"""Tests for normalizing STATS play-by-play into the tidy schema.

The play classification cases here are the ones that actually broke against real data: they
decide `is_scrimmage`, which sets `play_index` — the key that aligns film clips to plays. An
off-by-one here misaligns every clip after it.
"""

from formation_zero.data import stats_pbp
from formation_zero.data.stats_pbp import classify_play, infer_play_type, parse_yardline


def _play(play_type_id, text, **extra):
    return {"playType": {"playTypeId": play_type_id}, "playText": text, **extra}


# ---- yard lines ----------------------------------------------------------------------------


def test_parse_yardline_opponent_half():
    """Rams on the Bears' 37 are 37 yards from the goal they're attacking."""
    assert parse_yardline("Chi37", "LA") == 37


def test_parse_yardline_own_half():
    """Rams on their own 35 are 65 yards out."""
    assert parse_yardline("LAR35", "LA") == 65


def test_parse_yardline_handles_stats_to_nflverse_abbrev():
    """STATS writes the Rams 'LAR'; nflverse writes 'LA'. Own-half detection must survive that."""
    assert parse_yardline("LAR20", "LA") == 80


def test_parse_yardline_midfield():
    assert parse_yardline("Chi50", "LA") == 50
    assert parse_yardline("LAR50", "LA") == 50


def test_parse_yardline_returns_none_when_possession_unknown():
    assert parse_yardline("Chi37", None) is None


def test_parse_yardline_rejects_malformed_and_out_of_range():
    assert parse_yardline("", "LA") is None
    assert parse_yardline("midfield", "LA") is None
    assert parse_yardline("Chi73", "LA") is None


# ---- play classification -------------------------------------------------------------------


def test_pass_and_rush_are_scrimmage():
    assert classify_play(_play(1, "Williams pass to Moore for 34 yards.")) == ("pass", True)
    assert classify_play(_play(4, "Swift rush for 3 yards.")) == ("run", True)


def test_sack_and_interception_normalize_to_pass():
    assert classify_play(_play(3, "Stafford sacked for -7."))[0] == "pass"
    assert classify_play(_play(9, "Williams pass intercepted by Fuller."))[0] == "pass"


def test_dead_ball_penalty_is_not_a_snap():
    """'Penalty: False Start ...' — nothing was snapped, so there is no film clip."""
    play_type, is_scrimmage = classify_play(_play(10, "Penalty: False Start on Bears (Odunze) -5 yards."))
    assert play_type == "penalty"
    assert is_scrimmage is False


def test_penalty_after_a_completed_snap_is_a_snap():
    """The play happened and is on film, even though the penalty negated it."""
    text = ("Matthew Stafford pass to the right to Kyren Williams for 14 yards to the Chi28. "
            "Penalty: Unnecessary Roughness on Bears.")
    assert classify_play(_play(11, text)) == ("pass", True)


def test_penalty_rows_are_not_classified_by_type_id_alone():
    """Both are playTypeId 11 — only the text distinguishes them."""
    dead = classify_play(_play(11, "Penalty: Illegal Use of Hands on Bears 5 yards."))
    live = classify_play(_play(11, "Caleb Williams pass to Odunze for 17 yards. Penalty: Roughing Passer."))
    assert dead[1] is False and live[1] is True


def test_fumble_on_a_sack_is_a_scrimmage_play():
    text = "Stafford sacked at LAR23 for a loss of 1 yard. FUMBLE, recovered by LAR."
    assert classify_play(_play(14, text)) == ("pass", True)


def test_fumble_on_a_punt_return_is_not_a_scrimmage_play():
    """The id says 'Fumble', but the snap was a punt — counting it shifts every later clip."""
    text = "Ethan Evans punts for 61 yards to Chi8. Duvernay return for 15 yards. FUMBLE, recovered by Chi."
    play_type, is_scrimmage = classify_play(_play(14, text))
    assert play_type == "punt"
    assert is_scrimmage is False


def test_kicking_and_administrative_plays_are_not_scrimmage():
    for type_id, text in [(5, "Evans kicks off to the Chi10."), (7, "Evans punts for 45 yards."),
                          (42, "Santos 38 yard field goal is good."), (13, "Two Minute Warning"),
                          (57, "Timeout by Bears")]:
        assert classify_play(_play(type_id, text))[1] is False


def test_infer_play_type_prefers_kicking_over_pass_wording():
    """A punt narration mentioning a return must not be read as a scrimmage play."""
    assert infer_play_type("Evans punts for 61 yards. Duvernay return for 15 yards.") == "punt"


def test_infer_play_type_returns_none_for_unrecognized_text():
    assert infer_play_type("") is None
    assert infer_play_type("Officials review the previous ruling.") is None


# ---- normalization -------------------------------------------------------------------------


_EVENT = {
    "apiResults": [{"league": {"season": {"season": 2025, "eventType": [{"events": [{
        "eventId": 2879591,
        "week": 20,
        "startDate": [{"full": "2026-01-18T23:30:00", "dateType": "UTC"}],
        "teams": [
            {"teamId": 326, "teamLocationType": {"name": "home"}},
            {"teamId": 343, "teamLocationType": {"name": "away"}},
        ],
        "pbp": [
            {"playId": 1.0, "period": 1, "down": None, "distance": None, "yardLine": "LAR35",
             "startPossession": {"teamId": 343}, "playType": {"playTypeId": 5},
             "playText": "Evans kicks off to the Chi10.", "yards": 27},
            {"playId": 2.0, "period": 1, "down": "1", "distance": "10", "yardLine": "Chi37",
             "startPossession": {"teamId": 326}, "playType": {"playTypeId": 1},
             "playText": "Williams pass to Moore for 34 yards.", "yards": 34,
             "playersInvolved": [
                 {"playerInvolvedTypeId": 1, "player": {"playerId": 1, "firstName": "Caleb",
                                                        "lastName": "Williams", "uniform": "18"}},
                 {"playerInvolvedTypeId": 5, "player": {"playerId": 2, "firstName": "DJ",
                                                        "lastName": "Moore", "uniform": "2"}},
                 {"playerInvolvedTypeId": 2, "player": {"playerId": 3, "firstName": "Roger",
                                                        "lastName": "McCreary", "uniform": "25"}},
             ]},
            {"playId": 3.0, "period": 1, "down": "2", "distance": "10", "yardLine": "Chi35",
             "startPossession": {"teamId": 326}, "playType": {"playTypeId": 10},
             "playText": "Penalty: False Start on Bears -5 yards.", "yards": -5},
            {"playId": 4.0, "period": 1, "down": "2", "distance": "15", "yardLine": "Chi32",
             "startPossession": {"teamId": 326}, "playType": {"playTypeId": 4},
             "playText": "Swift rush for 5 yards.", "yards": 5},
        ],
    }]}]}}}]
}


def _normalized():
    from formation_zero.data.teams import TeamRegistry

    return stats_pbp.normalize_event(_EVENT, TeamRegistry.fallback())


def test_normalize_builds_expected_game_key():
    assert _normalized()["game_key"].iloc[0] == "2025_wk20_LA-CHI"


def test_normalize_marks_only_snaps_as_scrimmage():
    tidy = _normalized()
    assert list(tidy["is_scrimmage"]) == [False, True, False, True]


def test_play_index_counts_only_scrimmage_plays():
    """The kickoff and the dead-ball penalty must not consume an index."""
    tidy = _normalized()
    # pandas stores the gaps as NaN in a numeric column, as formation_zero.data.pbp does.
    assert tidy["play_index"].isna().tolist() == [True, False, True, False]
    assert tidy["play_index"].dropna().tolist() == [1, 2]
    assert list(tidy["play_uid"]) == [None, "2025_wk20_LA-CHI_p001", None, "2025_wk20_LA-CHI_p002"]


def test_normalize_converts_yardlines_and_possession():
    tidy = _normalized()
    assert tidy.loc[1, "posteam"] == "CHI"
    assert tidy.loc[1, "defteam"] == "LA"
    assert tidy.loc[1, "yardline_100"] == 63  # Bears' own 37


def test_normalize_extracts_uniforms_with_roles():
    players = _normalized().loc[1, "players"]
    assert {"role": "player", "player_id": 1, "name": "Caleb Williams", "uniform": "18"} in players
    assert {p["role"] for p in players} == {"player", "target", "tackler"}


def test_jersey_index_splits_by_side_of_ball():
    """Offensive roles belong to the possessing team; tacklers to the defense."""
    index = stats_pbp.jersey_index(_normalized())
    assert index["CHI"] == {"18", "2"}
    assert index["LA"] == {"25"}
