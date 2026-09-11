"""Tests for the STATS team registry (teamId is the join key, not letters)."""

import pytest

from formation_zero.data.teams import STATIC_TEAM_IDS, TeamRegistry, to_nflverse

_TEAMS_PAYLOAD = {
    "apiResults": [{"league": {"season": {"conferences": [{
        "name": "National Football Conference",
        "divisions": [{
            "name": "North",
            "teams": [
                {"teamId": 326, "location": "Chicago", "nickname": "Bears", "abbreviation": "Chi"},
                {"teamId": 335, "location": "Green Bay", "nickname": "Packers", "abbreviation": "GB"},
            ],
        }, {
            "name": "West",
            "teams": [
                {"teamId": 343, "location": "Los Angeles", "nickname": "Rams", "abbreviation": "LAR"},
            ],
        }],
    }, {
        "name": "American Football Conference",
        "divisions": [{
            "name": "West",
            "teams": [
                {"teamId": 339, "location": "Kansas City", "nickname": "Chiefs", "abbreviation": "KC"},
                {"teamId": 357, "location": "Los Angeles", "nickname": "Chargers", "abbreviation": "LAC"},
            ],
        }],
    }]}}}]
}


def test_to_nflverse_uppercases():
    assert to_nflverse("Chi") == "CHI"
    assert to_nflverse("Jax") == "JAX"


def test_to_nflverse_maps_rams_abbreviation():
    """STATS says 'LAR', nflverse says 'LA' — the one abbreviation that actually differs."""
    assert to_nflverse("LAR") == "LA"
    assert to_nflverse("LAC") == "LAC"  # Chargers are unaffected


def test_from_payload_walks_conferences_and_divisions():
    registry = TeamRegistry.from_payload(_TEAMS_PAYLOAD)
    assert {int(t["teamId"]) for t in registry.teams} == {326, 335, 343, 339, 357}


def test_from_payload_ignores_non_team_dicts():
    """Only records carrying a nickname are teams; venues and countries have ids too."""
    payload = {"teams": [{"teamId": 1, "nickname": "Bears"}, {"venueId": 9, "teamId": 2}]}
    assert len(TeamRegistry.from_payload(payload).teams) == 1


def test_abbreviation_is_nflverse_normalized_by_default():
    registry = TeamRegistry.from_payload(_TEAMS_PAYLOAD)
    assert registry.abbreviation(343) == "LA"
    assert registry.abbreviation(343, nflverse=False) == "LAR"


def test_abbreviation_of_unknown_id_is_none():
    assert TeamRegistry.from_payload(_TEAMS_PAYLOAD).abbreviation(99999) is None


def test_resolve_by_id_abbrev_nickname_and_location():
    registry = TeamRegistry.from_payload(_TEAMS_PAYLOAD)
    for needle in (326, "326", "CHI", "Chi", "Bears", "Chicago", "Chicago Bears"):
        assert registry.resolve_id(needle) == 326


def test_resolve_accepts_nflverse_spelling_of_the_rams():
    assert TeamRegistry.from_payload(_TEAMS_PAYLOAD).resolve_id("LA") == 343


def test_resolve_does_not_substring_match():
    """'CHI' must not also match the Chiefs — that silently returns the wrong game."""
    registry = TeamRegistry.from_payload(_TEAMS_PAYLOAD)
    assert registry.resolve_id("CHI") == 326
    assert registry.resolve_id("KC") == 339


def test_resolve_id_raises_on_no_match():
    with pytest.raises(SystemExit, match="No team matches"):
        TeamRegistry.from_payload(_TEAMS_PAYLOAD).resolve_id("Sharks")


def test_resolve_id_raises_on_ambiguity():
    """Two Los Angeles teams — 'Los Angeles' alone cannot identify one."""
    with pytest.raises(SystemExit, match="ambiguous"):
        TeamRegistry.from_payload(_TEAMS_PAYLOAD).resolve_id("Los Angeles")


def test_fallback_registry_covers_all_32_franchises():
    registry = TeamRegistry.fallback()
    assert len(STATIC_TEAM_IDS) == 32
    assert registry.abbreviation(326) == "CHI"
    assert registry.abbreviation(343) == "LA"


def test_fallback_registry_resolves_without_network():
    assert TeamRegistry.fallback().resolve_id("KC") == 339
