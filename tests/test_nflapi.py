"""NFL API port: token discipline, adapters, play classification. No network."""

from __future__ import annotations

import json

import pytest

from formation_zero.data.nflapi import stat_types as st
from formation_zero.data.nflapi.client import FRESH_TOKEN_S, NflApi, NotConfigured
from formation_zero.data.nflapi.config import NflApiConfig
from formation_zero.data.nflapi.game_detail import adapt_game_detail, canonical_team, classify, normalize_jersey, yardline_100
from formation_zero.data.nflapi.merge import merge_labels
from formation_zero.data.nflapi.rosters import adapt_roster, jersey_map, name_forms
from formation_zero.data.nflapi.teams import abbreviation_from_logo, adapt_teams

CFG = NflApiConfig(client_id="id", client_key="key", client_secret="secret", device_id="dev")


class FakeHttp:
    """Scripted HTTP: token mints always succeed; data GETs answer from a queue of statuses."""

    def __init__(self, data_statuses=(200,), ttl=900):
        self.calls = []
        self.data_statuses = list(data_statuses)
        self.mints = 0
        self.ttl = ttl

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        if url.endswith("/identity/v3/token"):
            self.mints += 1
            return 200, json.dumps({"access_token": f"tok{self.mints}", "expires_in": self.ttl})
        status = self.data_statuses.pop(0) if self.data_statuses else 200
        return status, json.dumps({"ok": True, "auth": headers.get("Authorization")})


def test_not_configured_says_so():
    api = NflApi(NflApiConfig(), http=FakeHttp())
    assert not api.is_configured()
    with pytest.raises(NotConfigured):
        api.get("/x")


def test_token_is_cached_until_near_expiry():
    now = [1000.0]
    http = FakeHttp(ttl=900)
    api = NflApi(CFG, http=http, clock=lambda: now[0])
    api.get("/a"); api.get("/b")
    assert http.mints == 1
    now[0] += 900 - 30            # inside the 60 s skew: refresh
    api.get("/c")
    assert http.mints == 2


def test_401_is_retried_once_with_a_fresh_token_unless_the_token_is_fresh():
    now = [0.0]
    http = FakeHttp(data_statuses=[401, 200])
    api = NflApi(CFG, http=http, clock=lambda: now[0])
    with pytest.raises(RuntimeError):
        api.get("/scope")          # minted seconds ago: a 401 is not staleness, no re-mint
    assert http.mints == 1
    now[0] += FRESH_TOKEN_S + 1
    http.data_statuses = [401, 200]
    payload = api.get("/data")
    assert http.mints == 2 and payload["auth"] == "Bearer tok2"


def test_401_that_survives_a_fresh_token_fails_loudly():
    now = [100.0]
    http = FakeHttp(data_statuses=[200, 401, 401])
    api = NflApi(CFG, http=http, clock=lambda: now[0])
    api.get("/warm"); now[0] += FRESH_TOKEN_S + 1
    with pytest.raises(RuntimeError):
        api.get("/dead")
    assert http.mints == 2


def test_raw_payload_is_cached_to_disk(tmp_path):
    api = NflApi(CFG, http=FakeHttp(), cache_dir=tmp_path)
    api.get("/x", cache_name="g/detail.json")
    assert json.loads((tmp_path / "g" / "detail.json").read_text())["ok"] is True


def test_yardline_100_is_relative_to_the_defending_goal():
    assert yardline_100("CHI 37", "CHI") == 63
    assert yardline_100("LA 29", "CHI") == 29
    assert yardline_100("50", "CHI") == 50
    assert yardline_100(None, "CHI") is None


@pytest.mark.parametrize("ptype,desc,expected", [
    ("PASS", "18-C.Williams pass short right to 2-D.Moore to LA 29 for 34 yards.", "pass"),
    ("RUSH", "4-D.Swift right guard to LA 23 for 6 yards.", "run"),
    ("SACK", "9-M.Stafford sacked at LA 20 for -7 yards.", "pass"),
    ("PASS", "(Shotgun) 18-C.Williams scrambles left tackle ran ob at LA 37 for 26 yards.", "run"),
    ("KICK_OFF", "3-C.Santos kicks 65 yards from CHI 35 to end zone, Touchback.", "kickoff"),
    ("PUNT", "8-T.Gill punts 48 yards to LA 20.", "punt"),
    ("FIELD_GOAL", "3-C.Santos 42 yard field goal is GOOD.", "field_goal"),
    ("XP_KICK", "3-C.Santos extra point is GOOD.", "extra_point"),
    ("RUSH", "9-M.Stafford kneels to LA 40 for -1 yards.", "qb_kneel"),
    (None, "Timeout #1 by CHI at 02:14.", None),
    (None, "Two-Minute Warning", None),
    ("PENALTY", "PENALTY on CHI-62-J.Thuney, False Start, 5 yards, enforced at LA 30 - No Play.", "no_play"),
])
def test_classify(ptype, desc, expected):
    assert classify(ptype, desc) == expected


def test_admin_play_needs_a_marker_and_no_real_action():
    assert st.is_admin_play("Ravens timeout.", "Offense Time Out")
    assert not st.is_admin_play("… for 21 yards. ** Injury Update: timeout", "PASS")


GAME = {
    "id": "game-uuid", "time": "2026-01-18T20:00:00Z",
    "homeTeam": {"id": "home-uuid", "currentLogo": "https://static.www.nfl.com/x/league/api/clubs/logos/CHI", "fullName": "Chicago Bears"},
    "awayTeam": {"id": "away-uuid", "currentLogo": "https://static.www.nfl.com/x/league/api/clubs/logos/LA", "fullName": "Los Angeles Rams"},
    "externalIds": [{"source": "gsis", "id": "59999"}, {"source": "slug", "id": "rams-at-bears-2025-post-2"}],
    "venue": {"name": "Soldier Field"},
    "driveChart": {
        "drives": [{"sequence": 1, "teamId": "home-uuid"}, {"sequence": 2, "teamId": "away-uuid"}],
        "plays": [
            {"playId": 40, "quarter": 1, "clockTime": "15:00", "driveSequence": 1, "playType": "KICK_OFF", "yardLine": "LA 35",
             "playDescription": "(15:00) 7-E.Karty kicks 65 yards from LA 35 to end zone, Touchback.", "stats": [{"statType": 41, "teamId": "away-uuid", "yards": 65, "gsisPlayerId": "00-k", "gsisPlayerName": "E.Karty", "gsisPlayerJerseyNumber": "7"}]},
            {"playId": 56, "quarter": 1, "clockTime": "14:53", "driveSequence": 1, "down": 1, "yardsRemaining": 10, "yardLine": "CHI 37",
             "playType": "PASS", "playDescription": "(14:53) 18-C.Williams pass short right to 2-D.Moore to LA 29 for 34 yards (25-R.McCreary).", "yardsGained": 34,
             "stats": [{"statType": 15, "teamId": "home-uuid", "yards": 34, "gsisPlayerId": "00-cw", "gsisPlayerName": "C.Williams", "gsisPlayerJerseyNumber": "18"},
                       {"statType": 21, "teamId": "home-uuid", "yards": 34, "gsisPlayerId": "00-dm", "gsisPlayerName": "D.Moore", "gsisPlayerJerseyNumber": "2"},
                       {"statType": 79, "teamId": "away-uuid", "yards": 0, "gsisPlayerId": "00-rm", "gsisPlayerName": "R.McCreary", "gsisPlayerJerseyNumber": "25"},
                       {"statType": 4, "teamId": "home-uuid"}]},
            {"playId": 70, "quarter": 1, "clockTime": "14:20", "driveSequence": 1, "playType": "TIMEOUT", "playDescription": "Timeout #1 by LA at 14:20.", "stats": []},
            {"playId": 90, "quarter": 1, "clockTime": "14:18", "driveSequence": 1, "down": 1, "yardsRemaining": 10, "yardLine": "LA 29",
             "playType": "RUSH", "playDescription": "(14:18) 4-D.Swift right guard to LA 23 for 6 yards (53-N.Landman).", "yardsGained": 6,
             "stats": [{"statType": 10, "teamId": "home-uuid", "yards": 6, "gsisPlayerId": "00-ds", "gsisPlayerName": "D.Swift", "gsisPlayerJerseyNumber": "4"}]},
            {"playId": 12, "playDeleted": True, "playDescription": "deleted"},
        ],
    },
}


def test_adapt_game_detail_builds_our_schema():
    out = adapt_game_detail(GAME, season=2025, week=20)
    g = out["game"]
    assert g["game_key"] == "2025_wk20_LA-CHI" and g["away"] == "LA" and g["home"] == "CHI"
    assert g["gsis_id"] == "59999" and g["slug"] == "rams-at-bears-2025-post-2"
    rows = out["plays"]
    assert [r["nfl_play_id"] for r in rows] == [40, 56, 70, 90]           # sorted, deleted dropped
    assert [r["play_type"] for r in rows] == ["kickoff", "pass", None, "run"]
    assert [r["play_index"] for r in rows] == [1, 2, None, 3]              # the timeout takes no index
    assert rows[1]["play_uid"] == "2025_wk20_LA-CHI_p002"
    assert rows[1]["posteam"] == "CHI" and rows[1]["defteam"] == "LA"
    assert rows[1]["yardline_100"] == 63 and rows[3]["yardline_100"] == 29
    assert rows[1]["time"] == "14:53" and rows[1]["desc"].startswith("18-C.Williams")
    roles = {(p["jersey"], p["role"]) for p in rows[1]["players"]}
    assert normalize_jersey("04") == "4" and normalize_jersey("0") == "0" and normalize_jersey(None) is None
    assert roles == {("18", "passer"), ("2", "receiver"), ("25", "tackler")}
    assert rows[1]["players"][0]["team"] == "CHI" and rows[1]["is_scrimmage"] and not rows[0]["is_scrimmage"]


def test_teams_adapter_and_logo_abbreviation():
    assert abbreviation_from_logo("https://static.www.nfl.com/{formatInstructions}/league/api/clubs/logos/PHI") == "PHI"
    t = adapt_teams({"teams": [{"id": "g1", "abbreviation": "chi", "fullName": "Chicago Bears"}, {"abbr": "LA"}]})
    assert t["teams"]["CHI"]["id"] == "g1" and t["understood"] == 1 and t["entries"] == 2


def test_roster_adapter_finds_players_inside_the_roster_object():
    payload = {"rosters": [{"season": 2025, "players": [
        {"gsisId": "00-1", "firstName": "Marquise", "commonFirstName": "Hollywood", "lastName": "Brown", "jerseyNumber": 5, "position": "WR"},
        {"gsisId": "00-2", "firstName": "Denniston", "commonFirstName": "DJ", "lastName": "Moore", "displayName": "DJ Moore", "jerseyNumber": "2", "position": "WR"},
        {"lastName": "Nobody"}]}], "pagination": {"token": ""}}
    r = adapt_roster(payload)
    assert r["entries"] == 3 and r["understood"] == 2 and r["more_pages"] is False
    assert name_forms("Marquise", "Hollywood", "Brown", "") == ["Marquise Brown", "M.Brown", "Hollywood Brown", "H.Brown"]
    assert r["players"][1]["display_name"] == "DJ Moore"
    assert set(jersey_map(r)) == {"5", "2"}


@pytest.mark.parametrize("ptype,desc", [("GAME_START", "GAME"), ("END_QUARTER", "END QUARTER 1"), ("END_GAME", "END GAME"),
                                        ("TIMEOUT", "Timeout #2 by CHI at 01:52."), ("UNKNOWN_THING", "")])
def test_bookkeeping_rows_are_never_filmable(ptype, desc):
    assert classify(ptype, desc) is None


def test_team_aliases_keep_the_rams_on_the_right_side_of_the_field():
    assert canonical_team("LAR") == "LA" and canonical_team("chi") == "CHI"
    assert yardline_100("LAR 15", "LA") == 85        # Rams' own 15 with the Rams in possession
    assert yardline_100("LAR 29", "CHI") == 29


def test_kicks_carry_no_down_or_distance():
    out = adapt_game_detail(GAME, season=2025, week=20)
    kick = out["plays"][0]
    assert kick["play_type"] == "kickoff" and kick["down"] is None and kick["ydstogo"] is None


def test_merge_labels_joins_by_play_id_and_prefers_numbered_descriptions():
    import pandas as pd
    api = pd.DataFrame([{"nfl_play_id": 56, "desc": "C.Williams pass short right to D.Moore for 34 yards.", "play_type": "pass"},
                        {"nfl_play_id": 90, "desc": "D.Swift right guard for 6 yards.", "play_type": "run"}])
    nv = pd.DataFrame([{"play_id": 56.0, "desc": "(14:53) 18-C.Williams pass short right to 2-D.Moore for 34 yards.",
                        "offense_formation": "UNDER CENTER", "offense_personnel": "1 RB, 2 TE, 2 WR", "epa": 2.05},
                       {"play_id": None, "desc": "Timeout", "offense_formation": None, "offense_personnel": None, "epa": None}])
    out = merge_labels(api, nv)
    assert out.loc[0, "offense_formation"] == "UNDER CENTER" and out.loc[0, "desc"].startswith("(14:53) 18-")
    assert out.loc[0, "desc_nfl"].startswith("C.Williams") and out.loc[0, "labels_source"] == "nflverse"
    assert pd.isna(out.loc[1, "offense_formation"]) and out.loc[1, "desc"].startswith("D.Swift") and pd.isna(out.loc[1, "labels_source"])
