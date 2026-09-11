"""Tests for the deterministic structured-query layer (no LLM / no API key needed)."""

import duckdb
import pandas as pd

from formation_zero.chat.query import run_query_plays


def make_con():
    df = pd.DataFrame([
        {"play_uid": "g_p001", "play_index": 1, "qtr": 1, "down": 3, "ydstogo": 8,
         "yardline_100": 75, "yrdln": "KC 25", "posteam": "KC", "defteam": "DET",
         "play_type": "pass", "off_formation": "SHOTGUN", "off_grouping": "11",
         "offense_personnel": "1 RB, 1 TE, 3 WR", "def_package": "nickel", "def_front": "4-2-5"},
        {"play_uid": "g_p002", "play_index": 2, "qtr": 1, "down": 1, "ydstogo": 10,
         "yardline_100": 15, "yrdln": "DET 15", "posteam": "KC", "defteam": "DET",
         "play_type": "run", "off_formation": "UNDER CENTER", "off_grouping": "21",
         "offense_personnel": "2 RB, 1 TE, 2 WR", "def_package": "base", "def_front": "4-3-4"},
        {"play_uid": "g_p003", "play_index": 3, "qtr": 2, "down": 3, "ydstogo": 2,
         "yardline_100": 40, "yrdln": "DET 40", "posteam": "DET", "defteam": "KC",
         "play_type": "run", "off_formation": "SHOTGUN", "off_grouping": "11",
         "offense_personnel": "1 RB, 1 TE, 3 WR", "def_package": "nickel", "def_front": "4-2-5"},
    ])
    con = duckdb.connect()
    con.register("p", df)
    con.execute("CREATE TABLE plays AS SELECT * FROM p")
    return con


def test_filter_team_and_grouping():
    out = run_query_plays(make_con(), offense="KC", off_grouping="11")
    assert out["count"] == 1
    assert out["plays"][0]["play_uid"] == "g_p001"


def test_formation_is_case_insensitive():
    out = run_query_plays(make_con(), off_formation="shotgun", count_only=True)
    assert out["count"] == 2


def test_third_and_long():
    out = run_query_plays(make_con(), down=3, distance_min=7, count_only=True)
    assert out["count"] == 1


def test_third_and_short():
    out = run_query_plays(make_con(), down=3, distance_max=2, count_only=True)
    assert out["count"] == 1


def test_red_zone():
    assert run_query_plays(make_con(), red_zone=True, count_only=True)["count"] == 1
    assert run_query_plays(make_con(), red_zone=False, count_only=True)["count"] == 2


def test_play_type_and_package():
    out = run_query_plays(make_con(), play_type="run", def_package="nickel", count_only=True)
    assert out["count"] == 1


def test_limit_caps_rows():
    out = run_query_plays(make_con(), limit=2)
    assert out["count"] == 3 and out["returned"] == 2
