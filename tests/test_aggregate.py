"""Tests for the efficiency aggregate + situational filters (no LLM needed)."""

import duckdb
import pandas as pd

from gridiron.chat.query import run_aggregate_plays, run_query_plays


def rich_con():
    # (off_grouping, formation, epa, yards_gained, shotgun, pass_location, play_type)
    rows = [
        ("11", "SHOTGUN", 0.5, 8, 1, "right", "pass"),
        ("11", "SHOTGUN", -0.3, 2, 1, "left", "pass"),
        ("11", "SHOTGUN", 1.2, 20, 1, None, "run"),
        ("12", "UNDER CENTER", -0.1, 3, 0, None, "run"),
        ("12", "UNDER CENTER", 0.2, 5, 0, None, "run"),
        ("21", "I_FORM", 2.0, 25, 0, None, "run"),
    ]
    df = pd.DataFrame([
        {"play_uid": f"p{i}", "play_index": i, "qtr": 1, "down": 1, "ydstogo": 10,
         "yardline_100": 50, "yrdln": "50", "posteam": "KC", "defteam": "DET",
         "play_type": pt, "off_grouping": g, "off_formation": fm, "offense_personnel": "x",
         "def_package": "nickel", "def_front": "4-2-5", "epa": epa, "yards_gained": yds,
         "shotgun": sg, "pass_location": pl}
        for i, (g, fm, epa, yds, sg, pl, pt) in enumerate(rows, 1)
    ])
    con = duckdb.connect()
    con.register("p", df)
    con.execute("CREATE TABLE plays AS SELECT * FROM p")
    return con


def test_aggregate_ranks_by_epa():
    out = run_aggregate_plays(rich_con(), group_by=["off_grouping"], sort_by="epa_per_play")
    groups = out["groups"]
    assert groups[0]["off_grouping"] == "21"          # epa 2.0 is highest
    assert "explosive_rate" in groups[0]              # yards_gained present -> metric included
    assert groups[0]["explosive_rate"] == 1.0         # its lone play gained 25


def test_aggregate_success_rate():
    out = run_aggregate_plays(rich_con(), group_by=["off_grouping"], sort_by="success_rate")
    by = {g["off_grouping"]: g for g in out["groups"]}
    assert round(by["11"]["success_rate"], 3) == 0.667   # 2 of 3 had EPA > 0


def test_aggregate_requires_valid_dim():
    out = run_aggregate_plays(rich_con(), group_by=["bogus"])
    assert "error" in out


def test_aggregate_min_plays():
    out = run_aggregate_plays(rich_con(), group_by=["off_grouping"], min_plays=3)
    assert len(out["groups"]) == 1 and out["groups"][0]["off_grouping"] == "11"


def test_bool_filter_shotgun():
    assert run_query_plays(rich_con(), shotgun=True, count_only=True)["count"] == 3


def test_pass_location_filter():
    assert run_query_plays(rich_con(), pass_location="right", count_only=True)["count"] == 1


def test_filter_absent_column_is_ignored():
    # `sack` column doesn't exist in this fixture -> filter is skipped, not an error.
    assert run_query_plays(rich_con(), sack=True, count_only=True)["count"] == 6
