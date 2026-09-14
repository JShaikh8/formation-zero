"""Play record v1: schema, official-only records, export."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from formation_zero.data.records import build_records, export_jsonl, official_record, personnel_code, write_records
from formation_zero.data.schemas import SCHEMA_VERSION, play_record_schema, validate_play_record
from formation_zero.intelligence import vocab

GAME = {"game_key": "2025_wk20_LA-CHI", "season": 2025, "week": 20, "away": "LA", "home": "CHI"}
ROW = {"play_uid": "2025_wk20_LA-CHI_p001", "play_index": 1, "qtr": 1, "down": 1, "ydstogo": 10, "yardline_100": 63,
       "yrdln": "CHI 37", "posteam": "CHI", "play_type": "pass", "yards_gained": 34, "first_down": 1.0, "touchdown": 0.0,
       "desc": "(14:53) 18-C.Williams pass short right to 2-D.Moore to LA 29 for 34 yards (25-R.McCreary).",
       "offense_formation": "UNDER CENTER", "offense_personnel": "1 C, 3 G, 1 QB, 1 RB, 1 T, 2 TE, 2 WR",
       "defense_personnel": "2 CB, 1 DE, 1 DT, 1 FS, 2 ILB, 1 NT, 2 OLB, 1 SS"}


def test_schema_loads_and_pins_vocabularies():
    s = play_record_schema()
    assert s["properties"]["schema_version"]["pattern"].startswith("^1")
    routes = s["properties"]["live"]["properties"]["routes"]["items"]["properties"]["route"]["enum"]
    assert set(routes) == set(vocab.ROUTES)


def test_official_record_fills_official_sections_and_nulls_the_rest():
    rec = official_record(ROW, GAME, players=[{"jersey": "18", "team": "CHI", "role": "passer"}])
    assert validate_play_record(rec) == []
    assert rec["schema_version"] == SCHEMA_VERSION and rec["generated"]["pass"] == "official"
    assert rec["situation"]["clock"] == "14:53" and rec["official"]["description"].startswith("18-C.Williams")
    assert rec["situation"]["los_x"] == 47 and rec["situation"]["first_down_x"] == 57
    assert rec["official"]["labels"]["offense_personnel"] == "12" and rec["personnel"]["offense"]["wr"] == 2
    assert rec["official"]["players_involved"][0]["role"] == "passer"
    assert rec["pre_snap"] is None and rec["live"] is None and rec["players"] == []
    assert rec["review"]["status"] == "unreviewed" and rec["corrections"] == []


def test_validation_rejects_bad_vocabulary():
    rec = official_record(ROW, GAME)
    rec["live"] = {"routes": [{"track_id": 1, "route": "banana"}], "events": [{"frame": 3, "type": "snap"}]}
    problems = validate_play_record(rec)
    assert problems and "banana" in problems[0]


def test_build_write_export_roundtrip(tmp_path):
    pbp = pd.DataFrame([ROW, {**ROW, "play_uid": "2025_wk20_LA-CHI_p002", "play_index": 2, "desc": "(14:18) 4-D.Swift right guard to LA 23 for 6 yards.", "play_type": "run"},
                        {**ROW, "play_uid": None, "play_index": None, "desc": "Timeout", "play_type": None}])
    recs = build_records(pbp, GAME)
    assert [r["play_uid"] for r in recs] == ["2025_wk20_LA-CHI_p001", "2025_wk20_LA-CHI_p002"]
    n = write_records(recs, tmp_path / "plays")
    assert n == 2 and (tmp_path / "plays" / "2025_wk20_LA-CHI_p002.json").exists()
    m = export_jsonl(tmp_path / "plays", tmp_path / "out.jsonl", "2025_wk20_LA-CHI")
    lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
    assert m == 2 and json.loads(lines[1])["official"]["play_type"] == "run"


def test_personnel_code():
    assert personnel_code("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR") == "11"
    assert personnel_code("1 C, 1 FB, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 2 WR") == "11"
    assert personnel_code(None) is None
