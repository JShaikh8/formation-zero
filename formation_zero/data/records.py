"""Play records from official data alone: the first layer every play gets (blueprint 9.1).

    fz-records --season 2025 --week 20 --away LA --home CHI      # writes derived/plays/<uid>.json
    fz-export  --season 2025 --week 20 --away LA --home CHI      # derived/exports/<game>.jsonl

A client integrating against these records today keeps working as the vision and intelligence
stages fill in the null sections later.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from formation_zero.configs.nfl_field import END_ZONE_DEPTH_YD
from formation_zero.data.schemas import SCHEMA_VERSION, validate_play_record
from formation_zero.ids import parse_game_key

PIPELINE_VERSION = "0.1.0"
_CLOCK_RE = re.compile(r"^\((\d{1,2}:\d{2})\)\s*")


def _none(v):
    """pandas NaN/NaT -> None; numpy scalars -> Python."""
    try:
        import pandas as pd

        if v is None or (not isinstance(v, (list, dict, str)) and pd.isna(v)):
            return None
    except Exception:
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def _int(v):
    v = _none(v)
    return int(v) if v is not None else None


def personnel_code(s: str | None) -> str | None:
    """'1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR' -> '11'."""
    if not isinstance(s, str):
        return None
    counts = {}
    for part in s.split(","):
        m = re.match(r"\s*(\d+)\s+([A-Z]+)", part)
        if m:
            counts[m.group(2)] = int(m.group(1))
    if "RB" not in counts and "TE" not in counts:
        return None
    return f"{counts.get('RB', 0)}{counts.get('TE', 0)}"


def official_record(row: dict, game: dict, players: list[dict] | None = None) -> dict:
    """One record with the official sections filled and every vision section null."""
    desc = row.get("desc") or ""
    m = _CLOCK_RE.match(desc)
    clock = _none(row.get("time")) or (m.group(1) if m else None)
    if m:
        desc = desc[m.end():]
    yl = _int(row.get("yardline_100"))
    los_x = END_ZONE_DEPTH_YD + (100 - yl) if yl is not None else None
    dist = _int(row.get("ydstogo"))
    labels = {k: _none(row.get(src)) for k, src in (("offense_formation", "offense_formation"),
                                                   ("offense_personnel", "offense_personnel"),
                                                   ("defense_personnel", "defense_personnel"))}
    has_labels = any(labels.values())
    grouping = personnel_code(labels["offense_personnel"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": SCHEMA_VERSION,
        "play_uid": row["play_uid"],
        "game": {"game_key": game["game_key"], "season": int(game["season"]), "week": int(game["week"]),
                 "away": game["away"], "home": game["home"], "nfl_game_id": game.get("nfl_game_id"),
                 "nfl_play_id": _int(row.get("nfl_play_id")), "gsis_id": game.get("gsis_id")},
        "situation": {"quarter": _int(row.get("qtr")), "clock": clock, "down": _int(row.get("down")), "distance": dist,
                      "possession": _none(row.get("posteam")), "yardline_100": yl, "yrdln": _none(row.get("yrdln")),
                      "los_x": los_x, "first_down_x": (min(110.0, los_x + dist) if los_x is not None and dist is not None else None),
                      "direction_of_attack": "right", "hash": None, "score": None},
        "official": {"play_type": _none(row.get("play_type")), "description": desc,
                     "yards_gained": _int(row.get("yards_gained")),
                     "first_down": bool(_none(row.get("first_down")) or False), "touchdown": bool(_none(row.get("touchdown")) or False),
                     "turnover": bool(_none(row.get("interception")) or False),
                     "players_involved": players or [],
                     "labels": {**{k: (str(v) if v is not None else None) for k, v in labels.items()},
                                "offense_personnel": grouping or labels["offense_personnel"],
                                "source": "nflverse" if has_labels else "official"}},
        "film": None,
        "personnel": ({"offense": {"grouping": grouping, "rb": int(grouping[0]) if grouping else None,
                                   "te": int(grouping[1]) if grouping else None,
                                   "wr": (5 - int(grouping[0]) - int(grouping[1])) if grouping else None,
                                   "source": "nflverse", "confidence": 1.0, "pass": "official"},
                       "defense": {"package": None, "dl": None, "lb": None, "db": None, "source": "nflverse",
                                   "confidence": None, "pass": "official"}} if grouping else None),
        "pre_snap": None, "at_snap": None, "live": None, "post_snap": None,
        "players": [], "tracking": None,
        "quality": {"note": "official sections only; vision analysis pending"},
        "review": {"status": "unreviewed", "by": None, "at": None},
        "corrections": [],
        "generated": {"pipeline_version": PIPELINE_VERSION, "at": now, "pass": "official"},
    }


def build_records(pbp, game: dict, players_by_uid: dict | None = None) -> list[dict]:
    rows = pbp[pbp["play_uid"].notna()].sort_values("play_index")
    out = []
    for r in rows.to_dict("records"):
        rec = official_record(r, game, (players_by_uid or {}).get(r["play_uid"]))
        problems = validate_play_record(rec)
        if problems:
            raise ValueError(f"{rec['play_uid']}: " + "; ".join(problems[:3]))
        out.append(rec)
    return out


def write_records(records: list[dict], plays_dir: Path) -> int:
    plays_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (plays_dir / f"{rec['play_uid']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return len(records)


def export_jsonl(plays_dir: Path, out: Path, game_key: str) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for p in sorted(plays_dir.glob(f"{game_key}_p*.json")):
            fh.write(json.dumps(json.loads(p.read_text()), ensure_ascii=False) + "\n")
            n += 1
    return n


def _game_from_key(game_key: str, pbp_path: Path | None) -> dict:
    """Game metadata: the NFL API's game.json beside the play table when present, else the key."""
    g = {"game_key": game_key, **parse_game_key(game_key)}
    if pbp_path is not None:
        meta = pbp_path.with_suffix(".game.json")
        if meta.exists():
            g.update({k: v for k, v in json.loads(meta.read_text()).items() if k in ("nfl_game_id", "gsis_id", "slug", "venue", "kickoff")})
    return g


def main_records(argv: list[str] | None = None) -> int:
    import argparse

    import pandas as pd

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description="Write a play record for every play from official data alone.")
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True); ap.add_argument("--home", required=True)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home).ensure_dirs()
    pbp = pd.read_parquet(paths.pbp_path)
    players_path = paths.pbp_path.with_suffix(".players.json")
    players = json.loads(players_path.read_text()) if players_path.exists() else {}
    game = _game_from_key(paths.game_key, paths.pbp_path)
    n = write_records(build_records(pbp, game, players), paths.plays_dir)
    print(f"{n} play records -> {paths.plays_dir}")
    return 0


def main_export(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description="Export a game's play records as JSONL, one record per line.")
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True); ap.add_argument("--home", required=True)
    ap.add_argument("--data-root", default="data"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    out = Path(a.out) if a.out else paths.export_path
    n = export_jsonl(paths.plays_dir, out, paths.game_key)
    print(f"{n} records -> {out}")
    return 0


__all__ = ["official_record", "build_records", "write_records", "export_jsonl", "personnel_code", "PIPELINE_VERSION"]
