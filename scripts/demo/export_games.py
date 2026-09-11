"""Export the demo's game list and play lists from the free play-by-play files.

Reads data/pbp/<game_key>.parquet (nflverse) for the games listed below and writes
site/demo/data/games.js -> window.FZ_DEMO = { games: [...] }. The play list is the real
official one; a play is `featured` when a hand-made record exists for it in records.js.

Run:  .venv/bin/python scripts/demo/export_games.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "demo" / "data" / "games.js"
RECORDS = ROOT / "site" / "demo" / "data" / "records.js"

TEAMS = {
    "LA": {"abbr": "LA", "name": "Los Angeles Rams", "short": "Rams", "color": "#FFD100", "ink": "#003594"},
    "CHI": {"abbr": "CHI", "name": "Chicago Bears", "short": "Bears", "color": "#C83803", "ink": "#FFFFFF"},
    "DET": {"abbr": "DET", "name": "Detroit Lions", "short": "Lions", "color": "#0076B6", "ink": "#FFFFFF"},
    "KC": {"abbr": "KC", "name": "Kansas City Chiefs", "short": "Chiefs", "color": "#E31837", "ink": "#FFB81C"},
}

GAMES = [
    {"game_key": "2025_wk20_LA-CHI", "season": 2025, "week": 20, "label": "Divisional round", "date": "2026-01-18",
     "stadium": "Soldier Field", "away": "LA", "home": "CHI", "score": {"away": 20, "home": 17}, "overtime": True,
     "film": True},
    {"game_key": "2023_wk01_DET-KC", "season": 2023, "week": 1, "label": "Week 1, season opener", "date": "2023-09-07",
     "stadium": "GEHA Field at Arrowhead", "away": "DET", "home": "KC", "score": {"away": 21, "home": 20}, "overtime": False,
     "film": False},
]

_CLOCK = re.compile(r"^\((\d{1,2}:\d{2})\)\s*")


def personnel_code(s: str | None) -> str | None:
    """'1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR' -> '11' (running backs then tight ends)."""
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


def defense_code(s: str | None) -> str | None:
    if not isinstance(s, str):
        return None
    dl = lb = db = 0
    for part in s.split(","):
        m = re.match(r"\s*(\d+)\s+([A-Z]+)", part)
        if not m:
            continue
        n, pos = int(m.group(1)), m.group(2)
        if pos in ("DE", "DT", "NT"):
            dl += n
        elif pos in ("ILB", "OLB", "MLB", "LB"):
            lb += n
        elif pos in ("CB", "FS", "SS", "S", "DB"):
            db += n
    return f"{dl}-{lb}-{db}"


def featured_uids() -> set[str]:
    if not RECORDS.exists():
        return set()
    text = RECORDS.read_text()
    return set(re.findall(r'"(\d{4}_wk\d{2}_[A-Z]{2,3}-[A-Z]{2,3}_p\d{3})":\{"schema_version"', text))


def export_game(meta: dict, featured: set[str]) -> dict:
    df = pd.read_parquet(ROOT / "data" / "pbp" / f"{meta['game_key']}.parquet")
    df = df.sort_values("play_index")
    plays = []
    drive = 0
    last_pos = None
    for r in df.itertuples():
        # Administrative rows (timeouts, end of quarter, two-minute warning) carry no play id
        # and put nothing on film; the list is plays only.
        if not isinstance(r.play_uid, str) or pd.isna(r.play_index):
            continue
        desc = r.desc if isinstance(r.desc, str) else ""
        m = _CLOCK.match(desc)
        clock = m.group(1) if m else None
        if m:
            desc = desc[m.end():]
        pos = r.posteam if isinstance(r.posteam, str) else None
        if pos and pos != last_pos:
            drive += 1
            last_pos = pos
        plays.append({
            "uid": r.play_uid, "idx": int(r.play_index), "qtr": int(r.qtr) if pd.notna(r.qtr) else None,
            "clock": clock,
            "down": int(r.down) if pd.notna(r.down) else None,
            "ydstogo": int(r.ydstogo) if pd.notna(r.ydstogo) else None,
            "yardline_100": int(r.yardline_100) if pd.notna(r.yardline_100) else None,
            "yrdln": r.yrdln if isinstance(r.yrdln, str) else None,
            "posteam": pos, "defteam": r.defteam if isinstance(r.defteam, str) else None,
            "play_type": r.play_type if isinstance(r.play_type, str) else None,
            "yards": int(r.yards_gained) if pd.notna(r.yards_gained) else None,
            "desc": desc,
            "formation": r.offense_formation if isinstance(r.offense_formation, str) else None,
            "personnel": personnel_code(r.offense_personnel),
            "defense": defense_code(r.defense_personnel),
            "td": bool(r.touchdown == 1) if pd.notna(r.touchdown) else False,
            "first_down": bool(r.first_down == 1) if pd.notna(r.first_down) else False,
            "epa": round(float(r.epa), 2) if pd.notna(r.epa) else None,
            "scrimmage": bool(r.is_scrimmage),
            "drive": drive,
            "featured": r.play_uid in featured,
        })
    return {**meta, "away": TEAMS[meta["away"]], "home": TEAMS[meta["home"]],
            "plays": plays, "play_count": len(plays), "featured_count": sum(p["featured"] for p in plays)}


def main() -> int:
    featured = featured_uids()
    games = [export_game(g, featured) for g in GAMES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"games": games}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text("// Generated by scripts/demo/export_games.py from the free play-by-play files.\n"
                   f"window.FZ_DEMO = {payload};\n")
    for g in games:
        print(f"{g['game_key']}: {g['play_count']} plays, {g['featured_count']} featured")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
