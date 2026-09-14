"""The game-detail payload -> our per-play schema (the same one nflverse feeds).

From `driveChart.plays[]` we take `playId`, quarter, clock, down, `yardsRemaining`, `yardLine`
(team-relative, e.g. "PHI 1"), `playType`, `playDescription`, `playScored`, `playDeleted`,
`driveSequence`, and `stats[]` (statType, teamId, yards, gsisPlayerId, gsisPlayerName,
gsisPlayerJerseyNumber). Plays are joined to drives by `driveSequence` (which gives possession),
never by index. Rows are sorted by `playId`.

`play_index` counts filmable plays in order, exactly as `formation_zero.data.pbp` does, so the
play index built from film joins to this table without change.
"""

from __future__ import annotations

import re

from formation_zero.data.nflapi import stat_types as st
from formation_zero.data.nflapi.teams import abbreviation_from_logo
from formation_zero.ids import game_key as make_game_key
from formation_zero.ids import play_uid as make_play_uid

_YARDLINE_RE = re.compile(r"^\s*([A-Za-z]{2,3})\s*(\d{1,2})\s*$")

# The NFL spells a few clubs differently from the free data (and our game keys). Canonical = ours.
TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}

# Bookkeeping rows the drive chart carries that are not plays.
ADMIN_PLAY_TYPES = {"GAME_START", "END_QUARTER", "END_HALF", "END_GAME", "TIMEOUT", "TWO_MINUTE_WARNING", "COMMENT"}


def normalize_jersey(v) -> str | None:
    """'04' -> '4'; '0' stays '0'. The feed zero-pads; jerseys and rosters do not."""
    if v is None:
        return None
    t = str(v).strip().lstrip("0")
    return t or "0"


def canonical_team(abbr: str | None) -> str | None:
    if not abbr:
        return None
    a = abbr.upper()
    return TEAM_ALIASES.get(a, a)
_CLOCK_RE = re.compile(r"^\((\d{1,2}:\d{2})\)\s*")

FILMABLE = ("pass", "run", "punt", "field_goal", "extra_point", "kickoff", "qb_kneel", "qb_spike")


def yardline_100(yardline: str | None, posteam: str | None) -> int | None:
    """'CHI 37' with CHI in possession -> 63; 'LA 29' with CHI in possession -> 29; '50' -> 50."""
    if not yardline:
        return None
    s = yardline.strip()
    if s == "50":
        return 50
    m = _YARDLINE_RE.match(s)
    if not m:
        return None
    side, n = canonical_team(m.group(1)), int(m.group(2))
    if posteam and side == canonical_team(posteam):
        return 100 - n
    return n


def classify(play_type: str | None, desc: str) -> str | None:
    """NFL playType + description -> nflverse-style play_type. None for administrative rows."""
    d = (desc or "").lower()
    t = (play_type or "").upper()
    if t in ADMIN_PLAY_TYPES or st.is_admin_play(desc, play_type):
        return None
    if "no play" in d and "penalty" in d:
        return "no_play"
    if "kicks off" in d or "kickoff" in d or "KICK_OFF" in t or t == "KICKOFF":
        return "kickoff"
    if "punts" in d or t == "PUNT":
        return "punt"
    if "field goal" in d or t in ("FIELD_GOAL", "FG"):
        return "field_goal"
    if "extra point" in d or t in ("XP_KICK", "EXTRA_POINT", "PAT"):
        return "extra_point"
    if "kneels" in d:
        return "qb_kneel"
    if "spiked" in d:
        return "qb_spike"
    if "two-point" in d or "two point" in d:
        return "pass" if "pass" in d else "run"
    if "scrambles" in d:
        return "run"          # the feed says PASS; football (and nflverse) say a scramble is a run
    if "pass" in d or "sacked" in d or t in ("PASS", "SACK"):
        return "pass"
    if t in ("RUSH", "RUN") or re.search(r"\b(left|right) (end|tackle|guard)\b|up the middle", d):
        return "run"
    if t == "PENALTY" or "penalty" in d:
        return "no_play"
    return None          # unknown: never invent a filmable play


def adapt_game_detail(payload: dict, *, season: int, week: int) -> dict:
    """-> {'game': {...}, 'plays': [row, ...]} with rows in our tidy schema plus `players`."""
    home, away = payload.get("homeTeam") or {}, payload.get("awayTeam") or {}
    home_abbr = canonical_team(abbreviation_from_logo(home.get("currentLogo")) or home.get("abbreviation"))
    away_abbr = canonical_team(abbreviation_from_logo(away.get("currentLogo")) or away.get("abbreviation"))
    team_by_id = {home.get("id"): home_abbr, away.get("id"): away_abbr}
    gk = make_game_key(season, week, away_abbr, home_abbr)
    chart = payload.get("driveChart") or {}
    drive_team = {d.get("sequence"): team_by_id.get(d.get("teamId")) for d in (chart.get("drives") or [])}
    plays = [p for p in (chart.get("plays") or []) if isinstance(p, dict) and p.get("playDeleted") is not True
             and p.get("playId") is not None]
    plays.sort(key=lambda p: p["playId"])

    rows = []
    index = 0
    for p in plays:
        desc = p.get("playDescription") or ""
        m = _CLOCK_RE.match(desc)
        clock = p.get("clockTime") or (m.group(1) if m else None)
        if m:
            desc = desc[m.end():]
        posteam = drive_team.get(p.get("driveSequence"))
        defteam = away_abbr if posteam == home_abbr else home_abbr if posteam == away_abbr else None
        ptype = classify(p.get("playType"), desc)
        filmable = ptype in FILMABLE
        if filmable:
            index += 1
        players = []
        for s in p.get("stats") or []:
            code = s.get("statType")
            try:
                code = int(code)
            except (TypeError, ValueError):
                continue
            role = st.ROLE_BY_CODE.get(code)
            if role:
                players.append({"gsis_id": s.get("gsisPlayerId"), "name": s.get("gsisPlayerName"),
                                "jersey": normalize_jersey(s.get("gsisPlayerJerseyNumber")),
                                "team": team_by_id.get(s.get("teamId")), "role": role, "stat_type": code,
                                "yards": s.get("yards")})
        rows.append({
            "game_key": gk, "play_uid": make_play_uid(gk, index) if filmable else None,
            "play_index": index if filmable else None, "nfl_play_id": p.get("playId"),
            "drive_id": p.get("driveSequence"), "qtr": p.get("quarter"), "time": clock,
            "down": p.get("down") or None, "ydstogo": (p.get("yardsRemaining") if p.get("down") else None),
            "yrdln": p.get("yardLine"),
            "yardline_100": yardline_100(p.get("yardLine"), posteam),
            "posteam": posteam, "defteam": defteam, "play_type": ptype, "nfl_play_type": p.get("playType"),
            "desc": desc, "yards_gained": p.get("yardsGained"), "touchdown": bool(p.get("playScored")) and "touchdown" in desc.lower(),
            "is_scrimmage": ptype in ("pass", "run", "qb_kneel", "qb_spike"), "players": players,
        })
    game = {"game_key": gk, "season": season, "week": week, "away": away_abbr, "home": home_abbr,
            "nfl_game_id": payload.get("id"), "gsis_id": next((e.get("id") for e in (payload.get("externalIds") or []) if e.get("source") == "gsis"), None),
            "slug": next((e.get("id") for e in (payload.get("externalIds") or []) if e.get("source") == "slug"), None),
            "venue": (payload.get("venue") or {}).get("name"), "kickoff": payload.get("time")}
    return {"game": game, "plays": rows}


def to_frame(adapted: dict):
    import pandas as pd

    return pd.DataFrame(adapted["plays"])


__all__ = ["adapt_game_detail", "classify", "yardline_100", "canonical_team", "normalize_jersey", "to_frame", "FILMABLE", "ADMIN_PLAY_TYPES", "TEAM_ALIASES"]
