"""Stable identifiers and game/play keys used across the whole pipeline.

Conventions (see the build plan):
  game_key  = "2024_wk05_BUF-NYJ"          (season_wkNN_AWAY-HOME)
  play_uid  = "2024_wk05_BUF-NYJ_p017"     (game_key + zero-padded play index)

These join the video segments, the PBP rows, and the derived tracking/labels together.
nflverse `game_id` uses "2024_05_BUF_NYJ" (underscores); `nflverse_game_id()` converts.
"""

from __future__ import annotations

import re

_GAME_KEY_RE = re.compile(r"^(\d{4})_wk(\d{2})_([A-Z]{2,3})-([A-Z]{2,3})$")


def game_key(season: int, week: int, away: str, home: str) -> str:
    """Canonical game key, e.g. game_key(2024, 5, "BUF", "NYJ") -> '2024_wk05_BUF-NYJ'."""
    return f"{season}_wk{week:02d}_{away.upper()}-{home.upper()}"


def play_uid(game_key_str: str, play_index: int) -> str:
    """Per-play id, e.g. play_uid('2024_wk05_BUF-NYJ', 17) -> '2024_wk05_BUF-NYJ_p017'."""
    return f"{game_key_str}_p{play_index:03d}"


def parse_game_key(game_key_str: str) -> dict:
    """Inverse of game_key(): returns {season, week, away, home}."""
    m = _GAME_KEY_RE.match(game_key_str)
    if not m:
        raise ValueError(f"not a valid game_key: {game_key_str!r}")
    season, week, away, home = m.groups()
    return {"season": int(season), "week": int(week), "away": away, "home": home}


def nflverse_game_id(season: int, week: int, away: str, home: str) -> str:
    """nflverse's game_id format, e.g. '2024_05_BUF_NYJ' (used to filter PBP)."""
    return f"{season}_{week:02d}_{away.upper()}_{home.upper()}"
