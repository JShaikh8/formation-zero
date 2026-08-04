"""STATS team registry: numeric `teamId` <-> nflverse abbreviation.

STATS identifies teams by integer `teamId` (Bears=326, Rams=343), not by letters. Its
`abbreviation` field exists but is inconsistently cased ("Chi", "LAR", "NE") and disagrees with
nflverse in one place, so `teamId` is the join key and abbreviations are a convenience layer.

The registry is fetched from the `/teams/` endpoint and cached under
``data/pbp/stats_raw/teams_<season>.json``; a static id->abbrev table backs it up so parsing
works offline and in tests.
"""

from __future__ import annotations

import json
from pathlib import Path

# nflverse spells the Rams "LA"; STATS spells them "LAR". Every other abbreviation matches
# after upper-casing.
_STATS_TO_NFLVERSE = {"LAR": "LA"}

# teamId -> STATS abbreviation, for the 32 current franchises. Only a fallback: the live
# `/teams/` payload wins when available.
STATIC_TEAM_IDS: dict[int, str] = {
    323: "Atl", 324: "Buf", 325: "Hou", 326: "Chi", 327: "Cin", 329: "Cle", 331: "Dal",
    332: "Den", 334: "Det", 335: "GB", 336: "Ten", 338: "Ind", 339: "KC", 341: "LV",
    343: "LAR", 345: "Mia", 347: "Min", 348: "NE", 350: "NO", 351: "NYG", 352: "NYJ",
    354: "Phi", 355: "Ari", 356: "Pit", 357: "LAC", 359: "SF", 361: "Sea", 362: "TB",
    363: "Was", 364: "Car", 365: "Jax", 366: "Bal",
}


def to_nflverse(abbreviation: str) -> str:
    """STATS abbreviation -> nflverse abbreviation ('Chi'->'CHI', 'LAR'->'LA')."""
    upper = abbreviation.upper()
    return _STATS_TO_NFLVERSE.get(upper, upper)


class TeamRegistry:
    """Lookup of STATS teams by id, abbreviation, nickname, or location."""

    def __init__(self, teams: list[dict]):
        self.teams = teams
        self._by_id = {int(t["teamId"]): t for t in teams if t.get("teamId") is not None}

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_payload(cls, payload: dict) -> "TeamRegistry":
        """Build from a `/teams/` response, walking conferences -> divisions -> teams."""
        teams: list[dict] = []
        seen: set[int] = set()
        for node in _walk(payload):
            tid = node.get("teamId")
            # Team records carry a nickname; event-participant records carry scores instead.
            if tid is None or "nickname" not in node or int(tid) in seen:
                continue
            seen.add(int(tid))
            teams.append(node)
        return cls(teams)

    @classmethod
    def fallback(cls) -> "TeamRegistry":
        """Registry built from the static id->abbrev table (no network)."""
        return cls([{"teamId": tid, "abbreviation": abbr} for tid, abbr in STATIC_TEAM_IDS.items()])

    # -- lookups -----------------------------------------------------------------------

    def by_id(self, team_id: int | str | None) -> dict | None:
        if team_id is None:
            return None
        return self._by_id.get(int(team_id))

    def abbreviation(self, team_id: int | str | None, *, nflverse: bool = True) -> str | None:
        """Abbreviation for a teamId, nflverse-normalized by default."""
        team = self.by_id(team_id)
        if not team:
            return None
        abbr = team.get("abbreviation")
        if not abbr:
            return None
        return to_nflverse(abbr) if nflverse else abbr

    def resolve(self, needle: str | int) -> list[dict]:
        """Find teams by id, abbreviation, nickname, or location. Exact, case-insensitive.

        Exact matching is deliberate: substring matching makes "CHI" also match the Chiefs,
        which silently returns the wrong game.
        """
        if isinstance(needle, int) or (isinstance(needle, str) and needle.isdigit()):
            team = self.by_id(needle)
            return [team] if team else []
        text = str(needle).strip().lower()
        hits = []
        for team in self.teams:
            names = {
                str(team.get(key, "")).lower()
                for key in ("abbreviation", "nickname", "location")
                if team.get(key)
            }
            abbr = team.get("abbreviation")
            if abbr:
                names.add(to_nflverse(abbr).lower())
            if abbr and str(team.get("location", "")) and team.get("nickname"):
                names.add(f"{team['location']} {team['nickname']}".lower())
            if text in names:
                hits.append(team)
        return hits

    def resolve_id(self, needle: str | int) -> int:
        """Resolve to exactly one teamId, or raise with the ambiguity spelled out."""
        hits = self.resolve(needle)
        if not hits:
            raise SystemExit(f"No team matches {needle!r}. Try an id (326), abbrev (CHI), or nickname (Bears).")
        if len(hits) > 1:
            listed = ", ".join(f"{t.get('nickname')}({t.get('teamId')})" for t in hits)
            raise SystemExit(f"{needle!r} is ambiguous: {listed}")
        return int(hits[0]["teamId"])


def _walk(node, depth: int = 0):
    """Yield every dict nested in a JSON structure (depth-bounded)."""
    if depth > 12:
        return
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value, depth + 1)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, depth + 1)


def load_registry(season: int = 2025, data_root: str | Path = "data",
                  fetch: bool = True) -> TeamRegistry:
    """Cached registry: disk cache, then the API, then the static fallback."""
    cache = Path(data_root) / "pbp" / "stats_raw" / f"teams_{season}.json"
    if cache.exists():
        return TeamRegistry.from_payload(json.loads(cache.read_text()))
    if fetch:
        try:
            from gridiron.data.statsapi import fetch_teams, save_raw

            payload = fetch_teams(season)
            save_raw(payload, f"teams_{season}", data_root)
            return TeamRegistry.from_payload(payload)
        except SystemExit:
            pass  # no credentials / network — fall through
    return TeamRegistry.fallback()


__all__ = ["TeamRegistry", "load_registry", "to_nflverse", "STATIC_TEAM_IDS"]
