"""Team GUIDs from /experience/v1/teams. Rosters are addressed by GUID, not abbreviation."""

from __future__ import annotations

ID_KEYS = ("id", "teamId", "guid")
ABBR_KEYS = ("abbreviation", "abbr", "teamAbbr", "nickAbbr", "triCode")
NAME_KEYS = ("fullName", "displayName", "name", "nickName")


def _first(rec: dict, keys) -> str:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)):
            return str(v)
    return ""


def _records(payload) -> list[dict]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for k in ("teams", "data", "items"):
            if isinstance(payload.get(k), list):
                return [r for r in payload[k] if isinstance(r, dict)]
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return [r for r in v if isinstance(r, dict)]
    return []


def adapt_teams(payload) -> dict:
    """{'teams': {ABBR: {'id', 'name'}}, 'entries', 'understood'}. Never throws."""
    recs = _records(payload)
    teams = {}
    for r in recs:
        tid, abbr = _first(r, ID_KEYS), _first(r, ABBR_KEYS).upper()
        if tid and abbr:
            teams[abbr] = {"id": tid, "name": _first(r, NAME_KEYS)}
    return {"teams": teams, "entries": len(recs), "understood": len(teams)}


def abbreviation_from_logo(url: str | None) -> str:
    """The team code sits in the logo URL of a game detail payload: .../clubs/logos/PHI"""
    import re

    m = re.search(r"/logos/([A-Z]+)(?:[/?#]|$)", url or "")
    return m.group(1) if m else ""


__all__ = ["adapt_teams", "abbreviation_from_logo"]
