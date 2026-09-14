"""Rosters from /football/v2/rosters: jersey, position and every name form per gsisId.

Players nest INSIDE the roster object (`{rosters: [{...players...}], pagination}`), not beside
it. Key names are candidates read off one payload; the adapter reports how many entries it
understood so a drift shows as `understood: 0` on a non-empty roster instead of silence.
"""

from __future__ import annotations

ID_KEYS = ("gsisId", "gsisPlayerId", "gsis_id")
PERSON_KEYS = ("personId", "person_id", "nflId")
FIRST_KEYS = ("firstName", "first_name", "givenName")
COMMON_KEYS = ("commonFirstName", "common_first_name", "preferredFirstName", "nickName")
LAST_KEYS = ("lastName", "last_name", "familyName")
DISPLAY_KEYS = ("displayName", "display_name", "footballName")
JERSEY_KEYS = ("jerseyNumber", "jersey", "uniformNumber", "number")
POSITION_KEYS = ("position", "positionAbbr", "positionAbbreviation")
LIST_KEYS = ("data", "players", "roster", "rosters", "athletes", "entries")


def _first(rec: dict, keys) -> str:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
    return ""


def _looks_like_player(rec: dict) -> bool:
    p = rec.get("player") if isinstance(rec.get("player"), dict) else rec
    return bool((_first(p, ID_KEYS) or _first(rec, ID_KEYS)) and _first(p, LAST_KEYS))


def _candidate_lists(value) -> list[list]:
    out: list[list] = []
    if isinstance(value, list):
        out.append(value)
    if isinstance(value, dict):
        for k in LIST_KEYS:
            if isinstance(value.get(k), list):
                out.append(value[k])
        for k, v in value.items():
            if isinstance(v, list) and k not in LIST_KEYS:
                out.append(v)
    return out


def _find_players(value, depth: int = 0) -> list | None:
    cands = _candidate_lists(value)
    for lst in cands:
        if any(isinstance(e, dict) and _looks_like_player(e) for e in lst):
            return lst
    if depth >= 3:
        return None
    for lst in cands:
        for e in lst:
            inner = _find_players(e, depth + 1)
            if inner is not None:
                return inner
    return None


def name_forms(first: str, common: str, last: str, display: str) -> list[str]:
    if not last:
        return []
    out: list[str] = []

    def push(s: str) -> None:
        t = s.strip()
        if t and t not in out:
            out.append(t)

    push(display)
    for given in (first, common):
        if given:
            push(f"{given} {last}")
            push(f"{given[0]}.{last}")
    return out


def adapt_roster(payload) -> dict:
    """{'players': [...], 'entries', 'understood', 'more_pages'}. Never throws."""
    lst = _find_players(payload)
    if lst is None:
        lst = (_candidate_lists(payload) or [[]])[0]
    players = []
    for entry in lst:
        if not isinstance(entry, dict):
            continue
        p = entry.get("player") if isinstance(entry.get("player"), dict) else entry
        gsis = _first(p, ID_KEYS) or _first(entry, ID_KEYS)
        if not gsis:
            continue
        first, common, last, display = _first(p, FIRST_KEYS), _first(p, COMMON_KEYS), _first(p, LAST_KEYS), _first(p, DISPLAY_KEYS)
        forms = name_forms(first, common, last, display)
        if not forms:
            continue
        players.append({"gsis_id": gsis, "person_id": _first(p, PERSON_KEYS) or _first(entry, PERSON_KEYS),
                        "jersey": _first(p, JERSEY_KEYS) or _first(entry, JERSEY_KEYS),
                        "position": _first(p, POSITION_KEYS) or _first(entry, POSITION_KEYS),
                        "forms": forms, "display_name": display or f"{common or first} {last}".strip()})
    pag = payload.get("pagination") if isinstance(payload, dict) else None
    more = bool(isinstance(pag, dict) and any(isinstance(pag.get(k), str) and pag[k].strip() for k in ("token", "nextToken", "next")))
    return {"players": players, "entries": len(lst), "understood": len(players), "more_pages": more}


def jersey_map(roster: dict) -> dict[str, dict]:
    """jersey -> player, the closed vocabulary the jersey classifier is constrained to."""
    return {p["jersey"]: p for p in roster["players"] if p["jersey"]}


__all__ = ["adapt_roster", "name_forms", "jersey_map"]
