"""Parse PBP personnel/formation strings into structured football labels.

This is the first slice of the intelligence layer and needs **no CV** — it runs on the
play-by-play we already pull. It also defines the *target labels* the perception layer must
later reproduce from video (same grouping names, derived from tracked positions instead).

Examples
--------
    >>> parse_personnel("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR")["RB"]
    1
    >>> offensive_grouping(parse_personnel("1 RB, 2 TE, 2 WR"))
    '12'
    >>> defensive_front(parse_personnel("2 CB, 2 DE, 2 DT, 1 FS, 1 ILB, 1 OLB, 2 SS"))
    {'dl': 4, 'lb': 2, 'db': 5, 'front': '4-2-5', 'package': 'nickel'}
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"(\d+)\s+([A-Za-z]+)")

# Position families. Backs count toward the personnel grouping's tens digit.
_BACK_POS = {"RB", "FB", "HB"}
_TE_POS = {"TE"}
_DL_POS = {"DE", "DT", "NT", "DL", "EDGE"}
_LB_POS = {"ILB", "OLB", "MLB", "LB", "WLB", "SLB"}
_DB_POS = {"CB", "FS", "SS", "S", "DB", "NB"}


def parse_personnel(text: str | None) -> dict[str, int]:
    """Parse 'N POS' tokens into {POS: count}. Tolerant of OL-inclusive or skill-only strings."""
    out: dict[str, int] = {}
    if not text:
        return out
    for count, pos in _TOKEN_RE.findall(text):
        out[pos.upper()] = out.get(pos.upper(), 0) + int(count)
    return out


def offensive_grouping(personnel: dict[str, int]) -> str | None:
    """Canonical offensive personnel, e.g. 1 back + 1 TE -> '11', 2 backs + 1 TE -> '21'.

    Returns None when personnel is unknown. Feeds without a personnel field (e.g. the STATS
    play-by-play) must yield *missing* labels, not '00' — a fabricated grouping is
    indistinguishable from a real one downstream.
    """
    if not personnel:
        return None
    backs = sum(personnel.get(p, 0) for p in _BACK_POS)
    tes = sum(personnel.get(p, 0) for p in _TE_POS)
    return f"{backs}{tes}"


def defensive_front(personnel: dict[str, int]) -> dict:
    """Defensive line/LB/DB counts + a front string and coverage-shell package label.

    Counts of zero mean "unknown", not "none on the field", so an empty personnel dict yields
    None throughout rather than a spurious 0-0-0 base front.
    """
    if not personnel:
        return {"dl": None, "lb": None, "db": None, "front": None, "package": None}
    dl = sum(personnel.get(p, 0) for p in _DL_POS)
    lb = sum(personnel.get(p, 0) for p in _LB_POS)
    db = sum(personnel.get(p, 0) for p in _DB_POS)
    if db <= 4:
        package = "base"
    elif db == 5:
        package = "nickel"
    elif db == 6:
        package = "dime"
    else:
        package = "quarter"
    return {"dl": dl, "lb": lb, "db": db, "front": f"{dl}-{lb}-{db}", "package": package}


def summarize_play(offense_personnel: str | None,
                   defense_personnel: str | None,
                   offense_formation: str | None = None) -> dict:
    """One-call play summary: offensive grouping + formation + defensive front/package."""
    off = parse_personnel(offense_personnel)
    deff = defensive_front(parse_personnel(defense_personnel))
    return {
        "off_grouping": offensive_grouping(off),
        "off_formation": (offense_formation or "").upper() or None,
        "wr": off.get("WR", 0) if off else None,
        "te": sum(off.get(p, 0) for p in _TE_POS) if off else None,
        "rb": sum(off.get(p, 0) for p in _BACK_POS) if off else None,
        "def_front": deff["front"],
        "def_package": deff["package"],
        "def_dl": deff["dl"],
        "def_lb": deff["lb"],
        "def_db": deff["db"],
    }


__all__ = ["parse_personnel", "offensive_grouping", "defensive_front", "summarize_play"]
