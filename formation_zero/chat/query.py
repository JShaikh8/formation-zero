"""The constrained semantic layer over the `plays` table — two tools Claude can call:

  query_plays      filter plays, return rows or a count
  aggregate_plays  group plays and compute efficiency metrics (EPA/play, success rate, etc.)

Claude picks structured filters; SQL is built here from a fixed column whitelist with bound
parameters (no injection) and is guarded on column presence, so it degrades gracefully if a
season's PBP lacks a field. No LLM dependency — fully testable.
"""

from __future__ import annotations

import re

_TEAM_RE = re.compile(r"^[A-Za-z]{2,3}$")

# All WHERE-clause filter names (shared by both tools). Used to whitelist tool args.
FILTER_NAMES = {
    "offense", "defense", "off_grouping", "off_formation", "def_package", "def_front",
    "play_type", "down", "quarter", "distance_min", "distance_max", "red_zone",
    "pass_location", "pass_length", "run_location", "run_gap",
    "shotgun", "no_huddle", "goal_to_go", "first_down", "sack", "interception",
    "touchdown", "qb_scramble",
}
QUERY_ARGS = FILTER_NAMES | {"count_only", "limit"}
AGG_ARGS = FILTER_NAMES | {"group_by", "sort_by", "min_plays", "limit"}

# Dimensions a user can group by, and the metrics that can sort the result.
_AGG_DIMS = {
    "off_grouping", "off_formation", "def_package", "def_front", "play_type",
    "down", "quarter", "posteam", "defteam", "shotgun", "pass_location", "run_gap",
}
_DIM_TO_COL = {"quarter": "qtr"}  # dim name -> actual column
_AGG_METRICS = ["n", "epa_per_play", "success_rate", "yards_per_play", "explosive_rate"]

_RETURN_COLS = [
    "play_uid", "qtr", "down", "ydstogo", "yrdln", "posteam", "defteam", "play_type",
    "off_formation", "off_grouping", "offense_personnel", "def_package", "def_front",
    "yards_gained", "epa",
]


def _available_columns(con) -> set[str]:
    return {d[0] for d in con.execute("SELECT * FROM plays LIMIT 0").description}


def _build_where(con, filters: dict):
    """Build a parameterized WHERE clause from filters, skipping any whose column is absent."""
    cols = _available_columns(con)
    where: list[str] = []
    params: list = []

    def add(cond, *p):
        where.append(cond)
        params.extend(p)

    f = filters
    off, deff = f.get("offense"), f.get("defense")
    if off and _TEAM_RE.match(str(off)) and "posteam" in cols:
        add("posteam = ?", off.upper())
    if deff and _TEAM_RE.match(str(deff)) and "defteam" in cols:
        add("defteam = ?", deff.upper())

    for name, col, transform in [
        ("off_grouping", "off_grouping", None),
        ("off_formation", "off_formation", str.upper),
        ("def_package", "def_package", None),
        ("def_front", "def_front", None),
        ("play_type", "play_type", None),
        ("pass_location", "pass_location", str.lower),
        ("pass_length", "pass_length", str.lower),
        ("run_location", "run_location", str.lower),
        ("run_gap", "run_gap", str.lower),
    ]:
        v = f.get(name)
        if v is not None and col in cols:
            add(f"{col} = ?", transform(v) if transform else v)

    for name, col in [("down", "down"), ("quarter", "qtr")]:
        v = f.get(name)
        if v is not None and col in cols:
            add(f"{col} = ?", int(v))

    if f.get("distance_min") is not None and "ydstogo" in cols:
        add("ydstogo >= ?", int(f["distance_min"]))
    if f.get("distance_max") is not None and "ydstogo" in cols:
        add("ydstogo <= ?", int(f["distance_max"]))

    rz = f.get("red_zone")
    if rz is True and "yardline_100" in cols:
        add("yardline_100 <= 20")
    elif rz is False and "yardline_100" in cols:
        add("yardline_100 > 20")

    # nflfastR boolean indicators are stored as float 0/1.
    for name, col in [
        ("shotgun", "shotgun"), ("no_huddle", "no_huddle"), ("goal_to_go", "goal_to_go"),
        ("first_down", "first_down"), ("sack", "sack"), ("interception", "interception"),
        ("touchdown", "touchdown"), ("qb_scramble", "qb_scramble"),
    ]:
        v = f.get(name)
        if v is True and col in cols:
            add(f"{col} = 1")
        elif v is False and col in cols:
            add(f"{col} = 0")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return clause, params


def run_query_plays(con, *, count_only: bool = False, limit: int = 25, **filters) -> dict:
    """Filter the `plays` table. Returns {count} (count_only) or {count, returned, plays}."""
    clause, params = _build_where(con, filters)
    count = con.execute(f"SELECT COUNT(*) FROM plays{clause}", params).fetchone()[0]
    if count_only:
        return {"count": int(count)}
    lim = max(1, min(int(limit or 25), 100))
    cols = _available_columns(con)
    sel = [c for c in _RETURN_COLS if c in cols]
    df = con.execute(
        f"SELECT {', '.join(sel)} FROM plays{clause} ORDER BY play_index LIMIT {lim}", params
    ).fetch_df()
    return {"count": int(count), "returned": len(df), "plays": df.to_dict(orient="records")}


def run_aggregate_plays(con, *, group_by=None, sort_by: str = "epa_per_play",
                        min_plays: int = 1, limit: int = 20, **filters) -> dict:
    """Group plays and compute efficiency metrics per group, sorted by `sort_by` descending."""
    cols = _available_columns(con)
    dims = [d for d in (group_by or []) if d in _AGG_DIMS and _DIM_TO_COL.get(d, d) in cols]
    if not dims:
        return {"error": "group_by must include at least one of: " + ", ".join(sorted(_AGG_DIMS))}
    if sort_by not in _AGG_METRICS:
        sort_by = "epa_per_play"

    clause, params = _build_where(con, filters)
    group_cols = [_DIM_TO_COL.get(d, d) for d in dims]
    metrics = ["COUNT(*) AS n"]
    # EPA is a modelled column, absent from raw feeds until our own EP model produces it.
    if "epa" in cols:
        metrics += [
            "ROUND(AVG(epa), 3) AS epa_per_play",
            "ROUND(AVG(CASE WHEN epa > 0 THEN 1.0 ELSE 0 END), 3) AS success_rate",
        ]
    if "yards_gained" in cols:
        metrics += [
            "ROUND(AVG(yards_gained), 2) AS yards_per_play",
            "ROUND(AVG(CASE WHEN yards_gained >= 15 THEN 1.0 ELSE 0 END), 3) AS explosive_rate",
        ]
    available = {m.split(" AS ")[-1] for m in metrics}
    if sort_by not in available:
        # Fall back to whatever metric this table can actually compute.
        sort_by = "yards_per_play" if "yards_per_play" in available else "n"

    sql = (
        f"SELECT {', '.join(group_cols + metrics)} FROM plays{clause} "
        f"GROUP BY {', '.join(group_cols)} HAVING COUNT(*) >= ? "
        f"ORDER BY {sort_by} DESC LIMIT ?"
    )
    lim = max(1, min(int(limit or 20), 50))
    df = con.execute(sql, params + [int(min_plays), lim]).fetch_df()
    return {"group_by": dims, "sort_by": sort_by, "groups": df.to_dict(orient="records")}


# ---- Tool definitions for the Messages API -------------------------------------------------

_FILTER_PROPS = {
    "offense": {"type": "string", "description": "Offensive team abbrev, e.g. KC, DET"},
    "defense": {"type": "string", "description": "Defensive team abbrev"},
    "off_grouping": {"type": "string", "description": "Personnel: '11','12','21' (1st digit=#backs, 2nd=#TEs)"},
    "off_formation": {"type": "string", "description": "UPPERCASE: SHOTGUN, UNDER CENTER, PISTOL, EMPTY, I_FORM, JUMBO, WILDCAT"},
    "def_package": {"type": "string", "enum": ["base", "nickel", "dime", "quarter"]},
    "def_front": {"type": "string", "description": "e.g. '4-2-5' (DL-LB-DB counts)"},
    "play_type": {"type": "string", "enum": ["run", "pass"]},
    "down": {"type": "integer", "description": "1-4"},
    "quarter": {"type": "integer", "description": "1-4 (5=OT)"},
    "distance_min": {"type": "integer", "description": "min yards to go"},
    "distance_max": {"type": "integer", "description": "max yards to go"},
    "red_zone": {"type": "boolean", "description": "snap inside opponent's 20"},
    "goal_to_go": {"type": "boolean"},
    "shotgun": {"type": "boolean"},
    "no_huddle": {"type": "boolean"},
    "first_down": {"type": "boolean", "description": "play resulted in a first down"},
    "sack": {"type": "boolean"},
    "interception": {"type": "boolean"},
    "touchdown": {"type": "boolean"},
    "qb_scramble": {"type": "boolean"},
    "pass_location": {"type": "string", "enum": ["left", "middle", "right"]},
    "pass_length": {"type": "string", "enum": ["short", "deep"]},
    "run_location": {"type": "string", "enum": ["left", "middle", "right"]},
    "run_gap": {"type": "string", "enum": ["guard", "tackle", "end"]},
}

QUERY_PLAYS_TOOL = {
    "name": "query_plays",
    "description": (
        "Filter the play database and return matching plays (or a count). All filters are optional "
        "and AND-combined. Use count_only=true for 'how many' questions."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **_FILTER_PROPS,
            "count_only": {"type": "boolean", "description": "return only the count"},
            "limit": {"type": "integer", "description": "max rows (default 25, cap 100)"},
        },
    },
}

AGGREGATE_PLAYS_TOOL = {
    "name": "aggregate_plays",
    "description": (
        "Group plays and compute efficiency metrics per group: n (plays), epa_per_play, "
        "success_rate (share with EPA>0), yards_per_play, explosive_rate (share gaining 15+ yds). "
        "Use for 'which X was most efficient / most productive' questions. Same optional filters as "
        "query_plays. Returns groups sorted by sort_by descending."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **_FILTER_PROPS,
            "group_by": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_AGG_DIMS)},
                "description": "dimensions to group by (1+)",
            },
            "sort_by": {"type": "string", "enum": _AGG_METRICS, "description": "metric to sort by (default epa_per_play)"},
            "min_plays": {"type": "integer", "description": "drop groups with fewer than N plays (default 1)"},
            "limit": {"type": "integer", "description": "max groups (default 20, cap 50)"},
        },
        "required": ["group_by"],
    },
}
