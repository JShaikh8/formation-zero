"""Normalize a STATS event payload into the tidy per-play schema used by `gridiron.data.pbp`.

What STATS gives that nflverse does not
---------------------------------------
Per-play `playersInvolved`, each carrying a **uniform number** and a role (passer, target,
tackler, pass defender, snapper, holder). That is direct supervision for the perception layer:
for every snap we know which jersey numbers were on the field and what each did — training and
validation signal for jersey OCR and identity assignment, which is otherwise the expensive part.

What STATS does *not* give
--------------------------
No personnel grouping, no offensive formation, no EPA. Those still come from nflverse, so the
two feeds are complementary rather than interchangeable — see `merge_nflverse_labels`.

Yard lines
----------
STATS writes the ball spot team-relative ("Chi37" = the Bears' 37). Converting to nflverse's
`yardline_100` (yards to the *defending* goal line) needs the possessing team, and that is what
`gridiron.configs.nfl_field.line_of_scrimmage_x` consumes to anchor field registration.
"""

from __future__ import annotations

import re

from gridiron.data.teams import TeamRegistry

_YARDLINE_RE = re.compile(r"^([A-Za-z]+)\s*(\d{1,2})$")

# STATS playTypeId -> nflverse-style play_type. Sacks and interceptions are dropbacks, so they
# normalize to "pass" the way nflverse does.
PLAY_TYPE_MAP: dict[int, str] = {
    1: "pass",          # Pass (complete)
    2: "pass",          # Incomplete Pass
    3: "pass",          # Sack
    4: "run",           # Rush
    5: "kickoff",
    7: "punt",
    9: "pass",          # Interception
    10: "penalty",
    11: "penalty",
    13: "two_minute_warning",
    14: "fumble",       # resolved below — usually a run/pass that ended in a fumble
    22: "extra_point",
    42: "field_goal",
    57: "timeout",
    58: "timeout",
}

# Play types that correspond to an actual snap, i.e. what coaches film shows as a clip.
SCRIMMAGE_TYPES = {"pass", "run"}

# A penalty row whose text *starts* with "Penalty:" is a dead-ball / pre-snap foul (false start,
# encroachment) — no snap happened, so there is no film clip. When the text instead narrates a
# play before the foul ("Stafford pass to ... Penalty: Unnecessary Roughness"), the snap did
# happen and must be counted, even though the play was negated.
_DEAD_BALL_PENALTY_RE = re.compile(r"^\s*penalty\s*:", re.I)

# Kicking plays are snaps but not scrimmage plays; nflverse types them separately, and offensive
# coaches film generally excludes them.
_PUNT_RE = re.compile(r"\bpunts?\b", re.I)
_KICKOFF_RE = re.compile(r"\bkicks off\b|\bkickoff\b", re.I)
_FIELD_GOAL_RE = re.compile(r"\bfield goal\b", re.I)
_EXTRA_POINT_RE = re.compile(r"\bextra point\b", re.I)
_PASS_RE = re.compile(r"\b(pass|sacked|scrambles|spike[sd]?)\b", re.I)
_RUN_RE = re.compile(r"\b(rush(?:es|ed)?|run[s]?|kneel[s]?|up the middle)\b", re.I)


def infer_play_type(text: str | None) -> str | None:
    """Infer an nflverse-style play_type from playText.

    Needed for rows whose `playTypeId` describes an *outcome* (Fumble, Penalty) rather than the
    action, so the underlying snap type has to come from the narration.
    """
    if not text:
        return None
    for pattern, play_type in (
        (_KICKOFF_RE, "kickoff"), (_PUNT_RE, "punt"),
        (_FIELD_GOAL_RE, "field_goal"), (_EXTRA_POINT_RE, "extra_point"),
        (_PASS_RE, "pass"), (_RUN_RE, "run"),
    ):
        if pattern.search(text):
            return play_type
    return None


def classify_play(play: dict) -> tuple[str | None, bool]:
    """Return (play_type, is_scrimmage) for one STATS play record.

    Resolves the two cases where `playTypeId` alone is wrong: Fumble rows (the id names the
    outcome, so a punt-return fumble looks like a scrimmage play) and Penalty rows (dead-ball
    fouls vs. fouls on a snap that actually occurred).
    """
    type_id = (play.get("playType") or {}).get("playTypeId")
    text = play.get("playText") or ""
    play_type = PLAY_TYPE_MAP.get(type_id)

    if play_type == "fumble":
        # The id names the outcome; recover the action from the narration.
        play_type = infer_play_type(text) or "fumble"
    elif play_type == "penalty":
        if _DEAD_BALL_PENALTY_RE.match(text):
            return "penalty", False  # pre-snap foul — nothing was snapped, nothing on film
        play_type = infer_play_type(text) or "penalty"

    return play_type, play_type in SCRIMMAGE_TYPES

# playersInvolved roles, by playerInvolvedTypeId.
INVOLVEMENT_ROLES = {
    1: "player", 2: "tackler", 3: "assister", 4: "pass_defender",
    5: "target", 6: "snapper", 7: "holder",
}


def parse_yardline(yardline: str | None, posteam_abbr: str | None) -> int | None:
    """Team-relative yard line -> nflverse `yardline_100` (yards to the defending goal).

    ``parse_yardline("Chi37", "LA")``  -> 37   (Rams have it on the Bears' 37)
    ``parse_yardline("LAR35", "LA")``  -> 65   (Rams have it on their own 35)

    Returns None when the spot or possession is unknown rather than guessing a side.
    """
    if not yardline or not posteam_abbr:
        return None
    match = _YARDLINE_RE.match(yardline.strip())
    if not match:
        return None
    side, yard = match.group(1), int(match.group(2))
    if not 0 <= yard <= 50:
        return None
    from gridiron.data.teams import to_nflverse

    own_half = to_nflverse(side) == posteam_abbr.upper()
    return 100 - yard if own_half else yard


def _players(play: dict) -> list[dict]:
    """Flatten `playersInvolved` into role/name/uniform records."""
    out = []
    for entry in play.get("playersInvolved") or []:
        player = entry.get("player") or {}
        out.append({
            "role": INVOLVEMENT_ROLES.get(entry.get("playerInvolvedTypeId"),
                                          entry.get("playerInvolvedType")),
            "player_id": player.get("playerId"),
            "name": " ".join(x for x in (player.get("firstName"), player.get("lastName")) if x),
            "uniform": player.get("uniform"),
        })
    return out


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def event_node(payload: dict) -> dict:
    """The single event record inside an event-detail payload."""
    from gridiron.data.statsapi import iter_events

    for _type_id, _name, event in iter_events(payload):
        return event
    raise SystemExit("No event found in payload — was it fetched with pbp=true?")


def normalize_event(payload: dict, registry: TeamRegistry | None = None):
    """STATS event payload -> tidy per-play DataFrame matching gridiron.data.pbp's schema.

    Adds `play_index` / `play_uid` over scrimmage plays only — the alignment key against
    segmented film clips (Nth clip -> Nth scrimmage play).
    """
    import pandas as pd

    from gridiron.ids import game_key, play_uid

    registry = registry or TeamRegistry.fallback()
    event = event_node(payload)

    from gridiron.data.statsapi import home_away, start_date

    home, away = home_away(event)
    home_abbr = registry.abbreviation(home.get("teamId")) if home else None
    away_abbr = registry.abbreviation(away.get("teamId")) if away else None

    season = _season_of(payload, event)
    week = event.get("week")
    gk = game_key(season, week, away_abbr or "UNK", home_abbr or "UNK")

    rows = []
    for play in event.get("pbp") or []:
        play_type, is_scrimmage = classify_play(play)
        posteam_id = (play.get("startPossession") or {}).get("teamId")
        posteam = registry.abbreviation(posteam_id)
        defteam = away_abbr if posteam == home_abbr else home_abbr

        down = _int_or_none(play.get("down"))
        rows.append({
            "game_key": gk,
            "stats_event_id": event.get("eventId"),
            "play_id": play.get("playId"),
            "drive_id": play.get("driveId"),
            "qtr": play.get("period"),
            "time": play.get("time"),
            "down": down,
            "ydstogo": _int_or_none(play.get("distance")),
            "yrdln": play.get("yardLine"),
            "yardline_100": parse_yardline(play.get("yardLine"), posteam),
            "posteam": posteam,
            "defteam": defteam,
            "posteam_id": posteam_id,
            "play_type": play_type,
            "stats_play_type": (play.get("playType") or {}).get("name"),
            "yards_gained": play.get("yards"),
            "direction": play.get("direction"),
            "desc": play.get("playText"),
            "is_review": play.get("isReview"),
            "is_continuation": play.get("isContinuation"),
            "players": _players(play),
            "is_scrimmage": is_scrimmage,
        })

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return tidy

    index = 0
    indices: list[int | None] = []
    uids: list[str | None] = []
    for is_scrimmage in tidy["is_scrimmage"]:
        if is_scrimmage:
            index += 1
            indices.append(index)
            uids.append(play_uid(gk, index))
        else:
            indices.append(None)
            uids.append(None)
    tidy.insert(1, "play_uid", uids)
    tidy.insert(2, "play_index", indices)
    tidy.attrs["start_date"] = start_date(event)
    return tidy


def _season_of(payload: dict, event: dict) -> int:
    """Season for an event, from the envelope, falling back to the start date's year."""
    from gridiron.data.statsapi import season_node, start_date

    season = season_node(payload).get("season")
    if season:
        return int(season)
    date = start_date(event)
    year = int(date[:4]) if date[:4].isdigit() else 0
    # Playoffs occur in January of the following calendar year.
    month = int(date[5:7]) if date[5:7].isdigit() else 0
    return year - 1 if month and month <= 2 else year


def jersey_index(tidy) -> dict[str, set[str]]:
    """Map team abbrev -> set of uniform numbers seen, for the perception layer.

    Ground truth for jersey OCR: these are the only numbers that can legitimately appear, which
    turns open-set number recognition into a bounded classification problem.
    """
    index: dict[str, set[str]] = {}
    for _, row in tidy.iterrows():
        for player in row["players"] or []:
            uniform = player.get("uniform")
            if not uniform:
                continue
            # Possession identifies the offense; defenders belong to the other side.
            team = row["posteam"] if player["role"] in ("player", "target", "snapper", "holder") \
                else row["defteam"]
            if team:
                index.setdefault(team, set()).add(str(uniform))
    return index


__all__ = [
    "normalize_event", "parse_yardline", "jersey_index", "event_node",
    "PLAY_TYPE_MAP", "SCRIMMAGE_TYPES", "INVOLVEMENT_ROLES",
]
