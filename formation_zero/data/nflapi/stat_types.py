"""GSIS statType codes, copied from the reference port's audited table. Never restate a code inline.

Source: the official GSIS Stat_ID document cross-checked against 9,224 records from fourteen
complete 2025 games (the reference project's audit). Codes 52/53 have their NAMES right and the
document's comments swapped; the games settled it. Do not "fix" that back.
"""

from __future__ import annotations

import re

LABELS: dict[int, str | None] = {
    1: None, 2: "Punt blocked (offense)", 3: "First down, rushing (team)", 4: "First down, passing (team)",
    5: "First down, penalty (team)", 6: "Third down converted (team)", 7: "Third down failed (team)",
    8: "Fourth down converted (team)", 9: "Fourth down failed (team)", 10: "Rushing yards", 11: "Rushing yards (TD)",
    12: "Rushing yards, no rushing attempt (lateral)", 13: "Rushing yards, no attempt (TD, lateral)",
    14: "Incomplete pass", 15: "Passing yards", 16: "Passing yards (TD)", 17: None, 18: None,
    19: "Passer on an intercepted pass", 20: "Sack yards lost (charged to the passer)", 21: "Receiving yards",
    22: "Receiving yards (TD)", 23: "Receiving yards, no reception (lateral)", 24: "Receiving yards, no reception (TD, lateral)",
    25: "Interception, with return yards", 26: "Interception returned for a touchdown",
    27: "Interception return yards, no interception (lateral)", 28: "Interception return yards, no interception (TD, lateral)",
    29: "Punt yards", 30: "Punt inside the 20", 31: "Punt into the end zone and run out", 32: "Punt with touchback (punter)",
    33: "Punt return yards", 34: "Punt return for a touchdown", 35: "Punt return yards, no return (lateral)",
    36: "Punt return yards, no return (TD, lateral)", 37: "Punt out of bounds", 38: "Punt downed, no return",
    39: "Punt fair catch", 40: "Punt touchback, no return (team)", 41: "Kickoff yards", 42: "Kickoff inside the 20",
    43: "Kickoff into the end zone and run out", 44: "Kickoff with touchback", 45: "Kick return yards",
    46: "Kick return for a touchdown", 47: "Kick return yards, no return (lateral)", 48: "Kick return yards, no return (TD, lateral)",
    49: "Kickoff out of bounds", 50: "Kickoff fair catch", 51: "Kickoff touchback (team)", 52: "Fumble (a defender forced it)",
    53: "Fumble (nobody forced it)", 54: "Fumble out of bounds", 55: "Fumble recovered by own team, with yards",
    56: "Fumble recovered by own team (TD)", 57: "Own-recovery yards, no recovery (lateral)", 58: "Own-recovery yards, no recovery (TD, lateral)",
    59: "Fumble recovered by the opposition, with yards", 60: "Opponent recovery for a touchdown",
    61: "Opponent-recovery yards, no recovery (lateral)", 62: "Opponent-recovery yards, no recovery (TD, lateral)",
    63: "Miscellaneous yards", 64: "Miscellaneous yards (TD)", 68: "Timeout (team)", 69: "Field goal missed, attempt length",
    70: "Field goal made, attempt length", 71: "Field goal blocked (kicker)", 72: "Extra point good", 73: "Extra point failed",
    74: "Extra point blocked", 75: "Two-point rush, good", 76: "Two-point rush, failed", 77: "Two-point pass, good",
    78: "Two-point pass, failed", 79: "Solo tackle", 80: "Assisted tackle (tackle made with help)", 81: None,
    82: "Tackle assist", 83: "Sack (unassisted)", 84: "Half sack (split between two players)", 85: "Pass defensed",
    86: "Punt blocked (defender)", 87: "Extra point blocked (defender)", 88: "Field goal blocked (defender)",
    89: "Safety (defender)", 90: None, 91: "Forced fumble (defender)", 93: "Penalty", 95: "Team tackled for a loss, with yardage",
    96: "Extra point safety", 99: "Two-point rush safety", 100: "Two-point pass safety", 102: "Kickoff downed by the kicking team",
    103: "Sack yards, no sack (lateral)", 104: "Two-point reception, good", 105: "Two-point reception, failed",
    106: "Fumble lost", 107: "Own kickoff recovery", 108: "Own kickoff recovery (TD)", 110: "Quarterback hit",
    111: "Pass length on a completion (air yards)", 112: "Pass length on a non-completion (air yards)",
    113: "Yards gained after the catch", 115: "Pass target", 120: "Tackle for a loss (defender)",
    301: "Extra point aborted", 402: "Tackle-for-loss yardage", 410: "Kickoff length, including end-zone yards",
    421: "Kickoff short of the landing zone", 422: "Kickoff via the landing zone into the end zone (touchback to the 20)",
    424: "Kickoff via the landing zone into the end zone (team)", 426: "Kickoff straight into the end zone (touchback to the 35)",
    427: "Kickoff straight into the end zone (team)",
}

RUSH_YARDS = {10, 11, 12, 13}
PASS_YARDS = {15, 16}
INCOMPLETE_PASS = 14
PASSER_INTERCEPTED = 19
SACK_YARDS_LOST = 20
RECEIVING_YARDS = {21, 22, 23, 24}
INTERCEPTION = {25, 26}
TACKLE = {79, 80, 82}
SACK = {83, 84}
PASS_DEFENSED = 85
FORCED_FUMBLE = 91
PENALTY = 93
QB_HIT = 110
AIR_YARDS = {111, 112}
YAC = 113
PASS_TARGET = 115
TFL = 120
FUMBLE = {52, 53, 106}

# statType -> the role a player played on the play, for the identity layer.
ROLE_BY_CODE: dict[int, str] = {
    **{c: "passer" for c in (14, 15, 16, 19, 20)},
    **{c: "rusher" for c in RUSH_YARDS},
    **{c: "receiver" for c in RECEIVING_YARDS}, PASS_TARGET: "target",
    **{c: "tackler" for c in TACKLE}, **{c: "sacker" for c in SACK},
    PASS_DEFENSED: "pass_defender", **{c: "interceptor" for c in INTERCEPTION},
    FORCED_FUMBLE: "forced_fumble", **{c: "fumbler" for c in FUMBLE},
    QB_HIT: "qb_hit", TFL: "tackle_for_loss", 41: "kicker", 29: "punter", 45: "returner", 33: "returner",
    70: "kicker", 69: "kicker", 72: "kicker", 73: "kicker",
}

_ADMIN_MARKER = re.compile(r"\btime\s?out\b|two[-\s]?minute warning|end of (quarter|half|game|regulation|period)|end quarter", re.I)
_REAL_ACTION = re.compile(r"\b(pass|rush|scramble|sack|punt|kick|field goal|kneel|spike|handoff|intercept|fumble|lateral)\b", re.I)


def label(code: int) -> str:
    v = LABELS.get(code)
    if v:
        return v
    return f"statType {code} (meaning not established)" if code in LABELS else f"statType {code}"


def is_admin_play(text: str | None, play_type: str | None = None) -> bool:
    """A timeout, two-minute warning or end-of-period row: a marker present and no real action."""
    s = f"{text or ''} {play_type or ''}"
    return bool(_ADMIN_MARKER.search(s)) and not _REAL_ACTION.search(s)


__all__ = ["LABELS", "RUSH_YARDS", "PASS_YARDS", "INCOMPLETE_PASS", "SACK_YARDS_LOST", "RECEIVING_YARDS",
           "TACKLE", "SACK", "PASS_DEFENSED", "PENALTY", "PASS_TARGET", "ROLE_BY_CODE", "label", "is_admin_play"]
