"""Offensive formation taxonomy: tracked player positions -> named slots -> formation id.

The taxonomy is a set of **slots** (WRL, WSL, TEL, FBR, ...) and each formation is the set of
six skill slots it fills. That structure is what makes this tractable: rather than training a
96-way classifier, we predict which slots are occupied and look the formation up. Slot
occupancy is far more data-efficient to learn, degrades gracefully (trips-left can be right
while the backfield is wrong), and the label falls out deterministically.

Coordinate frame
----------------
Alignment is expressed relative to the ball at the snap, not in absolute field coordinates:

    lateral : yards left(-) / right(+) of the ball, from the offense's point of view
    depth   : yards behind the line of scrimmage (0 = on the line, positive = backfield)

`to_snap_frame` converts absolute (x, y) in `formation_zero.configs.nfl_field` coordinates into that
frame given the LOS and the direction of attack.

Region thresholds come from real football geometry, not from the display coordinates the
taxonomy was drawn with — those are stretched for legibility (they place the left tackle ~6.4
yd from center; the real figure is ~2.7) and would make poor decision boundaries.

Known limitations
-----------------
* Splits are measured from the ball, so a boundary-side receiver pinned by the sideline may
  read one slot tighter than a charter would call it. Hash position is not yet accounted for.
* 13 formation pairs in the source taxonomy are alignment-identical; see ALIASES.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- region thresholds (yards) -------------------------------------------------------------

ON_LINE_MAX_DEPTH = 1.5      # within this of the LOS counts as "on the line"
BACKFIELD_MIN_DEPTH = 3.0    # beyond this is a back, not a wing
SHOTGUN_MIN_DEPTH = 4.0      # QB depth separating under-center from shotgun
TACKLE_BOX_LATERAL = 4.0     # inside this is offensive line
INLINE_MAX_LATERAL = 7.0     # attached tight end zone, just outside the tackle
SLOT_MAX_LATERAL = 14.0      # beyond this is a wide receiver split

# Backfield shape thresholds.
OFFSET_MIN_LATERAL = 1.5     # a back this far off the ball is offset, not stacked
SPLIT_BACK_MAX_DEPTH_GAP = 1.5   # two backs within this depth gap are side-by-side


@dataclass(frozen=True)
class Alignment:
    """One player's position at the snap, in the snap frame."""

    lateral: float
    depth: float
    position: str | None = None   # QB/RB/FB/TE/WR when known; inferred otherwise
    jersey: str | None = None

    @property
    def side(self) -> str:
        return "L" if self.lateral < 0 else "R"

    @property
    def on_line(self) -> bool:
        return self.depth <= ON_LINE_MAX_DEPTH


def to_snap_frame(x: float, y: float, los_x: float, ball_y: float,
                  attacking: str = "right") -> tuple[float, float]:
    """Absolute nfl_field (x, y) in yards -> (lateral, depth) relative to the ball.

    `attacking` is the direction the offense moves along the field's x axis. Flipping it mirrors
    lateral so that "left" and "right" are always the offense's own.
    """
    if attacking not in ("right", "left"):
        raise ValueError(f"attacking must be 'right' or 'left', got {attacking!r}")
    if attacking == "right":
        return (y - ball_y), (los_x - x)
    return (ball_y - y), (x - los_x)


# ---- slot assignment -----------------------------------------------------------------------

# Receivers are numbered outside-in (WRL is the widest); tight ends inside-out (TEL is the one
# attached closest to the tackle). Both orderings come from the source taxonomy's own geometry.
_WR_SLOTS = ["WR{s}", "WS{s}", "WS2{s}", "WS3{s}", "WS4{s}"]
_TE_SLOTS = ["TE{s}", "TE2{s}", "TE3{s}"]


def _is_lineman(a: Alignment) -> bool:
    if a.position:
        return a.position.upper() in {"C", "G", "T", "LG", "RG", "LT", "RT", "OL"}
    return a.on_line and abs(a.lateral) <= TACKLE_BOX_LATERAL


def _is_back(a: Alignment) -> bool:
    if a.position:
        return a.position.upper() in {"RB", "FB", "HB"}
    return a.depth >= BACKFIELD_MIN_DEPTH


def _is_quarterback(a: Alignment) -> bool:
    return bool(a.position) and a.position.upper() == "QB"


def _assign_backfield(backs: list[Alignment]) -> dict[str, Alignment]:
    """Name the backfield slots: RB / FB / FBL / FBR / RBL / RBR."""
    if not backs:
        return {}
    backs = sorted(backs, key=lambda a: a.depth)

    if len(backs) == 1:
        return {"RB": backs[0]}

    if len(backs) == 2:
        shallow, deep = backs
        # Side-by-side at similar depth is a split backfield.
        if deep.depth - shallow.depth <= SPLIT_BACK_MAX_DEPTH_GAP:
            left, right = sorted(backs, key=lambda a: a.lateral)
            return {"RBL": left, "RBR": right}
        # Stacked: the shallow back is the fullback, offset if displaced from the ball.
        if abs(shallow.lateral) < OFFSET_MIN_LATERAL:
            return {"FB": shallow, "RB": deep}
        return {"FBL" if shallow.lateral < 0 else "FBR": shallow, "RB": deep}

    # Three backs: a fullback plus a split pair behind him.
    fullback, *rest = backs
    left, right = sorted(rest, key=lambda a: a.lateral)[:2]
    return {"FB": fullback, "RBL": left, "RBR": right}


def _assign_receivers(receivers: list[Alignment]) -> dict[str, Alignment]:
    """Name the receiver and tight end slots, per side."""
    slots: dict[str, Alignment] = {}
    for side in ("L", "R"):
        group = [a for a in receivers if a.side == side]
        if not group:
            continue
        # Attached = on the line, inside the tight end zone. Everyone else is a receiver split.
        attached = [a for a in group if a.on_line and abs(a.lateral) <= INLINE_MAX_LATERAL]
        detached = [a for a in group if a not in attached]

        for i, a in enumerate(sorted(attached, key=lambda a: abs(a.lateral))):
            if i < len(_TE_SLOTS):
                slots[_TE_SLOTS[i].format(s=side)] = a
        for i, a in enumerate(sorted(detached, key=lambda a: abs(a.lateral), reverse=True)):
            if i < len(_WR_SLOTS):
                slots[_WR_SLOTS[i].format(s=side)] = a
    return slots


def assign_slots(players: list[Alignment]) -> dict[str, Alignment]:
    """Map an offense's snap-frame alignments onto named taxonomy slots.

    Linemen are dropped — formations are defined by the six skill slots, with the five-man line
    assumed. Returns {slot_name: alignment}.
    """
    quarterbacks = [a for a in players if _is_quarterback(a)]
    rest = [a for a in players if a not in quarterbacks]

    if not quarterbacks:
        # No position labels: the shallowest player near the ball behind the line is the QB.
        candidates = [a for a in rest if abs(a.lateral) < OFFSET_MIN_LATERAL and not a.on_line]
        if candidates:
            qb = min(candidates, key=lambda a: a.depth)
            quarterbacks, rest = [qb], [a for a in rest if a is not qb]

    slots: dict[str, Alignment] = {}
    if quarterbacks:
        qb = quarterbacks[0]
        slots["QBS" if qb.depth >= SHOTGUN_MIN_DEPTH else "QB"] = qb

    skill = [a for a in rest if not _is_lineman(a)]
    backs = [a for a in skill if _is_back(a)]
    receivers = [a for a in skill if a not in backs]

    slots.update(_assign_backfield(backs))
    slots.update(_assign_receivers(receivers))
    return slots


# ---- the taxonomy --------------------------------------------------------------------------
#
# Each entry: id -> (name, six skill slots). Ids 23 and 24 do not exist in the source taxonomy.

_DEFINITIONS: dict[int, tuple[str, str]] = {
    1: ("2 RB side-by-side, 1 TE, 2 WR", "QB RBL RBR TER WRL WRR"),
    2: ("1 RB, 2 TE, 2 WR", "QB RB TEL TER WRL WRR"),
    3: ("2 RB I-Formation", "QB FB RB TER WRL WRR"),
    4: ("2 RB Offset I", "QB FBR RB TER WRL WRR"),
    5: ("Both WR left (Normal)", "QB RBL RBR TER WRL WSL"),
    6: ("Both WR right (Normal)", "QB RBL RBR TEL WRR WSR"),
    7: ("Both WR left (I-Formation)", "QB FB RB TER WRL WSL"),
    8: ("Both WR right (I-Formation)", "QB FB RB TEL WRR WSR"),
    9: ("Both WR left (Offset I)", "QB FBR RB TER WRL WSL"),
    10: ("Both WR right (Offset I)", "QB FBR RB TEL WRR WSR"),
    11: ("3 WR (2 left), 1 RB", "QBS RB TER WRL WSL WRR"),
    12: ("3 WR (2 right), 1 RB", "QBS RB TEL WRL WRR WSR"),
    13: ("3 WR (all left), 1 RB", "QBS RB TER WRL WSL WS2L"),
    14: ("3 WR (all right), 1 RB", "QBS RB TEL WRR WSR WS2R"),
    15: ("2 TE, 3 WR (left)", "QBS TEL TER WRL WSL WS2L"),
    16: ("2 TE, 3 WR (right)", "QBS TEL TER WRR WSR WS2R"),
    17: ("3 WR (2 left), 2 RB", "QB RBL RBR WRL WSL WRR"),
    18: ("3 WR (2 right), 2 RB", "QB RBL RBR WRL WRR WSR"),
    19: ("3 WR (2 left), 2 RB (I)", "QB FB RB WRL WSL WRR"),
    20: ("3 WR (2 right), 2 RB (I)", "QB FB RB WRL WRR WSR"),
    21: ("3 WR (2 left), 2 RB (Offset I)", "QB FBR RB WRL WSL WRR"),
    22: ("3 WR (2 right), 2 RB (Offset I)", "QB FBL RB WRL WRR WSR"),
    25: ("3 WR (all left), 2 RB", "QB RBL RBR WRL WSL WS2L"),
    26: ("3 WR (all right), 2 RB", "QB RBL RBR WRR WSR WS2R"),
    27: ("3 WR (all left), 2 RB (I)", "QB FB RB WRL WSL WS2L"),
    28: ("3 WR (all right), 2 RB (I)", "QB FB RB WRR WSR WS2R"),
    29: ("3 WR (all left), 2 RB (Offset I)", "QB FBR RB WRL WSL WS2L"),
    30: ("3 WR (all right), 2 RB (Offset I)", "QB FBL RB WRR WSR WS2R"),
    31: ("2 TE, 1 WR (left), 2 RB", "QB RBL RBR TEL TER WRL"),
    32: ("2 TE, 1 WR (right), 2 RB", "QB RBL RBR TEL TER WRR"),
    33: ("2 TE, 1 WR (left), 2 RB (I)", "QB FB RB TEL TER WRL"),
    34: ("2 TE, 1 WR (right), 2 RB (I)", "QB FB RB TEL TER WRR"),
    35: ("2 TE, 1 WR (left), 2 RB (Offset I)", "QB FBR RB TEL TER WRL"),
    36: ("2 TE, 1 WR (right), 2 RB (Offset I)", "QB FBL RB TEL TER WRR"),
    37: ("3 TE, 1 WR (left), 1 RB", "QB RB TEL TER TE2R WRL"),
    38: ("3 TE, 1 WR (right), 1 RB", "QB RB TEL TER TE2L WRR"),
    39: ("3 TE, 2 WR", "QBS TEL TER TE2R WRL WRR"),
    40: ("4 WR (3 left), 1 RB", "QBS RB WRL WSL WS2L WRR"),
    41: ("4 WR (3 right), 1 RB", "QBS RB WRL WRR WSR WS2R"),
    42: ("4 WR (all left), 1 RB", "QBS RB WRL WSL WS2L WS3L"),
    43: ("4 WR (all right), 1 RB", "QBS RB WRR WSR WS2R WS3R"),
    44: ("1 TE, 4 WR (left)", "QBS TER WRL WSL WS2L WS3L"),
    45: ("1 TE, 4 WR (right)", "QBS TEL WRR WSR WS2R WS3R"),
    46: ("5 WR (3 left)", "QBS WRL WSL WS2L WRR WSR"),
    47: ("5 WR (3 right)", "QBS WRL WSL WRR WSR WS2R"),
    48: ("Goal Line (Normal)", "QB RBL RBR TEL TER TE2R"),
    49: ("Goal Line (I)", "QB FB RB TEL TER TE2R"),
    50: ("Goal Line (Offset I)", "QB FBR RB TEL TER TE2R"),
    51: ("Victory Formation", "QB RB RBL RBR TEL TER"),
    52: ("Field Goal", "H K TEL TER TE2L TE2R"),
    53: ("Punt Formation", "P PP WGL WGR WRL WRR"),
    56: ("2 TE, 3 WR (2 left)", "QBS TEL TER WRL WSL WRR"),
    57: ("2 TE, 3 WR (2 right)", "QBS TEL TER WRL WRR WSR"),
    58: ("3 RB, 1 TE, 1 WR (left)", "QB FB RBL RBR TER WRL"),
    59: ("3 RB, 1 TE, 1 WR (right)", "QB FB RBL RBR TEL WRR"),
    60: ("3 RB, 2 TE (Goal Line)", "QB FB RBL RBR TEL TER"),
    61: ("3 RB, 2 WR", "QB FB RBL RBR WRL WRR"),
    62: ("1 TE, 4 WR (2 each side)", "QBS TER WRL WSL WRR WSR"),
    63: ("1 TE, 4 WR (3 left)", "QBS TER WRL WSL WS2L WRR"),
    64: ("1 TE, 4 WR (3 right)", "QBS TEL WRL WRR WSR WS2R"),
    65: ("Goal Line, 4 TE, 1 RB", "QB RB TEL TER TE2L TE2R"),
    66: ("5 WR (4 left)", "QBS WRL WSL WS2L WS3L WRR"),
    67: ("5 WR (4 right)", "QBS WRL WRR WSR WS2R WS3R"),
    68: ("Offset I (Weak)", "QB FBL RB TER WRL WRR"),
    70: ("Both WR left (Offset I - Weak)", "QB FBL RB TER WRL WSL"),
    72: ("Both WR right (Offset I - Weak)", "QB FBL RB TEL WRR WSR"),
    74: ("3 WR (2 left), 2 RB (Offset I - Weak)", "QB FBL RB WRL WSL WRR"),
    77: ("3 WR (2 right), 2 RB (Offset I - Strong)", "QB FBR RB WRL WRR WSR"),
    78: ("3 WR (all left), 2 RB (Offset I - Weak)", "QB FBL RB WRL WSL WS2L"),
    81: ("3 WR (all right), 2 RB (Offset I - Strong)", "QB FBR RB WRR WSR WS2R"),
    82: ("2 TE, 1 WR (left), 2 RB (Offset I - Weak)", "QB FBL RB TEL TER WRL"),
    85: ("2 TE, 1 WR (right), 2 RB (Offset I - Strong)", "QB FBR RB TEL TER WRR"),
    86: ("Goal Line Offset I (Weak)", "QB FBL RB TEL TER TE2R"),
    88: ("3 RB, 2 WR (left)", "QB FB RBL RBR WRL WSL"),
    89: ("3 RB, 2 WR (right)", "QB FB RBL RBR WRR WSR"),
    91: ("Polecat", "QBS RBL RBR WRL WRR WSL"),
    92: ("Goal Line, 5 TE", "QB TEL TER TE2L TE2R TE3R"),
    93: ("Goal Line, 4 TE + 1 WR (right)", "QB TEL TER TE2L TE2R WRR"),
    94: ("Goal Line, 4 TE + 1 WR (left)", "QB TEL TER TE2L TE2R WRL"),
    95: ("5 WR (all left)", "QBS WRL WSL WS2L WS3L WS4L"),
    96: ("5 WR (all right)", "QBS WRR WSR WS2R WS3R WS4R"),
    97: ("3 TE, 2 WR (right)", "QBS TEL TER TE2R WRR WSR"),
    98: ("3 TE, 2 WR (left)", "QBS TEL TER TE2L WRL WSL"),
}

# Ids removed as alignment-identical to a surviving formation. Ten are the "Offset I -
# Weak/Strong" block re-deriving entries from the 4-36 range. Three are a different problem: a
# fake is indistinguishable from the real thing at the snap by design, so it cannot be a
# formation label — it is a play call, and belongs in the play record, not here.
ALIASES: dict[int, int] = {
    69: 4, 71: 9, 73: 10, 75: 21, 76: 22, 79: 29, 80: 30, 83: 35, 84: 36, 87: 50,
    90: 61,
    54: 53,   # Fake Punt      -> Punt Formation
    55: 52,   # Fake Field Goal -> Field Goal
}

# Alignments that no camera can separate pre-snap, kept explicit so callers can surface the
# ambiguity rather than silently reporting the canonical label as certain.
UNOBSERVABLE_PAIRS: dict[int, tuple[str, ...]] = {
    52: ("Field Goal", "Fake Field Goal"),
    53: ("Punt Formation", "Fake Punt"),
}


@dataclass(frozen=True)
class Formation:
    """A formation in the cleaned taxonomy."""

    id: int
    name: str
    slots: frozenset[str]

    @property
    def qb_depth(self) -> str | None:
        if "QBS" in self.slots:
            return "shotgun"
        if "QB" in self.slots:
            return "under_center"
        return None

    @property
    def backfield(self) -> str:
        if {"K", "P"} & self.slots:
            return "special_teams"
        if "FB" in self.slots:
            return "i_form"
        if "FBL" in self.slots:
            return "offset_i_left"
        if "FBR" in self.slots:
            return "offset_i_right"
        backs = len({"RB", "RBL", "RBR"} & self.slots)
        if backs >= 2:
            return "split_backs"
        if backs == 1:
            return "shotgun_single" if "QBS" in self.slots else "single_back"
        return "empty"

    @property
    def personnel(self) -> str | None:
        """Offensive personnel grouping, e.g. '11' — comparable to nflverse offense_personnel."""
        if self.backfield == "special_teams":
            return None
        backs = len({"RB", "RBL", "RBR", "FB", "FBL", "FBR"} & self.slots)
        tes = sum(1 for s in self.slots if s.startswith("TE"))
        return f"{backs}{tes}"

    @property
    def strength(self) -> str:
        left = sum(1 for s in self.slots if s.endswith("L") and s[0] in "TW")
        right = sum(1 for s in self.slots if s.endswith("R") and s[0] in "TW")
        if left > right:
            return "left"
        return "right" if right > left else "balanced"


FORMATIONS: dict[int, Formation] = {
    fid: Formation(fid, name, frozenset(slots.split()))
    for fid, (name, slots) in _DEFINITIONS.items()
}

# Slot set -> formation id. Unique by construction now that aliases are removed.
_BY_SLOTS: dict[frozenset[str], int] = {f.slots: fid for fid, f in FORMATIONS.items()}

ALL_SLOTS: frozenset[str] = frozenset(s for f in FORMATIONS.values() for s in f.slots)


def canonical_id(formation_id: int) -> int:
    """Resolve an alias to its canonical formation id."""
    return ALIASES.get(formation_id, formation_id)


@dataclass(frozen=True)
class FormationMatch:
    """Result of a taxonomy lookup."""

    formation: Formation | None
    exact: bool
    missing: frozenset[str] = frozenset()   # slots the taxonomy expects but we did not observe
    extra: frozenset[str] = frozenset()     # slots observed that the matched formation lacks

    @property
    def ambiguous_with(self) -> tuple[str, ...]:
        """Other play types sharing this alignment — a fake looks like the real thing."""
        if self.formation is None:
            return ()
        return UNOBSERVABLE_PAIRS.get(self.formation.id, ())


def slots_to_formation(slots) -> FormationMatch:
    """Look up a set of occupied slots in the taxonomy.

    Falls back to the closest formation by symmetric difference when there is no exact match,
    so a partially-recovered alignment still yields a best guess plus what disagreed.
    """
    observed = frozenset(slots)
    exact = _BY_SLOTS.get(observed)
    if exact is not None:
        return FormationMatch(FORMATIONS[exact], exact=True)
    if not observed:
        return FormationMatch(None, exact=False)

    best_id = min(_BY_SLOTS.values(),
                  key=lambda fid: (len(FORMATIONS[fid].slots ^ observed), fid))
    best = FORMATIONS[best_id]
    return FormationMatch(best, exact=False,
                          missing=best.slots - observed, extra=observed - best.slots)


def classify_alignment(players: list[Alignment]) -> FormationMatch:
    """Tracked offensive alignments at the snap -> a formation match. End to end."""
    return slots_to_formation(assign_slots(players).keys())


__all__ = [
    "Alignment", "Formation", "FormationMatch",
    "to_snap_frame", "assign_slots", "slots_to_formation", "classify_alignment",
    "canonical_id", "FORMATIONS", "ALIASES", "ALL_SLOTS", "UNOBSERVABLE_PAIRS",
]
