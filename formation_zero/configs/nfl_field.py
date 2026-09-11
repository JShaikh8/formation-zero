"""Top-down model of an NFL field, mirroring roboflow/sports' SoccerPitchConfiguration.

This defines the **target** coordinate system for field registration: real-world (x, y)
positions, in **yards**, of the landmarks an operator/keypoint-detector can match against
pixels. `ViewTransformer` (from roboflow/sports) then solves the homography from clicked/
detected pixel points to these field coordinates.

Coordinate frame
----------------
    origin (0, 0) = back-left corner (back of the "left" end zone, bottom sideline)
    x : along the length of the field, 0 .. 120 yd  (includes two 10-yd end zones)
    y : across the width,              0 .. 53.333 yd
    left goal line  @ x = 10,  right goal line @ x = 110,  50-yd line @ x = 60

Dimensions (NFL rulebook)
-------------------------
    length incl. end zones : 120 yd
    width                  : 160 ft = 53.333 yd
    end zone depth         : 10 yd
    hash marks             : 70 ft 9 in = 23.583 yd from each sideline
                             (the two hash rows are 18 ft 6 in apart, centered)

The repeating 5-yd grid is intentionally regular — which is exactly why pure line/keypoint
calibration is ambiguous. We disambiguate using the play's line-of-scrimmage yard line from
play-by-play data (see formation_zero.data.pbp).
"""

from __future__ import annotations

from dataclasses import dataclass, field

YARD = 1.0
FOOT = 1.0 / 3.0
INCH = FOOT / 12.0

FIELD_LENGTH_YD: float = 120.0
FIELD_WIDTH_YD: float = 160 * FOOT          # 53.333
END_ZONE_DEPTH_YD: float = 10.0
HASH_FROM_SIDELINE_YD: float = 70 * FOOT + 9 * INCH   # 23.583

# Lateral (y) positions of the four longitudinal lines every yard line crosses.
SIDELINE_BOTTOM_Y: float = 0.0
SIDELINE_TOP_Y: float = FIELD_WIDTH_YD
HASH_BOTTOM_Y: float = HASH_FROM_SIDELINE_YD                 # 23.583
HASH_TOP_Y: float = FIELD_WIDTH_YD - HASH_FROM_SIDELINE_YD   # 29.750

# Lateral lines, named bottom -> top.
LATERAL_LINES: tuple[tuple[str, float], ...] = (
    ("sideline_b", SIDELINE_BOTTOM_Y),
    ("hash_b", HASH_BOTTOM_Y),
    ("hash_t", HASH_TOP_Y),
    ("sideline_t", SIDELINE_TOP_Y),
)


def _yard_line_x_positions() -> list[float]:
    """x of every painted cross-field line: end lines (0,120), goal lines, and every 5 yd."""
    xs = [0.0, FIELD_LENGTH_YD]                       # back of each end zone (end lines)
    x = END_ZONE_DEPTH_YD                             # left goal line @ 10
    while x <= FIELD_LENGTH_YD - END_ZONE_DEPTH_YD:   # .. right goal line @ 110
        xs.append(x)
        x += 5.0
    return sorted(set(xs))


def absolute_x_to_yard_number(x: float) -> int:
    """Painted yard *number* (0..50..0) for an absolute x. The 50 is at x=60; goal lines read 0."""
    yards_from_left_goal = x - END_ZONE_DEPTH_YD
    n = round(yards_from_left_goal)
    return 50 - abs(n - 50)


@dataclass
class NFLFieldConfiguration:
    """Vertices/labels of field landmarks in yards, for homography target points.

    Attributes
    ----------
    length, width : field dimensions (yd)
    vertices      : list of (x, y) landmark coordinates (yd)
    labels        : parallel list of names, e.g. "x60_hash_b" (50-yd line at bottom hash)
    edges         : index pairs forming the line segments (for drawing the top-down field)
    """

    length: float = FIELD_LENGTH_YD
    width: float = FIELD_WIDTH_YD
    vertices: list[tuple[float, float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.vertices:
            return
        index: dict[str, int] = {}
        for x in _yard_line_x_positions():
            xi = int(round(x))
            col_idx: list[int] = []
            for lat_name, y in LATERAL_LINES:
                label = f"x{xi:03d}_{lat_name}"
                index[label] = len(self.vertices)
                self.vertices.append((x, y))
                self.labels.append(label)
                col_idx.append(index[label])
            # Vertical segment (this cross-field line) connecting bottom->top sideline.
            for a, b in zip(col_idx, col_idx[1:]):
                self.edges.append((a, b))
        self._index = index

    # -- lookups -----------------------------------------------------------------

    def point(self, x_yardline: int, lateral: str) -> tuple[float, float]:
        """Field coords of a landmark, e.g. point(60, 'hash_b') -> the 50-yd-line bottom hash."""
        key = f"x{x_yardline:03d}_{lateral}"
        return self.vertices[self._index[key]]

    def line_of_scrimmage_x(self, yardline_100: int) -> float:
        """Absolute x (yd) of the LOS given nflverse `yardline_100` (yards from opponent goal).

        `yardline_100` is distance to the *defending* team's goal line, 0..100. We place it on
        the "left-attacking" convention: x = left_goal(10) + (100 - yardline_100).
        Use this to anchor the homography to a known yard line (kills the repeating-line ambiguity).
        """
        if not 0 <= yardline_100 <= 100:
            raise ValueError(f"yardline_100 must be 0..100, got {yardline_100}")
        return END_ZONE_DEPTH_YD + (100 - yardline_100)

    def vertices_array(self):
        """Vertices as an (N, 2) float numpy array (homography target points)."""
        import numpy as np

        return np.asarray(self.vertices, dtype=float)


__all__ = [
    "NFLFieldConfiguration",
    "FIELD_LENGTH_YD",
    "FIELD_WIDTH_YD",
    "HASH_FROM_SIDELINE_YD",
    "absolute_x_to_yard_number",
]
