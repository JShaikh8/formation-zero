"""Rejoin a player's track across the frames the tracker lost him, then fill the short gaps left.

ByteTrack sees one frame at a time, but a whole play is on disk, so this runs after it. Tracks of the
same take are joined where one ends and another starts soon after, near where the first was heading,
with teams not opposed (`links`). Gaps up to MAX_GAP_S inside a track are then filled on a straight
line, marked `filled` with no confidence (`fill`). The overlay draws those thin: an estimate, not a
sighting. Runs after the team vote, since colour is what keeps opposing players locked together apart,
and after the id-swap cuts, so the two halves of a swap join the players they really belong to.

Why (owner, 2026-11-01: players disappearing and reappearing during the play). On the 50 anchored takes
of plays 61-120, 301 s of live play: 1,313 times a box dropped out and came back, half of them 1-2
frames, 4.5% of player-frames missing; and 594 tracks ended inside the picture with a new one starting
beside them, median 5 frames later. Only 146 ends were players leaving the picture. At 300 of those
moments, run through the detector again: 40% had a box already held by a neighbour (two players locked
together), 36% a box the tracker had not taken, 24% nothing at all.

The join, on known answers: 2,429 unbroken stretches of real tracks cut at crowded moments, with every
real track end in the take to choose from. Right / wrong / left apart at a gap of 2 frames 99 / 0.6 /
0%, 5 frames 99 / 0.4 / 1%, 10 frames 96 / 2 / 2%, 15 frames 91 / 2 / 6%, 24 frames 82 / 3 / 14%. A
wider reach gets more of the long gaps (91% at 24 frames) for more wrong joins (4.5%), and a wrong join
is the worse failure: it swaps two teammates without saying so. Those stretches are easier than real
losses. By eye on real joins and fills, 48 pairs on plays 61-120 had 1 clearly wrong, a track on the
painted shield joined to a Chief, which WIDE now prevents, and 36 on fresh plays 121-142 had 1 clearly
wrong, with several in each too locked together to call. The straight fill against what the tracker
saw, in yards: median 0.09 at 2 frames, 0.24 at 15, 0.34 at 25, the worst point in a gap mostly under
1.5.

Result: 1,313 dropouts down to 36, all over a second long; player-frames missing 4.5% to 0.8%; frames
in live play with 20-24 players 54% to 72%, almost none over 24. 319 tracks still end beside a new one, about
6 a play, so a number can still change. Anchors and fronts held or improved: plays 1-60 54 of 55 and
51 of 55, plays 61-120 50 of 55 and 48 of 51, and plays 121-143, tracked for the first time with all of
this in place, 19 of 21 and 17 of 19. Pre-snap counts on the same plays did not move (filled boxes
barely occur before the snap); only 1.5% of filled boxes in live play sit on a teammate's real box.

WIDE: a track far wider than tall is the painted NFL shield or a man on the ground. It cannot be deleted
on shape, since lying players in the approved labels reach 3.0, but it is not worth a join.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP_S = 1.0         # the longest gap joined or filled
LEAD = 10               # frames at a track's end its velocity is read from
REACH = 0.6             # box heights: how far the two ends' predictions may miss each other...
REACH_PER_S = 1.0       # ...and how much more for each second of gap
WIDE = 1.9              # a track this many times wider than tall is joined to nothing


def velocity(f: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pixels a frame, from a straight line through a stretch of a track; still if it is too short."""
    if len(f) < 3:
        return 0.0, 0.0
    return float(np.polyfit(f, x, 1)[0]), float(np.polyfit(f, y, 1)[0])


def ends(boxes: pd.DataFrame) -> pd.DataFrame:
    """One row per track: first and last frame, foot point and velocity (px/frame) at each end, box height."""
    rows = []
    for t, g in boxes.sort_values("frame").groupby("track_id"):
        f = g["frame"].to_numpy(float)
        x, y, h = ((g["x1"] + g["x2"]) / 2).to_numpy(), g["y2"].to_numpy(), (g["y2"] - g["y1"]).to_numpy()
        head, tail = slice(0, LEAD), slice(-LEAD, None)
        rows.append({"track_id": int(t), "first": int(f[0]), "last": int(f[-1]), "x0": x[0], "y0": y[0], "x1": x[-1], "y1": y[-1],
                     "v0": velocity(f[head], x[head], y[head]), "v1": velocity(f[tail], x[tail], y[tail]),
                     "h0": float(np.median(h[head])), "h1": float(np.median(h[tail]))})
    return pd.DataFrame(rows)


def miss(a, b) -> float:
    """How far apart a track ending (`a`) and one starting (`b`) put each other, in box heights: each end
    is carried across the gap at its own velocity, and the two misses are averaged."""
    gap = b.first - a.last
    ahead = np.hypot(a.x1 + a.v1[0] * gap - b.x0, a.y1 + a.v1[1] * gap - b.y0)
    back = np.hypot(b.x0 - b.v0[0] * gap - a.x1, b.y0 - b.v0[1] * gap - a.y1)
    return float((ahead + back) / 2 / ((a.h1 + b.h0) / 2))


def links(boxes: pd.DataFrame, team_of: dict[int, int], fps: float, cuts=(), *, reach: float = REACH,
          per_s: float = REACH_PER_S, max_gap_s: float = MAX_GAP_S) -> dict[int, int]:
    """Pairs {later track: earlier track} to join: same label, teams not opposed, a gap under `max_gap_s`
    with no shot change in it, neither track wider than WIDE, and ends that predict each other within
    `reach` plus `per_s` a second of gap. Closest first; each end is used once."""
    e = ends(boxes)
    if e.empty:
        return {}
    label = boxes.groupby("track_id")["label"].first()
    wide = ((boxes["x2"] - boxes["x1"]) / (boxes["y2"] - boxes["y1"])).groupby(boxes["track_id"]).median()
    rows = [r for r in e.itertuples() if wide[r.track_id] < WIDE]
    pairs = []
    for a in rows:
        for b in rows:
            gap = b.first - a.last
            if not 0 < gap <= max_gap_s * fps or label[a.track_id] != label[b.track_id]:
                continue
            ta, tb = team_of.get(a.track_id, -1), team_of.get(b.track_id, -1)
            if ta >= 0 and tb >= 0 and ta != tb:
                continue
            if any(a.last < c <= b.first for c in cuts):
                continue
            m = miss(a, b)
            if m <= reach + per_s * gap / fps:
                pairs.append((m, a.track_id, b.track_id))
    used_a, used_b, out = set(), set(), {}
    for _, a, b in sorted(pairs):
        if a not in used_a and b not in used_b:
            used_a.add(a), used_b.add(b)
            out[b] = a
    return out


def stitch(boxes: pd.DataFrame, team_of: dict[int, int], fps: float, cuts=()) -> tuple[pd.DataFrame, dict[int, int], int]:
    """The on-field tracks of `boxes` joined by `links`, each chain under its earliest id. Returns the
    boxes, the teams for those ids, and how many joins were made."""
    off = boxes["off_field"] if "off_field" in boxes else pd.Series(False, index=boxes.index)
    join = links(boxes[~off], team_of, fps, cuts)
    root = {}
    for b in sorted(join, key=lambda t: boxes.loc[boxes["track_id"] == t, "frame"].min()):
        root[b] = root.get(join[b], join[b])
    if not root:
        return boxes, team_of, 0
    boxes = boxes.assign(track_id=boxes["track_id"].map(lambda t: root.get(int(t), int(t))))
    teams: dict[int, int] = {}
    for t, team in team_of.items():
        r = root.get(t, t)
        teams[r] = team if teams.get(r, -1) < 0 else teams[r]
    return boxes, teams, len(root)


def fill(boxes: pd.DataFrame, fps: float, cuts=(), *, max_gap_s: float = MAX_GAP_S) -> pd.DataFrame:
    """A box on every frame of each gap in a track up to `max_gap_s` long, straight between its two
    sides, marked `filled` with no confidence. Gaps across a shot change stay."""
    boxes = boxes.assign(filled=False)
    keys = ["x1", "y1", "x2", "y2"]
    new = []
    for _, g in boxes.sort_values("frame").groupby("track_id"):
        f = g["frame"].to_numpy()
        for i in np.flatnonzero((np.diff(f) > 1) & (np.diff(f) - 1 <= max_gap_s * fps)):
            a, b = int(f[i]), int(f[i + 1])
            if any(a < c <= b for c in cuts):
                continue
            row = g.iloc[np.full(b - a - 1, i)].copy()
            w = (np.arange(a + 1, b) - a)[:, None] / (b - a)
            row[keys] = g.iloc[i][keys].to_numpy(float) * (1 - w) + g.iloc[i + 1][keys].to_numpy(float) * w
            row["frame"], row["conf"], row["filled"] = np.arange(a + 1, b), np.nan, True
            new.append(row)
    if new:
        boxes = pd.concat([boxes, *new]).sort_values(["frame", "track_id"])
    return boxes.reset_index(drop=True)


__all__ = ["LEAD", "MAX_GAP_S", "REACH", "REACH_PER_S", "WIDE", "ends", "fill", "links", "miss", "stitch", "velocity"]
