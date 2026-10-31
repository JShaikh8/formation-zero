"""Bird's-eye positions of tracked players in absolute field yards, and the bird's-eye video (milestone 4).

    PYTHONPATH=. .venv-sam3d/bin/python -W ignore -m formation_zero.perception.field.topdown \\
        --season 2026 --week 1 --away SF --home LA --plays 1-10 --video

Writes, per sideline take, derived/tracking/<game>_pNNN.sideline.topdown.parquet (one row per tracked
box: frame, track_id, label, team, x_yd absolute or NaN, y_yd, x_rel_yd in the take's own relative
frame, easy = registered directly on that frame, source = direct | bridged | interpolated |
extrapolated) and a .topdown.json beside it (snap, steps, anchor); with --video, an MP4 on the film
drive under birdseye/. End-zone takes are not handled (their registration is relative only).

1. Camera motion, every frame: `frame_motion` tracks corners of the picture (players' boxes left out)
   from each frame to the next and fits a homography to them. The press-box camera pans, tilts and
   zooms about a fixed point, so one homography moves the whole picture, crowd and all.
2. Registration, every EVERY frames: `homography.register` (relative: right up to whole 5-yd steps).
3. `bridge`: registered frames are put in one relative frame. Each one is predicted from the one
   before through the camera motion; the whole 5-yd step between prediction and registration is
   the change of relative frame (`step_offset`), and what is left over says how sure that is. A
   registration that disagrees with the motion from both its neighbours is dropped. Frames between
   registrations blend the motion carried forward from one and back from the other (`blend`), so
   the drift of the motion is spread over the gap; past the first or last registration the motion
   alone carries the homography, for at most EXTRAPOLATE_S.
4. `anchor_take`: on the last ANCHOR_SET_S before the snap (`snap_frame`), the tracked players' feet
   and teams give the line of scrimmage, and `homography.anchor` against the official yardline_100
   gives the one shift (whole 5-yd steps) that makes the take absolute. Which club is team 0:
   `choose_team_club`. Kickoffs have no line of scrimmage and stay relative (x_yd NaN).
5. `positions`: feet (bottom middle; boxes cut off by the frame's bottom left out, as
   `homography.feet` says with the take's camera, `take_camera`) of every tracked box not off the
   field, through the absolute homography.
6. `render`: a to-scale field under the broadcast overlay frame (with the absolute yard lines the
   homography puts on it), dots by team, the official line of scrimmage, the camera's footprint.

Measured 2026-10-30 on 49ers-Rams (2026 wk1 SF at LA, 1080p) plays 1-10, sideline takes, 3,425 frames:
- registration of every frame: 2,807 fit (82%), 2,510 easy (73%; 65-89% a take).
- camera motion: a fit on all 3,415 frame pairs (530-780 corners kept, take medians), 12 ms a frame.
- carrying between registrations, scored on every easy frame not used, at the tracked players' feet,
  after whole steps: registrations every 2 to 25 frames all land 0.033-0.039 yd from the direct one
  (per-frame median; p95 0.11-0.12) and 0.45-0.46 yd worst (per-frame max, p95), the direct
  registrations' own scatter. Blending registrations without the motion: 0.50 yd worst (p95) every 2
  frames, 3.5 every 5, 7.7 every 10. EVERY = 10 for the cost: 0.10 s a frame in all, CPU (0.86 s a
  registration, 12 ms of motion), without the video.
- steps (every 10 frames): 111 of 253 consecutive registrations changed relative frame; the motion
  left 0.007 yd from a whole step (median; p99 0.06, max 0.14), none ambiguous (over STEP_SURE_YD).
  Taking the camera as still instead: 14 of 195 over 1 yd (max 2.2). Player continuity picks the
  same step on 245 of 253; the 8 others all fall in p005's punt pan (frames 4670-4810), where its best
  step still leaves the players 5-22 yd from where they were (ids switch in a fast pan); the painted
  numbers agree with the motion there by eye. Registrations dropped: none.
- the motion alone, carried from a registration and scored against later ones: 0.05 yd (per-frame
  median; p95 0.16) within 1 s, 0.08 (p95 0.26, worst p95 0.74) at 3-4 s, 0.10 (0.52; 1.2) at 5-6 s,
  where steps start to drift (leftover up to 1.9 yd). Hence EXTRAPOLATE_S = 4: every frame of the 10
  takes then has a homography, 15% of them extrapolated (mostly close-ups after the whistle).
- anchor: 9 of 10 takes (p001 is a kickoff). Team 0 = SF: summed |residual| 1.39 yd against 12.24
  with the clubs swapped. Every one of the 38 frames before each snap gives the same shift; the
  players' line lands 0.03-0.33 yd from the official one (median 0.15).
- at the snap (5 frames before it): the offense's three front-most players stand 0.10-1.10 yd behind
  the official line (median 0.79) and the defense's 0.42-1.37 beyond (median 0.80), on 7 of 9 takes.
  All five players over 0.5 yd on the wrong side (p002 1, p004 2, p005 1, p007 1) carry the other
  club's team label (the tracker's), by eye; on p004 and p005 they move the front means.
- feet: the take's camera (`take_camera`) came out 86.5-89.3 yd back and 27.1-27.4 yd up; with it,
  `homography.feet` leaves out a further 584 of 79,306 boxes whose head needs feet below the frame
  (near benches seen over the heads in front, which landed up to 2.9 yd inside the field, and legs
  hidden at the bottom of close-ups); the 90th percentile of players on the field drops by one on
  p004, p008 and p010; nothing changes before the snap.
- players on the field (within FIELD_MARGIN_YD): 22 on 58% of frames before the snap, 21-23 on 89%,
  20-24 on 95%; over whole takes 20-24 on 68% (take medians 19-22, p10 6-19: tight shots after the
  snap). Officials: 1-2 a frame (median).
- speeds (SMOOTH-frame centred mean, consecutive frames): 3.0% of 78,634 samples over MAX_SPEED, at
  the same rate on direct (2.7%) and bridged (2.9%) frames, 4.3% on extrapolated ones, so mostly not
  the homography: by eye the worst are a box's bottom jumping when its player is hidden or merged
  with another, and id switches. Over 0.4 s instead: p99 9.9 yd/s, 0.54% over MAX_SPEED.
- formation: the shallowest offense player within 1.5 yd across of the ball and over 2.3 yd behind
  the line, at the snap: 4.8-5.8 yd deep on the four '(Shotgun)' plays where there is one (p002, 3, 7,
  8), 7.4 and 8.5 on the two without (p006, 10: the back behind a quarterback under centre, who
  stands within the line's 2.3 yd); nobody there on p004 and p009 (shotgun). Rule set after looking.

Held out, 2026-10-31: plays 11-60 of the same game, none looked at while building (50 sideline takes,
18,627 frames; 0.11 s a frame with the video), nothing changed after the run:
- 421 frames (2.3%) without a homography, 167 of them on p019; 2 registrations dropped; no step in
  doubt. 77% of frames bridged, 15% extrapolated, 8% direct.
- anchor: 45 of 47 scrimmage plays (the two kickoffs stay relative); the players' line 0.16 yd from
  the official one (median; p90 0.5, max 0.9). Left relative: p019, a pass from the 2 with the end
  zone filling the picture (1 easy registration of 37, no snap seen); p029, where four 49ers in the
  clump at the line carry the Rams' team label and the line comes out 1.7 yd off, which the check
  refuses. Team 0 = SF again (12.0 against 52.5).
- by eye, one random frame of each anchored take (46, from 12 s before the snap to 9 s after): the
  drawn yard lines sit on the painted lines and numbers on all 46, the goal line on the pylon on two
  end-zone corners after the whistle.
- fronts at the snap in the band on 39 of 46 (plays 1-10 the same way: 8 of 9); two of the seven miss
  it by 0.02 yd, the others have one or two players on the wrong side, on p042 two Rams carrying the
  49ers' label (by eye).
- players on the field, last 1.5 s before the snap: 22 on 42% of frames, 21-23 on 83%, 20-24 on 93%
  (plays 1-10 the same way: 53, 92, 97%).
- formation rule, unchanged: decides on 23 of 42 pass and run plays and agrees with '(Shotgun)' on
  21; the two others are runs with a player 3.9 and 4.3 yd deep that the play-by-play does not call
  shotgun (not resolved by eye). Of the 19 it leaves, 17 are under centre; on 14 nobody stands behind
  the ball, 13 of them under centre (on plays 1-10 both such were shotgun), so that case decides
  nothing alone.

Field frame: nfl_field's, x along the field (left goal line 10, the 50 at 60, right goal line 110 as
the sideline camera sees it), y across from the near sideline (0) to the far one (53.33).
"""

from __future__ import annotations

import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from formation_zero.configs.nfl_field import (
    END_ZONE_DEPTH_YD,
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    HASH_FROM_SIDELINE_YD,
)
from formation_zero.perception.field import homography as hg

PRIOR = (-87.3, 27.2)       # the game camera (y, height) of the clear game, yd (homography docstring)
EVERY = 10                  # frames between registrations (measured above)
STEP = hg.YARD_LINE_STEP    # 5 yd: the relative frame is right up to whole steps of this
STEP_SURE_YD = 1.0          # a step whose leftover is under this is sure; between this and 2.5, ambiguous
AGREE_YD = 1.0              # a registration agrees with the motion-carried one if within this (x mod 5, and y)
EXTRAPOLATE_S = 4.0         # the motion alone carries the homography this far past the registrations
FLOW_SCALE = 0.5            # `frame_motion` works on frames scaled by this
FLOW_MIN_INLIERS = 40       # ... and trusts a fit kept by this many tracked corners
FLOW_RANSAC_PX = 1.0        # ... within this (scaled px)
SNAP_SPEED = 2.0            # yd/s: the players' median speed passes this at the snap ...
SNAP_HOLD_S = 0.4           # ... and stays over it this long; the snap is just after the last frame
SNAP_STILL = 0.5            # ... before that with the median under this (the set)
SMOOTH = 5                  # frames: positions are smoothed over this centred window before speeds
MAX_SPEED = 11.0            # yd/s: nobody runs faster; above it is a registration or tracking error
ANCHOR_SET_S = 1.5          # `anchor_take` uses the last this many seconds before the snap
FIELD_MARGIN_YD = 1.0       # "on the field" in counts: within this of the field's edges


# ---------------------------------------------------------------- homographies through a take


def shift(g: np.ndarray, dx: float) -> np.ndarray:
    """Image -> field homography g with dx added to field x."""
    return hg.to_absolute(g, dx)


def probe_points(shape: tuple[int, int], n: tuple[int, int] = (7, 5), top: float = 0.3) -> np.ndarray:
    """A grid of image points (px) over the lower part of a frame of `shape` (h, w), where the field is."""
    h, w = shape
    xs, ys = np.linspace(0.05 * w, 0.95 * w, n[0]), np.linspace(top * h, 0.97 * h, n[1])
    return np.array([(x, y) for y in ys for x in xs], float)


def on_field(g: np.ndarray, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Which image points `pts` g puts on the field across (y within margin of 0 .. 53.33)."""
    f = hg.apply(g, pts)
    return np.isfinite(f).all(1) & (f[:, 1] >= -margin) & (f[:, 1] <= FIELD_WIDTH_YD + margin)


def blend(ga: np.ndarray, gb: np.ndarray, w: float, pts: np.ndarray) -> np.ndarray:
    """The homography between ga (w = 0) and gb (w = 1): the field points of the image points `pts`
    through each, moved linearly, then refitted. Both in the same field frame."""
    if w <= 0:
        return ga
    if w >= 1:
        return gb
    f = (1 - w) * hg.apply(ga, pts) + w * hg.apply(gb, pts)
    g = hg.dlt(pts, f)
    return ga if g is None else g


def step_offset(g_pred: np.ndarray, g_obs: np.ndarray, pts: np.ndarray) -> dict | None:
    """The whole 5-yd step that takes a registration g_obs into the field frame of g_pred (the same
    frame's homography carried from elsewhere): field x of the image points `pts` that both put on
    the field. Returns {"dx": step to add to g_obs's x (a multiple of 5), "left": yd from the nearest
    step (0 = sure, 2.5 = a coin toss), "dy": median gap across the field (yd)} or None."""
    ok = on_field(g_pred, pts, 2.0) & on_field(g_obs, pts, 2.0)
    if ok.sum() < 4:
        return None
    fp, fo = hg.apply(g_pred, pts[ok]), hg.apply(g_obs, pts[ok])
    d = float(np.median(fp[:, 0] - fo[:, 0]))
    dx = STEP * round(d / STEP)
    return {"dx": dx, "left": abs(d - dx), "dy": float(np.median(np.abs(fp[:, 1] - fo[:, 1])))}


def carry(g: np.ndarray, motions: list[np.ndarray | None], forward: bool = True) -> list[np.ndarray] | None:
    """g carried through frame-to-frame motions (each maps a frame's px to the previous frame's px):
    forward, the homographies of the frames after g's, one per motion; backward (motions in time
    order, ending at g's frame), those before, nearest first. None if a motion is missing."""
    out = []
    seq = motions if forward else motions[::-1]
    for m in seq:
        if m is None:
            return None
        g = g @ m if forward else g @ np.linalg.inv(m)
        g = g / np.linalg.norm(g)
        out.append(g)
    return out


def bridge(frames: list[int], regs: dict[int, np.ndarray], motion: dict[int, np.ndarray | None],
           shape: tuple[int, int], fps: float, extrapolate_s: float = EXTRAPOLATE_S) -> dict:
    """Image -> relative field homographies for every frame of a take, in one relative frame.

    `frames`: the take's frames in order; `regs`: frame -> registered G (easy ones only), each in its
    own relative frame; `motion`: frame -> homography from that frame's px to the previous frame's
    (None where it failed). Returns {"G": frame -> G, "source": frame -> "direct" | "bridged" |
    "interpolated" | "extrapolated", "steps": [dict per consecutive pair of registrations kept],
    "dropped": registered frames left out}.

    Registrations are chained in time: each is predicted from the last one kept by carrying it
    through the motion, and `step_offset` puts it in the first one's frame. First, one that
    disagrees with the prediction from each neighbour (x off a whole step, or y, by more than
    AGREE_YD) while those neighbours agree elsewhere is dropped as the odd one out. Where the motion
    is missing between two registrations the step is taken as if the camera held still, marked
    "still", and their frames are "interpolated" (`blend` of the two registrations) instead of
    "bridged" (`blend` of the motion carried forward from one and back from the other).
    """
    pts = probe_points(shape)
    pos = {f: i for i, f in enumerate(frames)}

    def pair(a: int, ga: np.ndarray, b: int, gb: np.ndarray) -> dict | None:
        ms = [motion.get(frames[i]) for i in range(pos[a] + 1, pos[b] + 1)]
        c = carry(ga, ms, forward=True)
        st = step_offset(c[-1] if c else ga, gb, pts)
        if st is not None:
            st.update(a=a, b=b, via="motion" if c else "still",
                      agree=bool(st["left"] <= AGREE_YD and st["dy"] <= AGREE_YD))
        return st

    order = [f for f in frames if f in regs]
    pairs = [pair(a, regs[a], b, regs[b]) for a, b in pairwise(order)]
    ok = [p is not None and p["agree"] for p in pairs]
    dropped = []
    for i, f in enumerate(order):        # odd one out: disagrees with every neighbour, which agree elsewhere
        left = ok[i - 1] if i > 0 else None
        right = ok[i] if i < len(pairs) else None
        if left is None and right is None:
            continue
        if left is None:
            odd = not right and i + 1 < len(pairs) and ok[i + 1]
        elif right is None:
            odd = not left and i - 2 >= 0 and ok[i - 2]
        else:
            odd = not left and not right
        if odd:
            dropped.append(f)
    order = [f for f in order if f not in dropped]
    kept: dict[int, np.ndarray] = {}
    steps = []
    for b in order:
        if not kept:
            kept[b] = regs[b]
            continue
        a = max(kept)
        st = pair(a, kept[a], b, regs[b])
        if st is None:
            dropped.append(b)
            continue
        kept[b] = shift(regs[b], st["dx"])
        steps.append(st)
    out_g: dict[int, np.ndarray] = {}
    source: dict[int, str] = {}
    ks = sorted(kept)
    for f in ks:
        out_g[f], source[f] = kept[f], "direct"
    for a, b in pairwise(ks):
        inner = frames[pos[a] + 1:pos[b]]
        if not inner:
            continue
        ms = [motion.get(frames[i]) for i in range(pos[a] + 1, pos[b] + 1)]
        fwd = carry(kept[a], ms[:-1], forward=True) if len(ms) > 1 else []
        bwd = carry(kept[b], ms[1:], forward=False) if len(ms) > 1 else []
        n = pos[b] - pos[a]
        for j, f in enumerate(inner, start=1):
            w = j / n
            if fwd is not None and bwd is not None:
                out_g[f], source[f] = blend(fwd[j - 1], bwd[len(inner) - j], w, pts), "bridged"
            else:
                out_g[f], source[f] = blend(kept[a], kept[b], w, pts), "interpolated"
    if ks:
        reach = round(extrapolate_s * fps)
        first, last = pos[ks[0]], pos[ks[-1]]
        g = kept[ks[-1]]
        for i in range(last + 1, min(len(frames), last + 1 + reach)):
            m = motion.get(frames[i])
            if m is None:
                break
            g = g @ m
            g = g / np.linalg.norm(g)
            out_g[frames[i]], source[frames[i]] = g, "extrapolated"
        g = kept[ks[0]]
        for i in range(first, max(0, first - reach), -1):
            m = motion.get(frames[i])
            if m is None:
                break
            g = g @ np.linalg.inv(m)
            g = g / np.linalg.norm(g)
            out_g[frames[i - 1]], source[frames[i - 1]] = g, "extrapolated"
    return {"G": out_g, "source": source, "steps": steps, "dropped": dropped}


def frame_motion(prev: np.ndarray, cur: np.ndarray, boxes: np.ndarray | None = None,
                 scale: float = FLOW_SCALE) -> tuple[np.ndarray | None, int]:
    """Homography from `cur`'s px to `prev`'s px (two grey uint8 frames, full size) from corners
    tracked by pyramidal Lucas-Kanade, players' boxes (n, 4; full-size px) masked out, forward-
    backward checked, RANSAC. Returns (M or None, corners kept). Needs OpenCV."""
    import cv2

    small = [cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for x in (prev, cur)]
    mask = np.full(small[0].shape, 255, np.uint8)
    if boxes is not None:
        for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4) * scale:
            mask[max(0, int(y1) - 3):int(y2) + 4, max(0, int(x1) - 3):int(x2) + 4] = 0
    p0 = cv2.goodFeaturesToTrack(small[0], maxCorners=800, qualityLevel=0.01, minDistance=8, mask=mask)
    if p0 is None or len(p0) < 8:
        return None, 0
    lk = {"winSize": (21, 21), "maxLevel": 3,
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(small[0], small[1], p0, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(small[1], small[0], p1, None, **lk)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.hypot(*(back - p0).reshape(-1, 2).T) < 0.5)
    if ok.sum() < 8:
        return None, int(ok.sum())
    m, inl = cv2.findHomography(p1[ok], p0[ok], cv2.RANSAC, FLOW_RANSAC_PX)
    n = int(inl.sum()) if inl is not None else 0
    if m is None or n < FLOW_MIN_INLIERS:
        return None, n
    s = np.diag([scale, scale, 1.0])
    m = np.linalg.inv(s) @ m @ s
    return m / m[2, 2], n


# ---------------------------------------------------------------- the snap, the anchor, positions


def smooth_tracks(pos: pd.DataFrame, window: int = SMOOTH, fps: float = 25.0) -> pd.DataFrame:
    """x_yd, y_yd smoothed per track over a centred window of `window` frames (in time: a track's
    positions either side of a gap of two frames or more are never averaged together), and speed
    (yd/s) between consecutive smoothed positions at most two frames apart."""
    out = []
    for _, t in pos.sort_values("frame").groupby("track_id", sort=False):
        t = t.drop_duplicates("frame").copy()
        frames = t["frame"].to_numpy()
        full = pd.RangeIndex(frames.min(), frames.max() + 1)
        for c in ("x_yd", "y_yd"):
            r = pd.Series(t[c].to_numpy(), index=frames).reindex(full).rolling(window, center=True, min_periods=1).mean()
            t[c + "_s"] = r.loc[frames].to_numpy()
        df = t["frame"].diff()
        v = np.hypot(t["x_yd_s"].diff(), t["y_yd_s"].diff()) / df * fps
        t["speed"] = v.where(df <= 2)
        out.append(t)
    return pd.concat(out) if out else pos.assign(x_yd_s=[], y_yd_s=[], speed=[])


def snap_frame(pos: pd.DataFrame, fps: float, speed: float = SNAP_SPEED, hold_s: float = SNAP_HOLD_S,
               still: float = SNAP_STILL) -> int | None:
    """The snap: the players' median speed (positions in any one field frame, smoothed:
    `smooth_tracks`) first stays over `speed` for `hold_s`; the snap is the frame after the last one
    before that with the median under `still` (the set). None if the players never move together,
    or never stand still before they do (the take starts after the snap)."""
    p = pos[(pos["label"] == "player") & pos["speed"].notna()]
    if p.empty:
        return None
    med = p.groupby("frame")["speed"].median().sort_index()
    hold = max(1, round(hold_s * fps))
    fast = (med > speed).to_numpy()
    for i in range(len(fast) - hold + 1):
        if fast[i:i + hold].all():
            set_ = np.flatnonzero(med.to_numpy()[:i] < still)
            return int(med.index[set_[-1] + 1]) if len(set_) else None
    return None


def anchor_take(pos: pd.DataFrame, offense_team: int, yardline_100: float, snap: int, fps: float,
                set_s: float = ANCHOR_SET_S) -> dict | None:
    """The take's absolute shift from the frames in the last `set_s` before the snap: per frame,
    `homography.line_of_scrimmage` on the players' relative x (team known, on the field) and
    `homography.anchor`; the shift most frames give wins. Returns {"shift", "frames", "agree": share
    of frames giving it, "residual" (median, yd), "attack", "los_rel"} or None."""
    lo = snap - round(set_s * fps)
    p = pos[(pos["frame"] >= lo) & (pos["frame"] < snap) & (pos["label"] == "player") & (pos["team"] >= 0)
            & (pos["y_yd"] > -1) & (pos["y_yd"] < FIELD_WIDTH_YD + 1)]
    res = []
    for f, rows in p.groupby("frame"):
        los = hg.line_of_scrimmage(rows["x_yd"].to_numpy(), (rows["team"] == offense_team).to_numpy())
        if los is None:
            continue
        a = hg.anchor(los["x"], los["attack"], yardline_100)
        res.append((int(f), a["shift"], a["residual"], los["attack"], los["x"], los["misplaced"]))
    if not res:
        return None
    r = pd.DataFrame(res, columns=["frame", "shift", "residual", "attack", "los_rel", "misplaced"])
    best = r["shift"].mode().iloc[0]
    k = r[r["shift"] == best]
    return {"shift": float(best), "frames": len(r), "agree": float(len(k) / len(r)),
            "residual": float(k["residual"].median()), "attack": int(k["attack"].mode().iloc[0]),
            "los_rel": float(k["los_rel"].median()), "misplaced": float(k["misplaced"].median())}


def take_camera(gs: dict[int, np.ndarray], source: dict[int, str], shape: tuple[int, int]) -> np.ndarray | None:
    """The camera centre (field yd, z up) in the frame of the homographies `gs`: the median of
    `homography.camera` over the frames registered directly (the camera pans and zooms but stays
    put). None without any."""
    cams = [hg.camera(np.linalg.inv(g), shape) for f, g in gs.items() if source.get(f) == "direct"]
    cs = [c["C"] for c in cams if c is not None]
    return np.median(np.array(cs), axis=0) if cs else None


def positions(boxes: pd.DataFrame, gs: dict[int, np.ndarray], source: dict[int, str],
              shape: tuple[int, int], team_of: dict[int, int] | None = None,
              centre: np.ndarray | None = None) -> pd.DataFrame:
    """Field (x, y) yd of the feet of every tracked box not off the field and not cut off by the
    bottom of the frame (`homography.feet`, with the camera `centre` when given), on frames with a
    homography `gs` (frame -> image -> field)."""
    b = boxes[~boxes.get("off_field", pd.Series(False, index=boxes.index)).astype(bool)]
    b = b[b["frame"].isin(gs.keys())]
    rows = []
    for f, r in b.groupby("frame"):
        feet, whole = hg.feet(r[["x1", "y1", "x2", "y2"]].to_numpy(float), shape, gs[f], centre)
        xy = hg.apply(gs[f], feet[whole])
        rr = r[whole]
        rows.append(pd.DataFrame({"frame": int(f), "track_id": rr["track_id"].astype(int).to_numpy(),
                                  "label": rr["label"].to_numpy(), "x_yd": xy[:, 0], "y_yd": xy[:, 1],
                                  "source": source[f]}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["frame", "track_id", "label", "x_yd", "y_yd", "source"])
    team = team_of or {}
    out["team"] = [team.get(int(t), -1) if lab == "player" else -1 for t, lab in zip(out["track_id"], out["label"])]
    out["easy"] = out["source"] == "direct"
    return out[["frame", "track_id", "label", "team", "x_yd", "y_yd", "easy", "source"]]


def in_field(x, y, margin: float = FIELD_MARGIN_YD):
    return (x >= -margin) & (x <= FIELD_LENGTH_YD + margin) & (y >= -margin) & (y <= FIELD_WIDTH_YD + margin)


# ---------------------------------------------------------------- drawing


HASH_NEAR = HASH_FROM_SIDELINE_YD
HASH_FAR = FIELD_WIDTH_YD - HASH_FROM_SIDELINE_YD
TEAM_COLOURS = {0: (255, 210, 63), 1: (63, 208, 255)}      # as the broadcast overlay's boxes
REFEREE = (255, 93, 115)
UNKNOWN = (235, 235, 235)
LOS_COLOUR = (40, 110, 255)
GRASS, END_ZONE, PAINT = (46, 112, 58), (34, 88, 46), (238, 238, 238)


class Field:
    """A to-scale top-down field, `ppy` px a yard, with `margin` yd around it: the near sideline at
    the bottom and x to the right, as the sideline camera sees it."""

    def __init__(self, ppy: float = 10.0, margin: float = 4.0):
        self.ppy, self.margin = ppy, margin
        self.width = round((FIELD_LENGTH_YD + 2 * margin) * ppy)
        self.height = round((FIELD_WIDTH_YD + 2 * margin) * ppy)
        self._base = None

    def px(self, x, y):
        """Field yd -> canvas px (arrays or scalars)."""
        return ((np.asarray(x, float) + self.margin) * self.ppy,
                (FIELD_WIDTH_YD + self.margin - np.asarray(y, float)) * self.ppy)

    def base(self) -> np.ndarray:
        """The empty field (RGB uint8, height x width): grass, end zones, yard lines every 5 yd,
        numbers every 10, hash marks and sideline ticks every yard."""
        if self._base is not None:
            return self._base.copy()
        from PIL import Image, ImageDraw, ImageFont

        im = Image.new("RGB", (self.width, self.height), (30, 70, 38))
        d = ImageDraw.Draw(im)
        d.rectangle([*self.px(0, FIELD_WIDTH_YD), *self.px(FIELD_LENGTH_YD, 0)], fill=GRASS)
        for x0 in (0.0, FIELD_LENGTH_YD - END_ZONE_DEPTH_YD):
            d.rectangle([*self.px(x0, FIELD_WIDTH_YD), *self.px(x0 + END_ZONE_DEPTH_YD, 0)], fill=END_ZONE)
        lw = max(1, round(self.ppy / 6))
        for x in np.arange(END_ZONE_DEPTH_YD, FIELD_LENGTH_YD - END_ZONE_DEPTH_YD + 1, 5):
            d.line([self.px(x, 0), self.px(x, FIELD_WIDTH_YD)], fill=PAINT, width=lw * (2 if x % 50 == 10 else 1))
        tick = 2 / 3                          # 2-ft marks
        for x in range(int(END_ZONE_DEPTH_YD) + 1, int(FIELD_LENGTH_YD - END_ZONE_DEPTH_YD)):
            if x % 5 == 0:
                continue
            for y0 in (hg.TICK_IN_YD - 1 / 3, HASH_NEAR - tick, HASH_FAR, FIELD_WIDTH_YD - hg.TICK_IN_YD - 1 / 3):
                d.line([self.px(x, y0), self.px(x, y0 + tick)], fill=PAINT, width=1)
        d.rectangle([*self.px(0, FIELD_WIDTH_YD), *self.px(FIELD_LENGTH_YD, 0)], outline=PAINT, width=lw * 2)
        font = ImageFont.load_default(size=max(10, round(2.0 * self.ppy)))
        for x in range(20, 101, 10):
            n = str(50 - abs(x - 60))
            for y, rot in ((10.0, 0), (FIELD_WIDTH_YD - 10.0, 180)):
                cx, cy = self.px(x, y)
                if rot:
                    t = Image.new("RGBA", (int(6 * self.ppy), int(3 * self.ppy)), (0, 0, 0, 0))
                    ImageDraw.Draw(t).text((t.width / 2, t.height / 2), n, fill=PAINT, font=font, anchor="mm")
                    t = t.rotate(180)
                    im.paste(t, (int(cx - t.width / 2), int(cy - t.height / 2)), t)
                else:
                    d.text((cx, cy), n, fill=PAINT, font=font, anchor="mm")
        self._base = np.asarray(im)
        return self._base.copy()

    def draw(self, pos: pd.DataFrame, los_x: float | None = None, footprint: np.ndarray | None = None,
             label_ids: bool = True) -> np.ndarray:
        """The field with one frame's positions (rows of x_yd, y_yd, label, team, track_id), the
        official line of scrimmage, and `footprint` (field points of the picture's edge) outlined."""
        from PIL import Image, ImageDraw, ImageFont

        im = Image.fromarray(self.base())
        d = ImageDraw.Draw(im)
        if footprint is not None and len(footprint) > 2:
            fx, fy = self.px(footprint[:, 0], footprint[:, 1])
            d.line(list(zip(fx, fy)) + [(fx[0], fy[0])], fill=(150, 190, 150), width=1)
        if los_x is not None:
            d.line([self.px(los_x, -0.5), self.px(los_x, FIELD_WIDTH_YD + 0.5)], fill=LOS_COLOUR,
                   width=max(2, round(self.ppy / 4)))
        r = max(3, round(0.55 * self.ppy))
        font = ImageFont.load_default(size=max(9, round(1.1 * self.ppy)))
        for p in pos.itertuples():
            colour = REFEREE if p.label == "referee" else TEAM_COLOURS.get(int(p.team), UNKNOWN)
            cx, cy = self.px(p.x_yd, p.y_yd)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour, outline=(0, 0, 0))
            if label_ids:
                d.text((cx + r + 1, cy - r - 1), str(int(p.track_id)), fill=(255, 255, 255), font=font,
                       anchor="lb", stroke_width=1, stroke_fill=(0, 0, 0))
        return np.asarray(im)


def footprint(g: np.ndarray, shape: tuple[int, int], n: int = 12) -> np.ndarray:
    """Field points of a frame's edge (image -> field g), clipped across the field to -8 .. 61 yd."""
    h, w = shape
    t = np.linspace(0, 1, n, endpoint=False)
    edge = np.vstack([np.column_stack([t * w, np.zeros(n)]), np.column_stack([np.full(n, w), t * h]),
                      np.column_stack([(1 - t) * w, np.full(n, h)]), np.column_stack([np.zeros(n), (1 - t) * h])])
    f = hg.apply(g, edge)
    f = f[np.isfinite(f).all(1)]
    f[:, 1] = np.clip(f[:, 1], -8, FIELD_WIDTH_YD + 8)
    return f


def yard_lines_on_picture(image: np.ndarray, g: np.ndarray, scale: float = 1.0, every: float = 5.0) -> np.ndarray:
    """The picture with the absolute yard lines (every `every` yd, goal line to goal line) drawn as
    the absolute homography g (image px at full size -> field) puts them, labelled with their painted
    number; `scale` maps full-size px to `image` px."""
    from PIL import Image, ImageDraw, ImageFont

    h_inv = np.linalg.inv(g)
    im = Image.fromarray(image)
    d = ImageDraw.Draw(im, "RGBA")
    font = ImageFont.load_default(size=max(12, image.shape[0] // 40))
    for x in np.arange(END_ZONE_DEPTH_YD, FIELD_LENGTH_YD - END_ZONE_DEPTH_YD + 0.1, every):
        p = hg.apply(h_inv, np.array([[x, -0.5], [x, FIELD_WIDTH_YD + 0.5]])) * scale
        if not np.isfinite(p).all():
            continue
        d.line([tuple(p[0]), tuple(p[1])], fill=(255, 0, 255, 200), width=2)
        n = 50 - abs(round(x) - 60)
        lab = "G" if n == 0 else str(n)
        d.text(tuple(p[1] + [0, -4]), lab, fill=(255, 0, 255, 255), font=font, anchor="mb",
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for y in (0.0, HASH_NEAR, HASH_FAR, FIELD_WIDTH_YD):
        xs = np.linspace(END_ZONE_DEPTH_YD, FIELD_LENGTH_YD - END_ZONE_DEPTH_YD, 41)
        p = hg.apply(h_inv, np.column_stack([xs, np.full_like(xs, y)])) * scale
        p = p[np.isfinite(p).all(1)]
        d.line([tuple(q) for q in p], fill=(255, 0, 255, 120), width=1)
    return np.asarray(im)


# ---------------------------------------------------------------- a take, end to end


def offense_team(team_club: dict[int, str], posteam: str) -> int | None:
    for t, c in team_club.items():
        if c == posteam:
            return t
    return None


def take_motion_and_registrations(source, start: int, end: int, boxes: pd.DataFrame, every: int = EVERY,
                                  regs: dict | None = None, motion: dict | None = None) -> tuple[dict, dict, dict]:
    """One pass over frames [start, end) of `source` (a FileSource): the camera motion into every
    frame (`frame_motion`, players' boxes masked) and `homography.register` every `every` frames.
    Precomputed `regs` (frame -> fit dict or None) or `motion` (frame -> M or None) skip that work.
    Returns (motion, registrations: frame -> {"G", "easy", "rms_px"} or None, timings)."""
    import cv2

    from formation_zero.perception.field import hashes

    by = {f: r[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, r in boxes.groupby("frame")}
    need_motion, need_regs = motion is None, regs is None
    motion = {} if motion is None else dict(motion)
    regs = {} if regs is None else {f: v for f, v in regs.items() if (f - start) % every == 0}
    prev, pf = None, None
    t_reg, t_mot, n_reg = 0.0, 0.0, 0
    for fr in source.iter_frames(start, end):
        if need_motion:
            grey = cv2.cvtColor(fr.image, cv2.COLOR_RGB2GRAY)
            if prev is not None:
                t = time.perf_counter()
                motion[fr.index] = frame_motion(prev, grey, by.get(pf))[0]
                t_mot += time.perf_counter() - t
            prev, pf = grey, fr.index
        if need_regs and (fr.index - start) % every == 0:
            t = time.perf_counter()
            fit = hg.register(fr.image, found=hashes.find(fr.image, "sideline"), prior=PRIOR)["fit"]
            t_reg += time.perf_counter() - t
            n_reg += 1
            regs[fr.index] = None if fit is None else {"G": fit["G"], "easy": fit["easy"], "rms_px": fit["rms_px"]}
        if not need_motion and not need_regs:
            break
    return motion, regs, {"register_s": t_reg, "registered": n_reg, "motion_s": t_mot}


def relative_take(paths, play_index: int, source, every: int = EVERY, regs: dict | None = None,
                  motion: dict | None = None) -> dict:
    """Everything about one sideline take that needs no anchor: boxes, homographies in one relative
    frame (`bridge`), relative positions, smoothed speeds, and the snap."""
    from formation_zero.perception.tracking import take_paths

    box_path, summary_path, _ = take_paths(paths, play_index, "sideline")
    summary = json.loads(summary_path.read_text())
    boxes = pd.read_parquet(box_path)
    start, end = summary["start_frame"], summary["end_frame"]
    team_of = {int(t["track_id"]): int(t.get("team", -1)) for t in summary["tracks_detail"]}
    shape = (source.height, source.width)
    motion, regs, timing = take_motion_and_registrations(source, start, end, boxes, every, regs, motion)
    easy = {f: r["G"] for f, r in regs.items() if r is not None and r["easy"]}
    frames = list(range(start, end))
    br = bridge(frames, easy, motion, shape, source.fps)
    centre = take_camera(br["G"], br["source"], shape)
    pos = positions(boxes, br["G"], br["source"], shape, team_of, centre)
    sm = smooth_tracks(pos, fps=source.fps)
    snap = snap_frame(sm, source.fps)
    return {"play_index": play_index, "play_uid": summary["play_uid"], "start": start, "end": end, "boxes": boxes,
            "summary": summary, "team_of": team_of, "shape": shape, "motion": motion, "regs": regs, "bridge": br,
            "pos": pos, "smooth": sm, "snap": snap, "timing": timing, "fps": source.fps, "camera": centre,
            "motion_missing": sum(motion.get(f) is None for f in frames[1:])}


def choose_team_club(takes: list[dict], pbp: pd.DataFrame, away: str, home: str) -> tuple[dict[int, str], dict]:
    """Which club is team 0 (the tracker's colour split names no club). Swapping the clubs flips the
    attack direction `line_of_scrimmage` finds, so the misplaced count cannot tell; but the official
    line then counts from the other goal, 100 - 2 * yardline_100 yd away, which moves the anchor's
    residual off the whole steps unless yardline_100 is a multiple of 5. The assignment with the
    smaller summed |residual| over the takes wins. Returns ({0: club, 1: club}, {club as team 0:
    (summed |residual|, takes)})."""
    score = {}
    for club0, club1 in ((away, home), (home, away)):
        tc, total, n = {0: club0, 1: club1}, 0.0, 0
        for t in takes:
            play = pbp.loc[t["play_uid"]] if t["play_uid"] in pbp.index else None
            if play is None or t["snap"] is None or play["play_type"] == "kickoff":
                continue
            a = anchor_take(t["pos"], offense_team(tc, play["posteam"]), float(play["yardline_100"]), t["snap"], t["fps"])
            if a is not None:
                total += abs(a["residual"])
                n += 1
        score[club0] = (round(total, 2), n)
    club0 = min(score, key=lambda c: score[c][0])
    return {0: club0, 1: home if club0 == away else away}, score


def absolute_take(t: dict, play: pd.Series | None, team_club: dict[int, str]) -> dict:
    """The take's anchor and its absolute positions: x_yd absolute (NaN when the take cannot be
    anchored), x_rel_yd in the take's own relative frame (`bridge`), y_yd absolute either way."""
    pos = t["pos"].copy()
    out = {"anchor": None, "los_x": None, "attack": None, "why": None}
    if play is None:
        out["why"] = "no play-by-play row"
    elif play["play_type"] == "kickoff":
        out["why"] = "kickoff: no line of scrimmage"
    elif t["snap"] is None:
        out["why"] = "no snap seen"
    else:
        off = offense_team(team_club, play["posteam"])
        a = anchor_take(pos, off, float(play["yardline_100"]), t["snap"], t["fps"])
        if a is None or abs(a["residual"]) > hg.ANCHOR_MAX_YD:
            out["why"] = "no line of scrimmage from the players" if a is None else "players' line too far from the official one"
            out["anchor"] = a
        else:
            out.update(anchor=a, attack=a["attack"], los_x=hg.los_absolute_x(float(play["yardline_100"]), a["attack"]),
                       offense_team=off)
    shift_x = out["anchor"]["shift"] if out["los_x"] is not None else np.nan
    pos.insert(pos.columns.get_loc("y_yd") + 1, "x_rel_yd", pos["x_yd"])
    pos["x_yd"] = pos["x_yd"] + shift_x
    out["pos"] = pos
    out["G"] = {f: shift(g, shift_x) for f, g in t["bridge"]["G"].items()} if out["los_x"] is not None else {}
    return out


def render(t: dict, ab: dict, play: pd.Series | None, overlay: Path | None, dst: Path, team_club: dict[int, str],
           field: Field | None = None) -> int:
    """The bird's-eye MP4 of one take: the broadcast overlay frame (with the absolute yard lines the
    homography puts on it) above the to-scale field."""
    from PIL import Image, ImageDraw, ImageFont

    from formation_zero.ingest.encode import VideoWriter

    field = field or Field()
    w = field.width
    bh = round(w * t["shape"][0] / t["shape"][1])
    head = 40
    hgt = head + bh + field.height
    hgt += hgt % 2
    scale = bh / t["shape"][0]
    by = dict(tuple(ab["pos"].groupby("frame")))
    frames = list(range(t["start"], t["end"]))
    pics = _overlay_frames(overlay, len(frames)) if overlay is not None and overlay.exists() else None
    font = ImageFont.load_default(size=15)
    title = f"{t['play_uid']}  " + ("" if play is None else f"Q{int(play['qtr'])} {play['posteam']} {play['play_type']} at "
                                                         f"{play['yrdln']}: {str(play['desc'])[:110]}")
    with VideoWriter(dst, t["fps"], w, hgt) as out:
        for i, f in enumerate(frames):
            canvas = np.zeros((hgt, w, 3), np.uint8)
            pic = next(pics) if pics is not None else None
            if pic is not None:
                if pic.shape[1] != w:
                    pic = np.asarray(Image.fromarray(pic).resize((w, bh)))
                if f in ab["G"]:
                    pic = yard_lines_on_picture(pic, ab["G"][f], scale)
                canvas[head:head + bh] = pic[:bh]
            p = by.get(f, ab["pos"].iloc[:0])
            p = p[in_field(p["x_yd"], p["y_yd"], field.margin)] if ab["los_x"] is not None else p.iloc[:0]
            fp = footprint(ab["G"][f], t["shape"]) if f in ab["G"] else None
            canvas[head + bh:head + bh + field.height] = field.draw(p, ab["los_x"], fp)
            im = Image.fromarray(canvas)
            d = ImageDraw.Draw(im)
            src = t["bridge"]["source"].get(f, "none")
            on = p[in_field(p["x_yd"], p["y_yd"])]
            npl, nref = int((on["label"] == "player").sum()), int((on["label"] == "referee").sum())
            snap = t["snap"]
            phase = "" if snap is None else ("  pre-snap" if f < snap else ("  SNAP" if f - snap < 5 else ""))
            d.text((8, 4), title, fill=(255, 255, 255), font=font)
            legend = (f"frame {f}  homography: {src}{phase}   on field: {npl} players, {nref} officials   "
                      f"{team_club[0]} gold, {team_club[1]} cyan, officials pink, blue = line of scrimmage")
            if ab["los_x"] is None:
                legend = f"frame {f}  NOT ANCHORED ({ab['why']}): no bird's-eye positions"
            d.text((8, 22), legend, fill=(220, 220, 220), font=font)
            out.write(np.asarray(im))
        return out.count


def _overlay_frames(path: Path, n: int):
    import av

    with av.open(str(path)) as c:
        for k, fr in enumerate(c.decode(video=0)):
            if k >= n:
                break
            yield fr.to_ndarray(format="rgb24")
    while True:
        yield None


def take_record(t: dict, ab: dict, play: pd.Series | None) -> dict:
    """The JSON summary of one take: registration, steps, anchor, snap and the checks."""
    src = pd.Series(t["bridge"]["source"]).value_counts().to_dict()
    regs = t["regs"]
    steps = t["bridge"]["steps"]
    rec = {"play_uid": t["play_uid"], "start_frame": t["start"], "end_frame": t["end"], "snap_frame": t["snap"],
           "registered": len(regs), "easy": sum(1 for r in regs.values() if r is not None and r["easy"]),
           "dropped": t["bridge"]["dropped"], "sources": src,
           "frames_without_homography": (t["end"] - t["start"]) - len(t["bridge"]["G"]),
           "motion_missing": t["motion_missing"],
           "steps": len(steps), "steps_changed": sum(1 for s in steps if s["dx"] != 0),
           "steps_unsure": sum(1 for s in steps if s["left"] > STEP_SURE_YD),
           "step_left_max": max([s["left"] for s in steps], default=None),
           "camera_y_z": None if t["camera"] is None else [float(t["camera"][1]), float(t["camera"][2])],
           "anchor": ab["anchor"], "los_x": ab["los_x"], "why_not": ab["why"], "timing": t["timing"]}
    return json.loads(json.dumps(rec, default=float))


def run(paths, plays: pd.DataFrame, every: int = EVERY, video: bool = True, cache: dict | None = None) -> list[dict]:
    """Bird's-eye positions (and videos) for the sideline takes of `plays` (play index rows).
    `cache`: play_index -> {"regs", "motion"} precomputed, to skip that work."""
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source

    pbp = pd.read_parquet(paths.pbp_path).dropna(subset=["play_uid"]).set_index("play_uid")
    takes = []
    with FileSource(best_source(paths)) as source:
        for play in plays.itertuples():
            c = (cache or {}).get(int(play.play_index), {})
            t0 = time.perf_counter()
            t = relative_take(paths, int(play.play_index), source, every, c.get("regs"), c.get("motion"))
            t["timing"]["total_s"] = time.perf_counter() - t0
            takes.append(t)
            print(f"  {t['play_uid']}: {t['end'] - t['start']} frames, snap {t['snap']}, "
                  f"{len(t['bridge']['G'])} with a homography, {t['timing']['total_s']:.0f}s", flush=True)
    team_club, score = choose_team_club(takes, pbp, paths.away, paths.home)
    print(f"  team 0 = {team_club[0]} (summed |anchor residual| with each club as team 0: {score})", flush=True)
    out = []
    for t in takes:
        play = pbp.loc[t["play_uid"]] if t["play_uid"] in pbp.index else None
        ab = absolute_take(t, play, team_club)
        stem = paths.tracking_dir / f"{t['play_uid']}.sideline"
        ab["pos"].to_parquet(stem.with_name(stem.name + ".topdown.parquet"), index=False)
        rec = take_record(t, ab, play)
        rec["team_club"] = {str(k): v for k, v in team_club.items()}
        stem.with_name(stem.name + ".topdown.json").write_text(json.dumps(rec, indent=1))
        if video:
            overlay = paths.raw_dir / "overlays" / f"p{t['play_index']:03d}.sideline.mp4"
            dst = paths.raw_dir / "birdseye" / f"p{t['play_index']:03d}.sideline.mp4"
            render(t, ab, play, overlay, dst, team_club)
        out.append({"take": t, "abs": ab, "record": rec})
        print(f"  {t['play_uid']}: anchor {rec['anchor'] and rec['anchor']['shift']} "
              f"(residual {rec['anchor'] and round(rec['anchor']['residual'], 2)}), {ab['why'] or 'ok'}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths
    from formation_zero.perception.tracking import _plays

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--plays", required=True, help="play_index numbers, e.g. 1-10 or 3,7,12")
    ap.add_argument("--every", type=int, default=EVERY, help="frames between registrations")
    ap.add_argument("--video", action="store_true", help="also write the bird's-eye MP4 of each take")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    if not paths.film_root_available():
        raise SystemExit(f"film drive not mounted: {paths.film_root}")
    index = pd.read_parquet(paths.play_index_path)
    plays = index[index["play_index"].astype(int).isin(_plays(a.plays))]
    run(paths, plays, a.every, a.video)
    return 0


__all__ = ["EVERY", "Field", "absolute_take", "anchor_take", "blend", "bridge", "carry", "choose_team_club", "footprint",
           "frame_motion", "positions", "probe_points", "relative_take", "render", "run", "shift", "smooth_tracks",
           "snap_frame", "step_offset", "take_camera", "yard_lines_on_picture"]


if __name__ == "__main__":
    raise SystemExit(main())
