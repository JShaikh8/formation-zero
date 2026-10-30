"""Painted field lines from one frame: yard lines, sidelines, and the field mask (GRD-37).

Classical, CPU only, about 0.2 s a frame (median; 0.7 s worst) at the 960 px working width:

1. `line_pixels`: white top-hat (clean on grass, keeps the wide sideline band) united with the
   frame's strongest bright ridges (a Hessian test that adapts to contrast, for the snow).
2. `segments`: probabilistic Hough.
3. `families`: steep and flat segments are each tried as the yard-line family: RANSAC vanishing
   point, greedy collinear grouping, and a test that each line looks painted rather than drawn
   through a crowd or logo (`coverage`). The camera angle says which to expect; the picture can
   overrule it (it did on three clear-game frames that sit on the wrong side of a
   take boundary the play index misplaced).
4. `boundaries`: among the other lines, the ones where the painted yard lines stop.
5. `field_mask`: the half-planes inside the boundaries found, or None without two yard lines.

Measured by eye on 120 sampled frames (15 per game, angle and half; contact sheets in GRD-37):
- clear game, end zone: yard lines right 30/30; sidelines are rarely in frame (1 of 3 found).
- clear game, sideline: yard lines right 25/30 (3 not found, 2 wrong on close-ups); far sideline
  found on 17 of 22 frames that show it; near sideline 2 of 6. All 3 frames labelled sideline
  that are really end-zone shots were read as end zone.
- snow game, sideline: yard lines right 22/30 (6 not found), far sideline only 2 of 6.
- snow game, end zone: 9/15 in the first half, about 1/15 in the second: faint horizontal lines
  broken by players. The weak case.
Against the teacher's people: on the 12 most crowded clear-game frames (kickoffs, punts) the mask
removes 16 of 52 people a frame, nearly all bench, staff and chain crew; on the snow game's it
removes almost none (no answer). Of 274 removals checked by eye, 20 were players on the field, all
on 3 frames with a false boundary (a dissolve between shots, two close-ups).
"""

from __future__ import annotations

import numpy as np

PROC_WIDTH = 960            # everything runs at this width; thresholds below are in its pixels
TOPHAT_PX = 31              # wider than a yard line or the near sideline band, so they survive the top-hat
TOPHAT_MIN = 10             # grey levels a line must stand above the local turf
TOPHAT_NOISE_K = 4.0        # ... or this many times the frame's median top-hat response
SAT_MAX = 90                # painted lines are white; goalposts, logos and jerseys are coloured
RIDGE_SIGMA = 2.5           # Hessian scale for the ridge test: a far yard line is 3-5 px wide
RIDGE_TOP = 0.07            # the ridge test keeps this fraction of the frame's pixels ...
RIDGE_MIN = 2.0             # ... if their ridge strength is at least this (a flat frame keeps none)
HOUGH_VOTES = 40
HOUGH_MIN_LEN = 40
HOUGH_GAP = 8
VP_TOL_PX = 2.0             # a segment agrees with a vanishing point if its ends are this close to the ray
VP_ITERS = 400
COLLINEAR_PX = 4.0          # segment ends this close to a line belong to it (the sideline band is ~10 px)
MIN_LINE_PX = 90            # a painted line needs this much support
MIN_COVERAGE = 0.5          # ... and look painted along its length (see `coverage`)
YARD_OFFSETS = (9.0,)       # where `coverage` looks beside a yard line ...
SIDE_OFFSETS = (6.0, 12.0, 20.0)      # ... and beside a boundary candidate, up to the wide near band
YARD_MIN_SPAN = 0.3         # a yard line is at least this fraction of the frame long (see `families`)
MIN_ENDS = 2                # a boundary needs at least this many yard lines ending on it
MIN_YARD_LINES = 2          # fewer than this and the frame has no yard-line family (and no mask)
OVERRIDE_LINES = 7          # the other family needs this many lines to overrule the camera angle
MAX_SEGMENTS = 400          # `collinear` keeps the longest this many segments
BORDER_PX = 3               # ignore line pixels this close to the frame edge (encoder artefacts)
INSIDE_PX, END_SHORT_PX = 90, 8       # `boundaries` looks for yard-line paint this far inside a candidate
END_OVER_PX, OUTSIDE_PX = 30, 60      # ... and this far outside it, past the band
PAINTED, UNPAINTED = 0.4, 0.25        # `coverage` scores that say painted / not painted
MERGE_PX = 6                # lines this close along their length are one painted line
MASK_MARGIN_PX = 10         # keep feet this far outside the boundary (officials stand on the border)
SPLIT_DEG = 15.0            # steeper than this from horizontal: sideline-camera yard lines, end-zone sidelines


def _cv2():
    import cv2

    cv2.setNumThreads(1)
    return cv2


def line_pixels(image: np.ndarray) -> np.ndarray:
    """Bool mask of painted-line pixels: the union of two tests, both on unsaturated pixels.

    - bright: a white top-hat at least TOPHAT_MIN grey levels (or TOPHAT_NOISE_K times the frame's
      median response) above the turf around it. Clean on grass, and keeps the wide sideline band.
    - ridge: the RIDGE_TOP brightest-ridge pixels of the frame (Hessian at RIDGE_SIGMA). Adapts to
      contrast, so it finds the faint lines on a snow-covered field that the first test misses.
    """
    cv2 = _cv2()
    s = image.shape[1] / PROC_WIDTH
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    k = max(3, round(TOPHAT_PX * s) | 1)
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    sat = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1]
    thr = max(TOPHAT_MIN, TOPHAT_NOISE_K * float(np.median(top)))
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), RIDGE_SIGMA * s)
    dxx, dyy = cv2.Sobel(g, cv2.CV_32F, 2, 0), cv2.Sobel(g, cv2.CV_32F, 0, 2)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1)
    ridge = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2) - (dxx + dyy)     # -2 x the most negative eigenvalue
    ridge_thr = max(float(np.quantile(ridge, 1 - RIDGE_TOP)), RIDGE_MIN)
    out = ((top > thr) | (ridge > ridge_thr)) & (sat < SAT_MAX)
    b = max(1, round(BORDER_PX * s))
    out[:b], out[-b:], out[:, :b], out[:, -b:] = False, False, False, False
    return out


def segments(mask: np.ndarray) -> np.ndarray:
    """(n, 4) x1 y1 x2 y2 straight segments in a line-pixel mask (probabilistic Hough)."""
    cv2 = _cv2()
    s = mask.shape[1] / PROC_WIDTH
    out = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 360, int(HOUGH_VOTES * s),
                          minLineLength=HOUGH_MIN_LEN * s, maxLineGap=HOUGH_GAP * s)
    return np.zeros((0, 4)) if out is None else out.reshape(-1, 4).astype(float)


# ---------------------------------------------------------------- geometry (numpy only)


def _angles(segs: np.ndarray) -> np.ndarray:
    """Degrees from horizontal, 0..90, ignoring direction."""
    d = segs[:, 2:] - segs[:, :2]
    return np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))


def _lengths(segs: np.ndarray) -> np.ndarray:
    return np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])


def _homog(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(len(points))])


def _vp_error(segs: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pixel distance of each segment's ends from the ray joining its midpoint to vanishing point v."""
    p1, p2 = _homog(segs[:, :2]), _homog(segs[:, 2:])
    m = (p1 + p2) / 2
    ray = np.cross(m, v)
    norm = np.hypot(ray[:, 0], ray[:, 1])
    ray = ray / np.maximum(norm, 1e-12)[:, None]
    return np.maximum(np.abs((ray * p1).sum(1)), np.abs((ray * p2).sum(1)))


def vanishing_point(segs: np.ndarray, tol: float = VP_TOL_PX, iters: int = VP_ITERS,
                    seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC vanishing point (homogeneous, unit norm; third coordinate 0 means at infinity) and inliers.

    Pairs are drawn weighted by length; the point with the most inlier length wins.
    """
    n = len(segs)
    if n < 2:
        return None, np.zeros(n, bool)
    lengths = _lengths(segs)
    lines = np.cross(_homog(segs[:, :2]), _homog(segs[:, 2:]))
    rng = np.random.default_rng(seed)
    p = lengths / lengths.sum()
    best_v, best_score, best_in = None, -1.0, np.zeros(n, bool)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False, p=p)
        v = np.cross(lines[i], lines[j])
        if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
            continue
        v = v / np.linalg.norm(v)
        inl = _vp_error(segs, v) < tol
        score = lengths[inl].sum()
        if score > best_score:
            best_v, best_score, best_in = v, score, inl
    return best_v, best_in


def _fit_line(segs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Length-weighted total least squares through segment ends: (point, unit direction)."""
    pts = np.vstack([segs[:, :2], segs[:, 2:]])
    w = np.tile(_lengths(segs), 2)
    c = (pts * w[:, None]).sum(0) / w.sum()
    cov = ((pts - c) * w[:, None]).T @ (pts - c)
    return c, np.linalg.eigh(cov)[1][:, -1]


def _normal_form(segs: np.ndarray) -> np.ndarray:
    """Homogeneous lines scaled so that line . (x, y, 1) is a signed pixel distance."""
    lines = np.cross(_homog(segs[:, :2]), _homog(segs[:, 2:]))
    return lines / np.maximum(np.hypot(lines[:, 0], lines[:, 1]), 1e-12)[:, None]


def collinear(segs: np.ndarray, tol: float = COLLINEAR_PX, min_px: float = MIN_LINE_PX) -> np.ndarray:
    """Greedy collinear grouping: (m, 4) x1 y1 x2 y2 lines over each group's support, strongest first.

    The segment whose line gathers the most segment length (both ends within `tol`) seeds a group,
    the group is refitted and taken out, and so on while groups carry `min_px`. Only the
    MAX_SEGMENTS longest segments take part.
    """
    out = []
    if len(segs) > MAX_SEGMENTS:
        segs = segs[np.argsort(-_lengths(segs))[:MAX_SEGMENTS]]
    if len(segs) == 0:
        return np.zeros((0, 4))
    lengths = _lengths(segs)
    p1, p2 = _homog(segs[:, :2]), _homog(segs[:, 2:])
    lines = _normal_form(segs)
    near = np.maximum(np.abs(lines @ p1.T), np.abs(lines @ p2.T)) < tol
    alive = np.ones(len(segs), bool)
    while alive.any():
        support = (near & alive) @ lengths
        support[~alive] = 0
        i = int(np.argmax(support))
        if support[i] < min_px:
            break
        q, u = _fit_line(segs[near[i] & alive])
        n = np.array([-u[1], u[0]])
        inl = alive & (np.maximum(np.abs((segs[:, :2] - q) @ n), np.abs((segs[:, 2:] - q) @ n)) < tol)
        if not inl.any():
            inl = near[i] & alive
        g = segs[inl]
        proj = np.concatenate([(g[:, :2] - q) @ u, (g[:, 2:] - q) @ u])
        a, b = q + proj.min() * u, q + proj.max() * u
        if np.hypot(*(b - a)) >= min_px:
            out.append([a[0], a[1], b[0], b[1]])
        alive &= ~inl
    return _merge(np.asarray(out, float).reshape(-1, 4), tol=MERGE_PX)


def _merge(lines: np.ndarray, tol: float) -> np.ndarray:
    """Fold a line into a longer one when both its ends lie within `tol` of the longer one's line
    (the two edges of one thick painted line come out of `collinear` as two lines)."""
    lines = lines[np.argsort(-_lengths(lines))] if len(lines) else lines
    kept: list[np.ndarray] = []
    for ln in lines:
        for k, big in enumerate(kept):
            nf = _normal_form(big[None])[0]
            if max(abs(nf @ [ln[0], ln[1], 1]), abs(nf @ [ln[2], ln[3], 1])) < tol:
                u = (big[2:] - big[:2]) / max(np.hypot(*(big[2:] - big[:2])), 1e-9)
                proj = [(p - big[:2]) @ u for p in (big[:2], big[2:], ln[:2], ln[2:])]
                kept[k] = np.concatenate([big[:2] + min(proj) * u, big[:2] + max(proj) * u])
                break
        else:
            kept.append(ln.astype(float))
    return np.asarray(kept, float).reshape(-1, 4)


def _paint(fat: np.ndarray, xs: np.ndarray, ys: np.ndarray, nx: float, ny: float,
           offsets: tuple[float, ...]) -> float:
    """Painted-ness of the points (xs, ys) on a line with unit normal (nx, ny); see `coverage`."""
    h, w = fat.shape
    keep = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    if keep.sum() < 3:
        return np.nan
    xs, ys = xs[keep], ys[keep]

    def frac(d):
        xi = np.clip(np.round(xs + d * nx).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys + d * ny).astype(int), 0, h - 1)
        return fat[yi, xi].mean()

    on = frac(0.0)
    return max(on - min(frac(d), frac(-d)) for d in offsets)


def _fat(mask: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def coverage(lines: np.ndarray, mask: np.ndarray, offsets: tuple[float, ...] = YARD_OFFSETS,
             step: float = 2.0) -> np.ndarray:
    """How much each line looks painted: the fraction of points along it on line pixels, less the
    fraction on line pixels a few pixels to one side or the other, whichever side is emptier (at
    the best of `offsets`, in processing pixels).

    A painted line on turf scores near 1; a line drawn through a crowd or a logo, where the mask is
    dense on both sides, scores near 0. Taking the emptier side keeps a line along the edge of the
    wide sideline band. Yard lines are tested at one offset; boundary candidates at several, since
    the near band can be 20 px wide.
    """
    fat = _fat(mask)
    k = mask.shape[1] / PROC_WIDTH
    offsets = tuple(d * k for d in offsets)
    out = np.zeros(len(lines))
    for i, (x1, y1, x2, y2) in enumerate(lines):
        length = np.hypot(x2 - x1, y2 - y1)
        n = max(2, int(length / step))
        nx, ny = -(y2 - y1) / max(length, 1e-9), (x2 - x1) / max(length, 1e-9)
        out[i] = np.nan_to_num(_paint(fat, np.linspace(x1, x2, n), np.linspace(y1, y2, n), nx, ny, offsets))
    return out


def families(segs: np.ndarray, angle: str | None = None, mask: np.ndarray | None = None) -> dict:
    """Split segments into the yard-line family and the sideline family, and merge each into lines.

    Steep and flat segments are each tried as the yard family (lines with a common vanishing point).
    The camera angle `angle` says which to expect; the other is taken only when the expected one
    has fewer than MIN_YARD_LINES lines and the other has OVERRIDE_LINES (the angle label is wrong
    on a few frames near cuts). With `mask`, lines that do not look painted along their length (see
    `coverage`: lines drawn through players, numbers, crowd) are dropped.

    Returns {"yard", "side": (m, 4) lines, "vanishing_point": (3,) homogeneous or None, "angle"}.
    """
    ang = _angles(segs) if len(segs) else np.zeros(0)
    split = {"sideline": ang > SPLIT_DEG, "endzone": ang <= SPLIT_DEG}
    cand = {}
    for view, m in split.items():
        fam = segs[m]
        v, inl = vanishing_point(fam)
        yard = collinear(fam[inl]) if v is not None else np.zeros((0, 4))
        in_yard = np.zeros(len(segs), bool)
        in_yard[np.flatnonzero(m)[inl]] = True
        side = collinear(segs[~in_yard])
        if mask is not None:
            yard = yard[coverage(yard, mask) >= MIN_COVERAGE]
            side = side[coverage(side, mask, SIDE_OFFSETS) >= MIN_COVERAGE]
            # yard lines cross much of the frame: most of its height seen from the sideline,
            # most of its width from the end zone
            span = mask.shape[0] if view == "sideline" else mask.shape[1]
            yard = yard[_lengths(yard) >= YARD_MIN_SPAN * span]
        cand[view] = {"yard": yard, "side": side, "vanishing_point": v, "angle": view}
    hint = angle or max(cand, key=lambda v: len(cand[v]["yard"]))
    other = "endzone" if hint == "sideline" else "sideline"
    if len(cand[hint]["yard"]) < MIN_YARD_LINES and len(cand[other]["yard"]) >= OVERRIDE_LINES:
        return cand[other]
    return cand[hint]


# ---------------------------------------------------------------- boundaries and the mask


def _beside(yard: np.ndarray, line: np.ndarray, outward: np.ndarray, fat: np.ndarray, k: float) -> np.ndarray:
    """For each yard line, painted-ness just inside `line` and just outside it, along the yard line.

    (n, 2) array; NaN where the stretch is out of the frame or the yard line is parallel to `line`.
    """
    out = np.full((len(yard), 2), np.nan)
    nf = _normal_form(line[None])[0]
    for i, y in enumerate(yard):
        a, b = y[:2], y[2:]
        u = (b - a) / max(np.hypot(*(b - a)), 1e-9)
        if u @ outward < 0:
            u = -u
        rate = nf[:2] @ u
        if abs(rate) < 0.05:                    # nearly parallel: this line cannot bound that one
            continue
        x = a - (nf @ [a[0], a[1], 1]) / rate * u      # where the yard line meets `line`
        for j, (t0, t1) in enumerate(((-INSIDE_PX, -END_SHORT_PX), (END_OVER_PX, END_OVER_PX + OUTSIDE_PX))):
            t = np.linspace(t0 * k, t1 * k, 24)
            out[i, j] = _paint(fat, x[0] + t * u[0], x[1] + t * u[1], -u[1], u[0], (YARD_OFFSETS[0] * k,))
    return out


def boundaries(fam: dict, mask: np.ndarray) -> dict:
    """The boundary lines: non-yard lines where the painted yard lines stop.

    Sideline camera: "far" (beyond the yard lines' upper ends) and "near" (below). End-zone camera:
    "left" and "right". Along each yard line, a candidate looks at the paint just inside it and just
    outside (past END_OVER_PX: the sideline band is wide and the yard lines run into it). A yard
    line painted inside and not outside ends there; one painted outside runs past. A candidate
    scores ends less runs-past; ties go to the outer line. Each entry is a line or None.
    """
    yard, side, angle = fam["yard"], fam["side"], fam["angle"]
    names = ("far", "near") if angle == "sideline" else ("left", "right")
    best = dict.fromkeys(names)
    if len(yard) < MIN_YARD_LINES or len(side) == 0:
        return best
    fat = _fat(mask)
    centre = ((yard[:, :2] + yard[:, 2:]) / 2).mean(0)
    k = mask.shape[1] / PROC_WIDTH
    for line in side:
        lm = (line[:2] + line[2:]) / 2
        outward = lm - centre
        if angle == "sideline":
            key = names[0] if outward[1] < 0 else names[1]
        else:
            key = names[0] if outward[0] < 0 else names[1]
        inside, outside = _beside(yard, line, outward, fat, k).T
        ends = int(((inside >= PAINTED) & (outside < UNPAINTED)).sum())
        past = int((outside >= PAINTED).sum())
        score = (ends - past, float(np.linalg.norm(outward)))
        if ends >= MIN_ENDS and score[0] > 0 and (best[key] is None or score > best[key][1]):
            best[key] = (line, score)
    return {k: (None if v is None else v[0]) for k, v in best.items()}


def _half_plane(line: np.ndarray, inside: np.ndarray, shape: tuple[int, int], margin: float) -> np.ndarray:
    h, w = shape
    a, b = line[:2], line[2:]
    n = np.array([-(b - a)[1], (b - a)[0]])
    n = n / max(np.hypot(*n), 1e-9)
    if (inside - a) @ n < 0:
        n = -n
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - a[0]) * n[0] + (yy - a[1]) * n[1] >= -margin


def detect(image: np.ndarray, angle: str | None = None) -> dict:
    """Everything this module finds on one RGB frame, in the frame's own pixels.

    {"yard", "side", "segments": (m, 4) x1 y1 x2 y2, "vanishing_point": (x, y, 1), or a direction
     (dx, dy, 0) when the yard lines are parallel, or None, "angle": "sideline"/"endzone" as read
     from the lines, "boundary": {"far"/"near" or "left"/"right": line or None}, "mask": bool (h, w)
     or None}. `angle` is the camera angle the frame is labelled with (None: read it).
    """
    cv2 = _cv2()
    h, w = image.shape[:2]
    s = PROC_WIDTH / w
    small = cv2.resize(image, (PROC_WIDTH, round(h * s)), interpolation=cv2.INTER_AREA) if w != PROC_WIDTH else image
    pix = line_pixels(small)
    segs = segments(pix)
    fam = families(segs, angle, pix)
    bnd = boundaries(fam, pix)
    mask = None
    if len(fam["yard"]) >= MIN_YARD_LINES:
        inside = ((fam["yard"][:, :2] + fam["yard"][:, 2:]) / 2).mean(0)
        mask = np.ones(small.shape[:2], bool)
        for line in bnd.values():
            if line is not None:
                mask &= _half_plane(line, inside, small.shape[:2], MASK_MARGIN_PX)
        if w != PROC_WIDTH:
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    back = lambda a: None if a is None else np.asarray(a, float) / s
    vp = fam["vanishing_point"]
    if vp is not None and abs(vp[2]) > 1e-12:
        vp = np.array([vp[0] / vp[2] / s, vp[1] / vp[2] / s, 1.0])
    return {"yard": back(fam["yard"]).reshape(-1, 4), "side": back(fam["side"]).reshape(-1, 4),
            "segments": back(segs).reshape(-1, 4), "vanishing_point": vp, "angle": fam["angle"],
            "boundary": {k: back(v) for k, v in bnd.items()}, "mask": mask}


def field_mask(image: np.ndarray, angle: str | None = None) -> np.ndarray | None:
    """Bool (h, w): inside the field's visible boundary lines, MASK_MARGIN_PX beyond them so the
    officials on the border stay in. None without two yard lines (it cannot tell)."""
    return detect(image, angle)["mask"]


__all__ = ["boundaries", "collinear", "coverage", "detect", "families", "field_mask", "line_pixels",
           "segments", "vanishing_point"]
