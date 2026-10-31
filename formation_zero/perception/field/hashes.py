"""Hash-mark rows, and which whole yard every point of a frame is on (GRD-37).

Classical, CPU only, on top of `lines.detect`; with `homography.register` about 0.9 s a 1080p
sideline frame (median; 2.7 s worst of 15 timed):

1. `yard_indexings`: the detected yard lines are put on the 5-yd grid. Every three lines, with gaps
   of 1 to MAX_GAP steps between them (players hide lines), fix a 1-D projective map from where the
   lines cross mid-frame to yards; the other lines that land on a multiple of 5 join. The lines kept
   are fitted as a `pencil`: the field x of any point of the picture, from the line through it and
   the yard lines' meeting point. Relative: the leftmost line kept is x = 0.
2. `dash_candidates`: small blobs of `lines.line_pixels` once the long lines are erased.
3. `rows`: hash marks, and the 2-ft ticks just inside each sideline, are painted at every yard, so
   their dashes sit on whole yards of the pencil (multiples of 5 left out: the yard lines' own
   cross-bars) and in straight rows across the yard lines. Which row is which (far ticks, far hash,
   near hash, near ticks) is left to `homography.register`, which tells by the camera each naming
   implies.

Measured 2026-10-30 on the clear game (2026 wk1 SF at LA, 1080p), with `homography.register`:
- hash rows, by eye on 30 random sideline frames (5 per half and phase; contact sheet in GRD-37):
  both rows lie on the painted dashes on 25 of 30, none placed wrong. On 21 both were found as
  dash rows; on 4 one was (with the far ticks) and the fit placed the other. The 5 misses: 4 close-
  ups after the whistle (2-4 yard lines, few dashes) and 1 pre-snap frame over the midfield logo.
- lens distortion: yard lines 300-1,700 px long bend by under 3.5 px (median about 1; 0.3-1.0 px
  RMS off a straight line) on 5 frames, and dash residuals binned across the frame width show no
  bend (bin medians within 1.7 px). Rows and yard lines are taken as straight.
- what failed: naming rows by their spacing over the 1-yd dash spacing (the hash rows' ratio and
  the far ticks' overlap on zoomed shots; the implied camera does it, see `homography.register`);
  a free 1-D fit with a stray line let a wrong indexing through (hence MAX_MISSING); a row was
  first scored over its whole span, so a logo hiding 15 yd of it lost the row (hence `_window`).
- end zone: hash marks are bars (ENDZONE_DASH) and Hough takes a column of them for a line, so
  only yard lines and boundaries are erased there.
"""

from __future__ import annotations

import itertools

import numpy as np

from formation_zero.perception.field import lines as L

DASH_ERASE_PX = 7           # yard and side lines are erased this wide before looking for dashes
DASH_AREA = (2, 250)        # px; a dash is a small blob (2 x 1 px on the widest shots, ~12 x 6 zoomed in)
DASH_MAX_SIDE = 30          # px; longer blobs are numbers, logos or players ...
ENDZONE_DASH = (90, 600)    # ... but from the end zone a hash mark is a bar, up to ~45 x 3 px: longest side, area
DUP_PX = 6.0                # yard lines crossing mid-frame this close together are one painted line
LINE_TOL_YD = 0.6           # a yard line belongs to an indexing if it lands this close to a multiple of 5 yd
MAX_GAP = 3                 # indexing tries yard-line gaps of up to this many 5-yd steps (missed lines) ...
MAX_MISSING = 2             # ... with at most this many lines missing in all, each costing half a line
PENCIL_TOL_YD = 0.4         # the fitted pencil must put every yard-line end this close to its yard line
WHOLE_YARD_TOL = 0.15       # a dash centre lies this close to a whole yard (the paint is 4 in wide)
ROW_TOL_PX = 2.5            # dash centres this close to a row's line are on it
ROW_MIN_DEG = 25.0          # a row crosses the yard lines at least this steeply
MIN_DASHES = 5              # a row needs dashes at this many whole yards ...
MIN_FILL = 0.5              # ... and at this share of the whole yards it spans (players hide the rest)
ROW_TRIES = 20              # candidate lines tried for each row before giving up
MAX_ROWS = 4                # far ticks, two hash rows, near ticks
MAX_INDEXINGS = 12          # `find` tries at most this many ways to index the yard lines ...
INDEX_LINE_DASHES = 5       # ... and counts each yard line an indexing keeps as this many dashes


def _homog(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points, np.ones(len(points))])


# ---------------------------------------------------------------- the yard-line pencil (numpy only)


def _crossings(lines: np.ndarray, mid: float, vertical: bool = False) -> np.ndarray:
    """Where each line (x1 y1 x2 y2, extended) crosses the horizontal y = mid: its x; or with
    `vertical`, the vertical x = mid: its y."""
    hl = np.cross(_homog(lines[:, :2]), _homog(lines[:, 2:]))
    c = np.cross(hl, [1.0, 0.0, -mid] if vertical else [0.0, 1.0, -mid])
    with np.errstate(divide="ignore", invalid="ignore"):
        return c[:, 1 if vertical else 0] / c[:, 2]


def _fit_1d(s: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Least-squares 1-D homography x = (a s + b) / (c s + 1); None if degenerate."""
    a = np.column_stack([s, np.ones_like(s), -s * x])
    p, *_ = np.linalg.lstsq(a, x, rcond=None)
    return p if np.isfinite(p).all() else None


def _apply_1d(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (p[0] * s + p[1]) / (p[2] * s + 1)


def _normaliser(points: np.ndarray) -> np.ndarray:
    """Similarity taking points to centroid 0, mean distance sqrt 2 (Hartley)."""
    c = points.mean(0)
    d = np.hypot(*(points - c).T).mean()
    k = np.sqrt(2) / max(d, 1e-9)
    return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1.0]])


def pencil(yard: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The yard-line pencil: q (6,) such that a point p of the field plane is on field line
    x(p) = (q[:3] . p~) / (q[3:] . p~) yd. Fitted to the ends and midpoints of the yard lines `yard`
    (m, 4), which are at field x `x` (m,), m >= 3.

    The yard lines all meet at the image of the point at infinity across the field, and a point's
    ray from there says which line x = const it is on: 2 numbers for that point plus 3 for the
    projective map from rays to yards. Algebraic least squares on normalised points.
    """
    pts, xs = [], []
    for (x1, y1, x2, y2), xv in zip(yard, x):
        for t in (0.0, 0.5, 1.0):
            pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            xs.append(xv)
    pts, xs = np.asarray(pts, float), np.asarray(xs, float)
    t = _normaliser(pts)
    ph = _homog(pts) @ t.T
    _, _, vt = np.linalg.svd(np.hstack([ph, -xs[:, None] * ph]))
    q = vt[-1]
    return np.concatenate([q[:3] @ t, q[3:] @ t])


def field_x(q: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Field x (yd, in the pencil's own frame) of image points (n, 2)."""
    ph = _homog(np.asarray(points, float).reshape(-1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ph @ q[:3]) / (ph @ q[3:])


def pencil_line(q: np.ndarray, x: float) -> np.ndarray:
    """Homogeneous image line of field line x = const."""
    return q[:3] - x * q[3:]


def yard_indexings(yard: np.ndarray, mid: float, vertical: bool = False) -> list[dict]:
    """Ways to put the detected yard lines on the 5-yd grid, best first.

    Lines are ordered by where they cross the horizontal y = `mid` (left to right: sideline frames)
    or, with `vertical`, the vertical x = `mid` (top to bottom: end-zone frames); near duplicates are
    dropped. Every three lines and every gap between them of 1 to MAX_GAP steps (a line can be
    missing behind players) fix a 1-D projective map from crossing to yards; the other lines that
    land within LINE_TOL_YD of a multiple of 5 join. An indexing scores its lines less half a line
    for each one missing between them (at most MAX_MISSING); kept: those within one line of the
    best, each refitted as a `pencil`. Each is {"keep": indices into `yard`, "x": field x
    (yd, the first kept line at 0), "q": pencil}.
    """
    m = len(yard)
    if m < 3:
        return []
    s = _crossings(yard, mid, vertical)
    ok = np.isfinite(s)
    order = [i for i in np.argsort(s) if ok[i]]
    lengths = np.hypot(yard[:, 2] - yard[:, 0], yard[:, 3] - yard[:, 1])
    keep: list[int] = []
    for i in order:
        if keep and abs(s[i] - s[keep[-1]]) < DUP_PX:
            if lengths[i] > lengths[keep[-1]]:
                keep[-1] = i
            continue
        keep.append(i)
    if len(keep) < 3:
        return []
    idx = np.array(keep)
    sv = s[idx]
    sn = (sv - sv.mean()) / max(sv.std(), 1e-9)
    found: dict[tuple, int] = {}
    for i, j, k in itertools.combinations(range(len(idx)), 3):
        for a, b in itertools.product(range(1, MAX_GAP + 1), repeat=2):
            p = _fit_1d(sn[[i, j, k]], np.array([0.0, a, a + b]))
            if p is None:
                continue
            n = _apply_1d(p, sn)
            if not np.all(np.isfinite(n)) or np.any(np.diff(n) <= 0):
                continue
            r = np.abs(n - np.round(n)) * 5
            best: dict[int, tuple[float, int]] = {}
            for t in np.flatnonzero(r < LINE_TOL_YD):
                key = round(n[t])
                if key not in best or r[t] < best[key][0]:
                    best[key] = (r[t], t)
            keys = sorted(best)
            steps = np.diff(keys)
            if len(keys) < 3 or steps.max() > MAX_GAP or (steps - 1).sum() > MAX_MISSING:
                continue
            sig = tuple(sorted((int(best[key][1]), key - keys[0]) for key in keys))
            found[sig] = len(sig) - 0.5 * float((steps - 1).sum())
    if not found:
        return []
    top = max(found.values())
    out = []
    for sig in sorted(found, key=lambda g: -found[g]):
        if found[sig] < top - 1:
            break
        rows = np.array([t for t, _ in sig])
        x = np.array([5.0 * n for _, n in sig])
        q = pencil(yard[idx[rows]], x)
        ends = np.concatenate([field_x(q, yard[idx[rows], :2]) - x, field_x(q, yard[idx[rows], 2:]) - x])
        if np.all(np.isfinite(ends)) and np.abs(ends).max() <= PENCIL_TOL_YD:
            out.append({"keep": idx[rows], "x": x, "q": q})
    return out


# ---------------------------------------------------------------- dashes and rows


def dash_candidates(pix: np.ndarray, erase: np.ndarray, max_side: float = DASH_MAX_SIDE,
                    max_area: float = DASH_AREA[1]) -> np.ndarray:
    """(n, 2) centres of small painted blobs in a line-pixel mask, once the long lines `erase`
    (x1 y1 x2 y2) are painted out."""
    cv2 = L._cv2()
    gone = np.zeros(pix.shape, np.uint8)
    for x1, y1, x2, y2 in erase:
        cv2.line(gone, (round(x1), round(y1)), (round(x2), round(y2)), 1, DASH_ERASE_PX)
    _, _, stats, centres = cv2.connectedComponentsWithStats(pix.astype(np.uint8) & (1 - gone), 8)
    stats, centres = stats[1:], centres[1:]
    a = stats[:, 4]
    ok = (a >= DASH_AREA[0]) & (a <= max_area) & (np.maximum(stats[:, 2], stats[:, 3]) <= max_side)
    return centres[ok]


def _line_through(points: np.ndarray) -> np.ndarray:
    """Total least squares line through points, as (a, b, c) with a^2 + b^2 = 1."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c)
    n = np.array([-vt[0][1], vt[0][0]])
    return np.array([n[0], n[1], -n @ c])


def _meet(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    return p[:2] / p[2] if abs(p[2]) > 1e-12 else np.array([np.nan, np.nan])


def _window(xs: np.ndarray) -> tuple[int, int] | None:
    """The run of a row's dashes (sorted whole yards `xs`) to keep: the most dashes, at least
    MIN_DASHES, over a stretch of whole yards (multiples of 5 left out) that they fill to MIN_FILL.
    A logo or a crowd of players can hide 15 yd of a row, and a stray dash on its line beyond them
    must not stretch the row over the gap. (first, last) index into xs, or None."""
    best, key = None, None
    for lo in range(len(xs)):
        for hi in range(lo + MIN_DASHES - 1, len(xs)):
            want = sum(1 for x in np.arange(xs[lo], xs[hi] + 1) if x % 5)
            have = hi - lo + 1
            if have >= MIN_FILL * want and (key is None or (have, have / want) > key):
                best, key = (lo, hi), (have, have / want)
    return best


def rows(points: np.ndarray, q: np.ndarray) -> list[dict]:
    """Rows of dashes on whole yards: lines through dash candidates `points` whose field x (by the
    pencil q) is a whole yard, crossing the yard lines at ROW_MIN_DEG or more, with dashes at
    MIN_DASHES whole yards and MIN_FILL of the yards they span (see `_window`). Greedy: for each
    row, the lines through pairs of such dashes are tried in order of dashes near them (at most
    ROW_TRIES), and the first with such a run is refitted to it and taken out.

    Each is {"line": (a, b, c), "points": (k, 2) one dash per yard, "x": (k,) whole yards, "fill"}.
    """
    if len(points) == 0:
        return []
    xs = field_x(q, points)
    xi = np.round(xs)
    on = np.isfinite(xs) & (np.abs(xs - xi) < WHOLE_YARD_TOL) & (np.mod(xi, 5) != 0)
    cand = np.flatnonzero(on)
    alive = np.ones(len(points), bool)
    out: list[dict] = []
    cos_max = np.cos(np.radians(ROW_MIN_DEG))
    while len(out) < MAX_ROWS:
        c = cand[alive[cand]]
        if len(c) < MIN_DASHES:
            break
        pts, xv = points[c], xi[c]
        i, j = np.triu_indices(len(c), 1)
        sel = xv[i] != xv[j]
        i, j = i[sel], j[sel]
        if len(i) == 0:
            break
        d = pts[j] - pts[i]
        u = d / np.hypot(d[:, 0], d[:, 1])[:, None]
        pl = np.array([pencil_line(q, x) for x in (xv[i] + xv[j]) / 2])
        pd = np.column_stack([pl[:, 1], -pl[:, 0]])
        pd /= np.maximum(np.hypot(pd[:, 0], pd[:, 1]), 1e-12)[:, None]
        sel = np.abs((u * pd).sum(1)) < cos_max
        i, u = i[sel], u[sel]
        if len(i) == 0:
            break
        n = np.column_stack([-u[:, 1], u[:, 0]])
        dist = np.abs((pts[None, :, 0] - pts[i][:, None, 0]) * n[:, :1]
                      + (pts[None, :, 1] - pts[i][:, None, 1]) * n[:, 1:])
        inl = dist < ROW_TOL_PX
        score = np.array([len(np.unique(xv[r])) for r in inl])
        tried = 0
        accepted = None
        for b in np.argsort(-score, kind="stable"):
            if score[b] < MIN_DASHES or tried >= ROW_TRIES:
                break
            tried += 1
            line = _line_through(pts[inl[b]])
            near = np.abs(_homog(pts) @ line) < ROW_TOL_PX
            if near.sum() < 2:
                continue
            line = _line_through(pts[near])
            d2 = np.abs(_homog(pts) @ line)
            near = d2 < ROW_TOL_PX
            px, xx = [], []
            for x in np.unique(xv[near]):
                k = np.flatnonzero(near & (xv == x))
                k = k[np.argmin(d2[k])]
                px.append(pts[k])
                xx.append(x)
            win = _window(np.asarray(xx))
            if win is None:
                continue
            px, xx = np.asarray(px)[win[0]:win[1] + 1], np.asarray(xx)[win[0]:win[1] + 1]
            want = sum(1 for x in np.arange(xx[0], xx[-1] + 1) if x % 5)
            accepted = {"line": _line_through(px), "points": px, "x": xx, "fill": len(xx) / want}
            break
        if accepted is None:
            break
        out.append(accepted)
        alive[c[np.abs(_homog(pts) @ accepted["line"]) < 2 * ROW_TOL_PX]] = False
    return out


def _row_at(line: np.ndarray, q: np.ndarray, x: float) -> np.ndarray:
    return _meet(line, pencil_line(q, x))


def _inside(row: dict, bounds: list[np.ndarray], centre: np.ndarray, q: np.ndarray) -> bool:
    """False if the row's middle lies beyond a detected boundary line (benches, crowd, signage)."""
    p = _row_at(row["line"], q, float(np.median(row["x"])))
    for b in bounds:
        nf = np.cross([b[0], b[1], 1.0], [b[2], b[3], 1.0])
        if (nf @ [p[0], p[1], 1.0]) * (nf @ [centre[0], centre[1], 1.0]) < 0:
            return False
    return True


def find(image: np.ndarray, angle: str = "sideline", det: dict | None = None) -> dict:
    """Yard lines on the 5-yd grid and the rows of dashes of one RGB frame, in its own pixels.

    {"det": `lines.detect` output, "dashes": (n, 2) all dash candidates, "alternatives": list, best
     first, of {"yard": (m, 4) yard lines kept, "x": (m,) their field x (yd; the leftmost at 0, so
     right up to a whole number of 5-yd steps; end zone: the topmost), "q": pencil, "rows": rows of
     dashes on whole yards (see `rows`) from the top of the picture down (end zone: left to
     right)}, and the best alternative's keys at the top level ("q" None and no rows without three
     usable yard lines)}. Which rows are which (hash or sideline ticks) is left to
     `homography.register`. The camera angle is the one `lines.detect` reads from the lines.

    An indexing scores its rows' dashes (less the yards they skip) plus INDEX_LINE_DASHES a yard line.
    """
    cv2 = L._cv2()
    det = det if det is not None else L.detect(image, angle)
    h, w = image.shape[:2]
    s = L.PROC_WIDTH / w
    small = image if w == L.PROC_WIDTH else cv2.resize(image, (L.PROC_WIDTH, round(h * s)),
                                                       interpolation=cv2.INTER_AREA)
    pix = L.line_pixels(small)
    yard_w = det["yard"] * s
    endzone = det["angle"] == "endzone"
    # from the end zone a column of dashes is close-packed in depth and Hough takes it for a line,
    # so only the yard lines and the boundaries are erased there
    other = [b for b in det["boundary"].values() if b is not None] if endzone else list(det["side"])
    erase = np.vstack([yard_w, *[np.reshape(b, (1, 4)) * s for b in other]]).reshape(-1, 4)
    pts = dash_candidates(pix, erase, *(ENDZONE_DASH if endzone else (DASH_MAX_SIDE, DASH_AREA[1])))
    out = {"det": det, "dashes": pts / s, "alternatives": [], "yard": np.zeros((0, 4)), "x": np.zeros(0),
           "q": None, "rows": []}
    centre = ((yard_w[:, :2] + yard_w[:, 2:]) / 2).mean(0) if len(yard_w) else None
    bounds = [b * s for b in det["boundary"].values() if b is not None]
    back = np.array([s, s, 1.0])
    scored = []
    mid = (small.shape[1] if endzone else small.shape[0]) / 2
    for ix in yard_indexings(yard_w, mid, endzone)[:MAX_INDEXINGS]:
        found = [r for r in rows(pts, ix["q"]) if _inside(r, bounds, centre, ix["q"])]
        score = sum(len(r["x"]) * (2 * r["fill"] - 1) for r in found) + INDEX_LINE_DASHES * len(ix["x"])
        q = np.concatenate([ix["q"][:3] * back, ix["q"][3:] * back])
        scaled = [{**r, "line": r["line"] * back, "points": r["points"] / s} for r in found]
        scaled.sort(key=lambda r: _row_at(r["line"], q, float(np.median(r["x"])))[0 if endzone else 1])
        scored.append((score, {"yard": det["yard"][ix["keep"]], "x": ix["x"], "q": q, "rows": scaled}))
    scored.sort(key=lambda t: -t[0])
    out["alternatives"] = [a for _, a in scored]
    if scored:
        out.update(scored[0][1])
    return out


__all__ = ["dash_candidates", "field_x", "find", "pencil", "pencil_line", "rows", "yard_indexings"]
