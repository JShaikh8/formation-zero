"""Pixel-to-yards homography on sideline frames, and its absolute anchor from the line of scrimmage (GRD-37).

Field frame of nfl_field: x along the field (yd), y across it from the near sideline (the press-box
camera's side), z up.

1. `register`: from `hashes.find`, the dash rows are named in picture order from LATERAL (far ticks
   y 52.89, far hash 30.08, near hash 23.25, near ticks 0.44: dash centres, not the painted lines'
   edges) and each naming is fitted by `ransac`: point pairs are the dash centres and the yard
   lines' crossings of the rows; every hypothesis also carries all the yard lines' points, which pin
   the tilt across the field that two rows 6.8 yd apart leave loose. A naming counts only if the
   `camera` behind its homography is a plausible press-box camera and every named row keeps
   ROW_INLIERS of its points. A wrong naming fits about as well (two lines of dashes and the yard
   lines always fit a homography, at 1-3 px) but puts the camera under the field, in front of it or
   140-240 yd away. The result is right up to whole 5-yd steps along the field.
2. `line_of_scrimmage`, `anchor`: on pre-snap frames, players' feet through the relative homography,
   offense and defense by shirt colour (`team.ColourTeams`); the line is midway between the two
   fronts, and the official yardline_100, with the side the offense lines up on, fixes the steps.

Measured 2026-10-30 on the clear game (2026 wk1 SF at LA, 1080p), all 1,001 sideline samples:
- registered 737 (74%), easy 654 (65%): pre-snap 269/334, live 278/334, after the whistle 107/333;
  first half 336/501, second 318/500. Easy frames: RMS reprojection 2.31 px median (p90 2.62; the
  rule caps it at 3), yard-line points 1.99 px, 91% of point pairs kept (medians, 1080p px).
- not easy: 83 registered (78 keep under MIN_INLIERS, 5 over EASY_RMS_PX; on the contact sheet
  most of these look right, so the rule is strict); 254 with no fit, 57 with under 3 yard lines
  and the rest without two rows that name plausibly: mostly close-ups after the whistle.
- camera: on the easy frames 87.3 yd behind the near sideline (IQR 86.0-88.5) and 27.2 yd up (IQR
  27.0-27.5), panning along the field (absolute x 42-69). Taken as the prior for a second pass
  (671 easy in the first). A tick row named as a hash row moves it 20 yd or more.
- extrapolation: fitted on the two hash rows alone (6.8 yd apart), the far tick row lands at y
  52.82 (painted 52.89; IQR 52.70-52.93; 10 of 496 frames over 0.5 yd off), the near one at 0.46
  (painted 0.44; 30 frames).
- anchor: 252 pre-snap frames of 123 scrimmage plays with an easy fit; SAM 3 teacher boxes (576p,
  score over 0.5), teams by `team.ColourTeams` fitted on 4,385 crops from 200 frames. The players'
  line lands 0.19 yd from the official one once shifted (median; max 1.02, so all `ok`); 233
  frames have nobody on the wrong side, 18 one. The offense's side gives the direction: the
  49ers attack +x on every Q1 and Q3 frame and -x on every Q2 and Q4 one (65, 69, 56, 62), the
  ends switching each quarter under a fixed camera. Against the painted yard numbers and their
  arrows, by eye: 69 of 69 anchored frames agree (two random samples of 36, 3 shared). One
  frame's shift was a step off (2.3 yd) until `feet` left out the near bench, cut off by the
  bottom of the frame and projected 3 yd onto the field; it now lands 0.4 yd off.
- end zone, 60 random frames, relative only: 41 easy (RMS 2.36 px median; pre-snap 23/28, live
  16/21, after the whistle 2/11), the camera 38.3 yd up (35.0-42.5). By eye 8 of the first 12
  registered, all right. Without the height prior 3 of the 41 took the two hash columns for a
  hash row and its sideline ticks (camera 51-55 yd up).
- speed: 0.9 s a frame median with `lines.detect` (2.7 s worst of 15), CPU, one thread.
"""

from __future__ import annotations

import itertools

import numpy as np

from formation_zero.configs.nfl_field import FIELD_WIDTH_YD, HASH_FROM_SIDELINE_YD

# A hash mark is 2 ft long and its inner end is 70 ft 9 in from its sideline, so its middle is a foot
# further out: dash centres sit 1/3 yd outside the hash lines of nfl_field.
HASH_NEAR_Y = HASH_FROM_SIDELINE_YD - 1 / 3            # 23.25
HASH_FAR_Y = FIELD_WIDTH_YD - HASH_FROM_SIDELINE_YD + 1 / 3      # 30.08
# The 2-ft ticks at every yard along each sideline start 4 in inside it: centres 16 in inside.
# Both checked by extrapolation: fitted on the hash rows alone, the far tick rows land at y 52.82
# (painted 52.89, 496 frames); with hash centres 1/3 yd inside instead they would land ~5 yd short.
TICK_IN_YD = 4 / 9
LATERAL = (("tick_far", FIELD_WIDTH_YD - TICK_IN_YD), ("hash_far", HASH_FAR_Y), ("hash_near", HASH_NEAR_Y),
           ("tick_near", TICK_IN_YD))       # top of a sideline picture to bottom
RANSAC_PX = 5.0             # 1080p px: a correspondence this close to the model is an inlier
RANSAC_ITERS = 300          # hypotheses a fit (each a DLT on 4 point pairs and the yard lines)
MIN_INLIERS = 0.8           # an easy frame keeps this share of its point correspondences ...
EASY_RMS_PX = 3.0           # ... at this RMS reprojection error (1080p px) or less, with a plausible camera
EASY_YARD_LINES = 3         # ... and this many yard lines
MAX_ALTERNATIVES = 3        # `register` tries this many of `hashes.find`'s yard-line indexings ...
MAX_FITS = 60               # ... and at most this many fits for each (4 rows take 53)
ROW_INLIERS = 0.6           # every row `register` names keeps this share of its point pairs ...
SUPPORT_PX = 4.0            # ... and namings rank by dash candidates this close (1080p px) to the whole
                            # yards of all four LATERAL rows they predict (see `support`)
BOUNDARY_TOL_YD = 6.0       # a detected far/near boundary must map within this of the sideline band
CAMERA_BEHIND_YD = (20.0, 150.0)    # a sideline camera is this far behind the near sideline ...
CAMERA_HEIGHT_YD = (8.0, 60.0)      # ... this high ...
CAMERA_ORTH_DEG = 10.0              # ... and its rotation is this close to a rotation (see `camera`)
CAMERA_SAME_YD = (6.0, 2.0)         # with a game's camera known: back and height within this of it
ENDZONE_BEHIND_YD = (20.0, 200.0)   # an end-zone camera is this far beyond the yard lines it sees ...
ENDZONE_ACROSS_YD = (-20.0, 73.0)   # ... between these field y ...
ENDZONE_SAME_YD = 4.0               # ... and with a game's camera known, its height within this of it
FRONT = 3                   # `line_of_scrimmage` averages the front-most this many of each team
EDGE_PX = 4.0               # 1080p px: a box ending this close to the bottom of the frame is cut off there
ANCHOR_MAX_YD = 1.5         # `anchor` is trusted only if the players' line is this close to the official one
YARD_LINE_STEP = 5.0        # yd between painted yard lines: the relative frame is right up to whole steps


def _homog(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, float).reshape(-1, 2)
    return np.column_stack([points, np.ones(len(points))])


def apply(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Points (n, 2) through the homography h (3 x 3)."""
    p = _homog(points) @ h.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:]


def _normaliser(points: np.ndarray) -> np.ndarray:
    c = points.mean(0)
    d = np.hypot(*(points - c).T).mean()
    k = np.sqrt(2) / max(d, 1e-9)
    return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1.0]])


def dlt(image_pts: np.ndarray, field_pts: np.ndarray, line_pts: np.ndarray | None = None,
        line_x: np.ndarray | None = None) -> np.ndarray | None:
    """Homography G (image -> field) from point pairs, plus image points known only to lie on the
    field line x = line_x (the ends of a yard line). Normalised algebraic least squares."""
    image_pts = np.asarray(image_pts, float).reshape(-1, 2)
    field_pts = np.asarray(field_pts, float).reshape(-1, 2)
    lp = np.zeros((0, 2)) if line_pts is None else np.asarray(line_pts, float).reshape(-1, 2)
    lx = np.zeros(0) if line_x is None else np.asarray(line_x, float)
    if 2 * len(image_pts) + len(lp) < 8 or len(field_pts) < 2:
        return None
    ti = _normaliser(np.vstack([image_pts, lp]))
    tf = _normaliser(field_pts)             # a similarity, so field lines x = c stay lines x' = c'
    p = _homog(image_pts) @ ti.T
    f = _homog(field_pts) @ tf.T
    rows = []
    z = np.zeros(3)
    for pi, fi in zip(p, f):
        fx, fy = fi[0] / fi[2], fi[1] / fi[2]
        rows.append(np.concatenate([pi, z, -fx * pi]))
        rows.append(np.concatenate([z, pi, -fy * pi]))
    for pi, x in zip(_homog(lp) @ ti.T, lx):
        fx = tf[0, 0] * x + tf[0, 2]
        rows.append(np.concatenate([pi, z, -fx * pi]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    g = vt[-1].reshape(3, 3)
    g = np.linalg.inv(tf) @ g @ ti
    if not np.isfinite(g).all() or abs(np.linalg.det(g)) < 1e-15:
        return None
    return g / np.linalg.norm(g)


def point_errors(g: np.ndarray, image_pts: np.ndarray, field_pts: np.ndarray) -> np.ndarray:
    """Pixel distance between each image point and its field point projected back into the image."""
    try:
        back = apply(np.linalg.inv(g), field_pts)
    except np.linalg.LinAlgError:
        return np.full(len(field_pts), np.inf)
    return np.nan_to_num(np.hypot(*(back - np.asarray(image_pts, float).reshape(-1, 2)).T), nan=np.inf)


def line_errors(g: np.ndarray, line_pts: np.ndarray, line_x: np.ndarray) -> np.ndarray:
    """Pixel distance of each image point from the image of its field line x = line_x."""
    lines = g[0][None] - np.asarray(line_x, float)[:, None] * g[2][None]
    d = np.abs((lines * _homog(line_pts)).sum(1)) / np.maximum(np.hypot(lines[:, 0], lines[:, 1]), 1e-12)
    return np.nan_to_num(d, nan=np.inf)


def ransac(image_pts: np.ndarray, field_pts: np.ndarray, line_pts: np.ndarray, line_x: np.ndarray,
           tol: float, iters: int = RANSAC_ITERS,
           seed: int = 0) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """RANSAC over the point pairs, then a refit on every inlier. Returns (G image -> field, point
    inliers, line-point inliers).

    Each hypothesis is 4 point pairs (two on each of two field y, three field x or more) together with
    all the yard-line points: the hash rows are only 6 yd apart, so points on them alone leave the
    tilt across the field loose, and the yard lines pin it. Inliers: within `tol` image px.
    """
    n = len(image_pts)
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    ys = np.unique(field_pts[:, 1])
    for _ in range(iters):
        if len(ys) >= 2:
            a, b = rng.choice(ys, 2, replace=False)
            ia, ib = np.flatnonzero(field_pts[:, 1] == a), np.flatnonzero(field_pts[:, 1] == b)
            if len(ia) < 2 or len(ib) < 2:
                continue
            pick = np.concatenate([rng.choice(ia, 2, replace=False), rng.choice(ib, 2, replace=False)])
        else:
            pick = rng.choice(n, 4, replace=False)
        if len(np.unique(field_pts[pick, 0])) < 3:
            continue
        g = dlt(image_pts[pick], field_pts[pick], line_pts, line_x)
        if g is None:
            continue
        score = int((point_errors(g, image_pts, field_pts) < tol).sum()
                    + (line_errors(g, line_pts, line_x) < tol).sum())
        if score > best_n:
            best, best_n = g, score
    if best is None:
        return None, np.zeros(n, bool), np.zeros(len(line_pts), bool)
    g = best
    for _ in range(3):
        pin = point_errors(g, image_pts, field_pts) < tol
        lin = line_errors(g, line_pts, line_x) < tol
        g2 = dlt(image_pts[pin], field_pts[pin], line_pts[lin], line_x[lin])
        if g2 is None:
            break
        g = g2
    return g, point_errors(g, image_pts, field_pts) < tol, line_errors(g, line_pts, line_x) < tol


def correspondences(found: dict, rows: list[tuple[dict, float]]) -> dict:
    """Point pairs (dash centres, and where the yard lines cross each row) for `rows`, a list of
    (row, field y), and the yard lines' ends and middles with their field x, in the relative frame
    of `hashes.find`."""
    img, fld, which = [], [], []
    lp, lx = [], []
    yard, xs = found["yard"], found["x"]
    for k, (row, y) in enumerate(rows):
        for p, x in zip(row["points"], row["x"]):
            img.append(p), fld.append((x, y)), which.append(k)
        for (x1, y1, x2, y2), x in zip(yard, xs):
            c = np.cross(row["line"], np.cross([x1, y1, 1.0], [x2, y2, 1.0]))
            if abs(c[2]) > 1e-12:
                img.append(c[:2] / c[2]), fld.append((x, y)), which.append(k)
    for (x1, y1, x2, y2), x in zip(yard, xs):
        for t in (0.0, 0.5, 1.0):
            lp.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            lx.append(x)
    return {"image": np.asarray(img, float).reshape(-1, 2), "field": np.asarray(fld, float).reshape(-1, 2),
            "row": np.asarray(which, int), "line_pts": np.asarray(lp, float).reshape(-1, 2),
            "line_x": np.asarray(lx, float)}


def camera(h: np.ndarray, shape: tuple[int, int]) -> dict | None:
    """Pinhole camera behind a field -> image homography `h`, for a frame of `shape` (h, w), assuming
    square pixels and the principal point at the frame centre (zoom lenses, no crop).

    h = K [r1 r2 t]: the focal length comes from |K^-1 h1| = |K^-1 h2| (well conditioned when the
    camera looks across the field; r1 . r2 = 0 is not, there), and "orth" is then the angle between
    r1 and r2, 90 for a true rotation. Returns {"f": px, "orth": deg, "C": camera centre (field yd;
    z up)} or None when no real focal length fits.
    """
    t = np.array([[1.0, 0, -shape[1] / 2], [0, 1, -shape[0] / 2], [0, 0, 1]])
    hc = t @ h
    h1, h2, h3 = hc[:, 0], hc[:, 1], hc[:, 2]
    den = h2[2] ** 2 - h1[2] ** 2
    if abs(den) < 1e-18:
        return None
    f2 = (h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den
    if not np.isfinite(f2) or f2 <= 0:
        return None
    f = float(np.sqrt(f2))
    ki = np.diag([1 / f, 1 / f, 1.0])
    a1, a2, a3 = ki @ h1, ki @ h2, ki @ h3
    lam = (np.linalg.norm(a1) + np.linalg.norm(a2)) / 2
    r1, r2 = a1 / np.linalg.norm(a1), a2 / np.linalg.norm(a2)
    tr = a3 / lam
    if tr[2] < 0:                           # the field is in front of the camera
        r1, r2, tr = -r1, -r2, -tr
    r = np.column_stack([r1, r2, np.cross(r1, r2)])
    return {"f": f, "orth": float(np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1)))), "C": -r.T @ tr}


def plausible(cam: dict | None, prior: tuple[float, float] | None = None, angle: str = "sideline",
              x_seen: float = 0.0) -> bool:
    """A sideline camera: behind the near sideline, above the field, a near-true rotation, and with
    `prior` (the game's camera y and height, yd) within CAMERA_SAME_YD of it. An end-zone camera:
    above the field, ENDZONE_BEHIND_YD along the field from `x_seen` (the yard lines in view, in
    the relative frame), ENDZONE_ACROSS_YD across it, and with `prior` its height within
    ENDZONE_SAME_YD (only the height: its y flips with the end and its x with the zoom)."""
    if cam is None or abs(cam["orth"] - 90) > CAMERA_ORTH_DEG:
        return False
    cx, cy, cz = cam["C"]
    if not CAMERA_HEIGHT_YD[0] <= cz <= CAMERA_HEIGHT_YD[1]:
        return False
    if angle == "endzone":
        return bool(ENDZONE_BEHIND_YD[0] <= abs(cx - x_seen) <= ENDZONE_BEHIND_YD[1]
                    and ENDZONE_ACROSS_YD[0] <= cy <= ENDZONE_ACROSS_YD[1]
                    and (prior is None or abs(cz - prior[1]) <= ENDZONE_SAME_YD))
    ok = CAMERA_BEHIND_YD[0] <= -cy <= CAMERA_BEHIND_YD[1]
    if ok and prior is not None:
        ok = abs(cy - prior[0]) <= CAMERA_SAME_YD[0] and abs(cz - prior[1]) <= CAMERA_SAME_YD[1]
    return bool(ok)


def fit(found: dict, rows: list[tuple[dict, float]], shape: tuple[int, int],
        prior: tuple[float, float] | None = None, angle: str = "sideline") -> dict | None:
    """The relative homography of one frame from (an alternative of) `hashes.find`, with `rows` a
    list of (row, field y). Tolerances are in 1080p pixels, scaled to the frame height `shape[0]`.

    Returns {"G": image -> relative field (3 x 3), "H": its inverse, "rms_px", "line_rms_px" (1080p
    px, inliers only), "inliers": share of point pairs kept, "n_points", "n_yard", "camera",
    "easy": bool, "corr", "point_inliers"} or None.
    """
    if len(found["yard"]) < 2 or len(rows) < 2:
        return None
    scale = 1080 / shape[0]
    c = correspondences(found, rows)
    g, pin, lin = ransac(c["image"], c["field"], c["line_pts"], c["line_x"], RANSAC_PX / scale)
    if g is None or pin.sum() < 4:
        return None
    pe = point_errors(g, c["image"], c["field"])[pin] * scale
    le = line_errors(g, c["line_pts"], c["line_x"])[lin] * scale
    rms = float(np.sqrt(np.mean(pe ** 2)))
    h = np.linalg.inv(g)
    cam = camera(h, shape)
    easy = (rms <= EASY_RMS_PX and pin.mean() >= MIN_INLIERS and len(found["yard"]) >= EASY_YARD_LINES
            and plausible(cam, prior, angle, float(np.mean(found["x"]))))
    lrms = float(np.sqrt(np.mean(le ** 2))) if len(le) else float("nan")
    return {"G": g, "H": h, "rms_px": rms, "line_rms_px": lrms, "inliers": float(pin.mean()), "n_points": len(pin),
            "n_yard": len(found["yard"]), "camera": cam, "easy": bool(easy), "corr": c, "point_inliers": pin}


def support(h: np.ndarray, dashes: np.ndarray, xs: np.ndarray, shape: tuple[int, int]) -> int:
    """How many whole yards of the four LATERAL rows, as the field -> image homography `h` puts them in
    the frame (5 yd either side of the yard lines `xs`), have a dash candidate within SUPPORT_PX.

    Two rows of dashes fit about as well under more than one naming (the far ticks and the near hash
    row named near hash and near ticks: both pairs ~23-30 yd apart); the right naming also puts the
    rows it did not name where their dashes are, the wrong one puts them in the crowd or off the field.
    """
    if len(dashes) == 0:
        return 0
    tol = SUPPORT_PX * shape[0] / 1080
    yards = [x for x in np.arange(np.floor(xs.min()) - 5, np.ceil(xs.max()) + 6) if x % 5]
    pts = apply(h, np.array([(x, y) for _, y in LATERAL for x in yards], float))
    ok = np.isfinite(pts).all(1) & (pts[:, 0] >= 0) & (pts[:, 0] < shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < shape[0])
    pts = pts[ok]
    d = np.hypot(pts[:, None, 0] - dashes[None, :, 0], pts[:, None, 1] - dashes[None, :, 1])
    return int((d.min(1) <= tol).sum())


def boundaries_agree(g: np.ndarray, boundary: dict) -> bool:
    """False if a sideline boundary `lines.detect` found (frame px) maps, through image -> field g,
    more than BOUNDARY_TOL_YD from the 2-yd white band outside its sideline (y 53.3 to 55.3 far,
    -2 to 0 near). A naming that takes the sideline ticks for a hash row is 20 yd or more out."""
    for name, lo, hi in (("far", FIELD_WIDTH_YD, FIELD_WIDTH_YD + 2), ("near", -2.0, 0.0)):
        b = boundary.get(name)
        if b is None:
            continue
        y = float(np.median(apply(g, np.linspace(b[:2], b[2:], 9))[:, 1]))
        if not np.isfinite(y) or y < lo - BOUNDARY_TOL_YD or y > hi + BOUNDARY_TOL_YD:
            return False
    return True


def _labellings(rows: list[dict], orders: tuple = (LATERAL,)):
    """Order-keeping ways to give rows (in picture order) names from LATERAL (or any of `orders`):
    each is a list of (row index, name, y). Most rows first, and among as many rows, the sets with
    the most dashes first."""
    n = len(rows)
    for k in range(min(n, len(LATERAL)), 1, -1):
        sets = sorted(itertools.combinations(range(n), k), key=lambda c: -sum(len(rows[i]["x"]) for i in c))
        for subset in sets:
            for lateral in orders:
                for names in itertools.combinations(range(len(lateral)), k):
                    yield [(r, *lateral[m]) for r, m in zip(subset, names)]


def register(image: np.ndarray, angle: str = "sideline", found: dict | None = None,
             prior: tuple[float, float] | None = None) -> dict:
    """Relative registration of one RGB frame: `hashes.find`, then which rows are which.

    For each of the first MAX_ALTERNATIVES yard-line indexings, rows are named from LATERAL in
    picture order (two hash rows, a hash row and the ticks along its sideline, ...), using as many
    rows as give a `plausible` camera with ROW_INLIERS of every named row's point pairs kept, and on
    sideline frames `boundaries_agree` with the sideline bands `lines.detect` found: a wrong
    naming fits about as well (two lines of dashes and the yard lines always fit a homography) but
    puts the camera in the crowd or under the field. Among namings with that many rows, the one with
    the most `support` wins, then the one keeping the most point pairs; the first indexing with any
    such naming is kept. `prior`:
    the game's camera (y, height), if known. End-zone frames (as `lines.detect` reads them) name
    rows left to right in either order, since the camera can be behind either end zone.

    Returns {"found": `hashes.find` output, "used": the alternative used, "fit": `fit` output or None,
    "rows": list of (row, name, y)}.
    """
    from formation_zero.perception.field import hashes

    found = found if found is not None else hashes.find(image, angle)
    angle = found["det"]["angle"] if "det" in found else angle
    orders = (LATERAL,) if angle == "sideline" else (LATERAL, LATERAL[::-1])     # end zone: either end
    for alt in found["alternatives"][:MAX_ALTERNATIVES]:
        best, size, fits = None, None, 0
        for lab in _labellings(alt["rows"], orders):
            if (size is not None and len(lab) < size) or fits >= MAX_FITS:
                break
            fits += 1
            r = fit(alt, [(alt["rows"][i], y) for i, _, y in lab], image.shape[:2], prior, angle)
            if r is None or not plausible(r["camera"], prior, angle, float(np.mean(alt["x"]))):
                continue
            row, pin = r["corr"]["row"], r["point_inliers"]
            if min(pin[row == k].mean() for k in range(len(lab))) < ROW_INLIERS:
                continue
            if angle == "sideline" and not boundaries_agree(r["G"], found.get("det", {}).get("boundary", {})):
                continue
            size = len(lab)
            key = (support(r["H"], found.get("dashes", np.zeros((0, 2))), alt["x"], image.shape[:2]),
                   int(pin.sum()), -r["rms_px"])
            if best is None or key > best[0]:
                best = (key, r, [(alt["rows"][i], name, y) for i, name, y in lab])
        if best is not None:
            return {"found": found, "used": alt, "fit": best[1], "rows": best[2]}
    return {"found": found, "used": None, "fit": None, "rows": []}


# ---------------------------------------------------------------- the absolute anchor


def feet(boxes: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Foot points (bottom middle) of boxes (n, 4: x1 y1 x2 y2, frame px) and which to use: a box cut
    off by the bottom of the frame (within EDGE_PX) has its feet out of the picture, and its bottom
    edge lands yards inside the field (the near bench, seen from the sideline)."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    pts = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
    return pts, b[:, 3] < shape[0] - EDGE_PX * shape[0] / 1080


def line_of_scrimmage(x: np.ndarray, offense: np.ndarray) -> dict | None:
    """Where the line of scrimmage is along the field, from the players' feet before the snap.

    `x`: field x of each player's feet (yd, any frame); `offense`: bool, True for the offense. The
    offense stands on one side and the defense on the other. For each direction the offense could
    be facing, the split point that misplaces the fewest players is found; the direction with fewer
    misplaced wins. The line is midway between the FRONT front-most players of each side (the two
    lines of linemen). Returns {"x", "attack": +1 if the offense faces +x, "misplaced", "gap": yd
    between the two fronts} or None without three players a side.
    """
    x, offense = np.asarray(x, float), np.asarray(offense, bool)
    if offense.sum() < FRONT or (~offense).sum() < FRONT:
        return None
    best = None
    for attack in (1, -1):
        z = attack * x
        cuts = np.sort(z)
        cuts = np.concatenate([[cuts[0] - 1], (cuts[1:] + cuts[:-1]) / 2, [cuts[-1] + 1]])
        wrong = np.array([(offense & (z > t)).sum() + (~offense & (z < t)).sum() for t in cuts])
        k = int(np.argmin(wrong))
        cand = (int(wrong[k]), attack, cuts[k])
        if best is None or cand[0] < best[0]:
            best = cand
    misplaced, attack, t = best
    z = attack * x
    off, dfn = np.sort(z[offense & (z <= t)]), np.sort(z[~offense & (z >= t)])
    if len(off) == 0 or len(dfn) == 0:
        return None
    front_o = float(np.mean(off[-FRONT:]))
    front_d = float(np.mean(dfn[:FRONT]))
    return {"x": attack * (front_o + front_d) / 2, "attack": attack, "misplaced": misplaced,
            "gap": front_d - front_o}


def los_absolute_x(yardline_100: float, attack: int) -> float:
    """Absolute field x (nfl_field: left goal line at 10, x growing to the right of the sideline
    camera's picture) of a line of scrimmage `yardline_100` yd from the goal the offense attacks."""
    return 110.0 - yardline_100 if attack > 0 else 10.0 + yardline_100


def anchor(los_rel: float, attack: int, yardline_100: float) -> dict:
    """The whole number of 5-yd steps that takes the relative frame to the absolute one: absolute x =
    relative x + "shift". "residual" (yd) is how far the players put the line from the official
    spot once shifted; past 2.5 yd the shift would be a step off, so "ok" asks for ANCHOR_MAX_YD."""
    target = los_absolute_x(yardline_100, attack)
    shift = YARD_LINE_STEP * round((target - los_rel) / YARD_LINE_STEP)
    residual = target - (los_rel + shift)
    return {"shift": shift, "residual": residual, "los_x": target, "ok": abs(residual) <= ANCHOR_MAX_YD}


def to_absolute(g: np.ndarray, shift: float) -> np.ndarray:
    """Image -> absolute field homography from a relative one and its `anchor` shift."""
    return np.array([[1.0, 0, shift], [0, 1, 0], [0, 0, 1]]) @ g


__all__ = ["HASH_FAR_Y", "HASH_NEAR_Y", "LATERAL", "anchor", "apply", "boundaries_agree", "camera",
           "correspondences", "dlt", "feet", "fit", "line_errors", "line_of_scrimmage", "los_absolute_x",
           "plausible", "point_errors", "ransac", "register", "support", "to_absolute"]
