"""A play you can walk around: the fitted bodies stood on a real field, for the film room (GRD-60).

The fits from `play.py` are each in their own camera space, about a fifth too small and too near.
This turns them into one scene in field yards. Runs in the spike's environment:

    PY=.venv-sam3d/bin/python; S=scripts/spikes/sam3d_body/room.py
    $PY $S camera --play 61                    # the camera's rotation through the take, and the film's clock
    $PY $S ball --play 61 --seed 53642,764,440 --spot 53562,695,685 --snapped 53570 --release 53626 --catch 53643 \
        --tuck 53680 --thrower 12 --catcher 3
    $PY $S build --play 61 --mark 3=Kelce 12=Mahomes --last 53724   # <game>/bodies/p061.room/: open index.html in Chrome

0. Time: the film was 30 frames a second and every sixth was dropped to make 25, so every fifth step
   in it is a double one. Everything here runs on the true 30 a second, a tick, and the dropped frame
   is filled in. On play 61 that took the ball's arc from 12 px to 3 px off the film at worst, and the
   five-a-second swing in the speed of a filtered limb from 15% to 6%.
1. Rotation: each body is turned from camera axes to the field's by the camera read from the painted
   lines every READ_EVERY frames.
2. Bad frames: a flip (hips off the track's running facing by more than FLIP) or a giant (too near the
   camera for the track) is dropped and bridged from the frames either side.
3. Filtering: what is left is smoothed over time, on the vertices themselves.
4. Placing: the pelvis goes where the box's foot point in yards says, by the fit's own offset between
   the two in the picture, at the height that puts the lowest point on the ground. Pinning the lowest
   point itself made bodies lurch at every stride: that point changes feet, the box's middle does not
   (0.069 yd a frame squared in play on play 40, 0.012 this way). Every body is scaled by the real
   distance from the camera over the model's.

5. Ball: on the ground it is read off the film. In flight it is followed from one hand-picked sighting
   and a ballistic arc is fitted to its rays and to the hands at each end. Held, it is put in the hands:
   that part is inferred, though on play 61 it lands about 10 px from the ball where the film shows it.

The page draws a to-scale field, not the film: the film has only the one view.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation, Slerp

from play import ANGLE, GIANT, YD_M, ROOT, people, stem, take_dir

sys.path.insert(0, str(ROOT))

from formation_zero.data.layout import GamePaths  # noqa: E402

READ_EVERY = 10             # frames between camera readings
CELL_CM = 3.5               # the mesh is thinned on a grid this size in its rest pose: about 1,900 points
FLIP = 60.0                 # degrees off the track's running facing
SPIN = 500.0                # degrees a second after filtering: a flip smeared out, not a player turning
FACE_WINDOW = 25            # frames in that running facing: a real turn stays inside it, a short flip does not
SMOOTH = 1.9                # ticks, sigma over time. Stride-speed motion kept against noise left, play 40 in play:
                            # 1.4 75% and 0.030 yd a frame squared, 2.4 48% and 0.016. Savitzky-Golay lands on the same curve
ROOT_SMOOTH = 3.6           # ticks, sigma on where the body stands
LEAN_LAST = 40              # frames at the end of a take: men standing, for the check on the rotation
ARMS = [41, 62, 8, 7]       # keypoints: right wrist, left wrist, right elbow, left elbow
HZ = 30.0                   # ticks a second: the film's clock before every sixth frame was dropped
DROP = 5                    # a double step every this many frames
LIFT = 26                   # the ball on turf: this much darker than the patch round it, summed over RGB
G_YD = 9.81 / YD_M          # gravity, yd/s/s
BALL_Z = 0.09               # yd: the middle of a ball lying on the ground
SNAP_TICKS = 13             # the ball's trip from the centre to a shotgun quarterback's hands
WIND = 10                   # ticks over which the ball goes from both hands to the throwing hand
TUCK = 14                   # ticks over which a catcher takes it from both hands into the arm
HAND = Path(__file__).with_name("room.html")


def cmd_camera(paths: GamePaths, a) -> int:
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source
    from formation_zero.perception.field import homography as hg

    frames = sorted(int(p.stem) for p in (take_dir(paths, a.play) / "fits").glob("*.npz"))
    b = people(paths, a.play, frames[0] if frames else a.first)
    span = int(b.frame.min()), int(b.frame.max()) + 1
    out, step, last = {}, {}, None
    with FileSource(best_source(paths)) as source:
        for fr in source.iter_frames(*span, step=1):
            small = cv2.resize(cv2.cvtColor(fr.image, cv2.COLOR_RGB2GRAY), (240, 135)).astype(np.float32)
            if last is not None:
                step[fr.index] = float(np.abs(small - last).mean())
            last = small
            if (fr.index - span[0]) % READ_EVERY:
                continue
            fit = hg.register(fr.image, ANGLE)["fit"]
            if fit is not None and hg.plausible(fit["camera"], None):
                c = fit["camera"]
                out[fr.index] = {"R": np.asarray(c["R"]).tolist(), "C": np.asarray(c["C"]).tolist(),
                                 "f": float(c["f"]), "G": np.asarray(fit["G"]).tolist()}
    d = take_dir(paths, a.play)
    (d / "camera.json").write_text(json.dumps({str(k): v for k, v in out.items()}))
    by = pd.Series(step).groupby(lambda f: f % DROP).median().sort_values(ascending=False)
    (d / "cadence.json").write_text(json.dumps({"phase": int(by.index[0]), "ratio": float(by.iloc[0] / by.iloc[1])}))
    print(f"{len(out)} of {(span[1] - span[0]) // READ_EVERY + 1} readings; "
          f"the double step lands on frames {int(by.index[0])} mod {DROP}, {by.iloc[0] / by.iloc[1]:.2f} times the next")
    return 0


def blobs(image: np.ndarray, boxes: np.ndarray, at: np.ndarray, reach: int = 44) -> list[np.ndarray]:
    """Small dark smears near `at`: on this film a ball in the air is darker than the turf, not brown.
    Inside a box is left out, so that a shoe is not taken for it."""
    x0, y0 = int(max(at[0] - reach, 0)), int(max(at[1] - reach, 0))
    crop = image[y0:int(at[1]) + reach, x0:int(at[0]) + reach].astype(np.float32)
    dark = np.median(crop.reshape(-1, 3), axis=0).sum() - crop.sum(axis=2) > LIFT
    for b in boxes.astype(int):
        dark[max(b[1] - 1 - y0, 0):max(b[3] + 2 - y0, 0), max(b[0] - 1 - x0, 0):max(b[2] + 2 - x0, 0)] = False
    n, _, stats, mid = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    return [mid[i] + (x0, y0) for i in range(1, n) if 8 <= stats[i][4] <= 130 and stats[i][2] <= 34 and stats[i][3] <= 20]


def follow(images: dict[int, np.ndarray], boxes: pd.DataFrame, f: int, p: np.ndarray, step: np.ndarray) -> dict[int, np.ndarray]:
    """Back through the flight from one sighting, taking the smear nearest where the last steps point."""
    out, miss = {f: p}, 0
    while f - 1 - miss in images and miss < 6:
        at = f - 1 - miss
        want = p + step * (1 + miss)
        near = sorted(blobs(images[at], boxes[boxes.frame == at][["x1", "y1", "x2", "y2"]].to_numpy(), want),
                      key=lambda q: np.hypot(*(q - want)))
        if near and np.hypot(*(near[0] - want)) < 40 and abs(near[0][1] - want[1]) < 14:
            step = 0.6 * step + 0.4 * (near[0] - p) / (1 + miss)
            f, p, miss = at, near[0], 0
            out[f] = p
        else:
            miss += 1
    return out


def cmd_ball(paths: GamePaths, a) -> int:
    """The film's sightings of the ball in flight and, for each frame, the map from picture to ground."""
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source
    from formation_zero.perception.field import homography as hg

    f0, u0, v0 = (float(x) for x in a.seed.split(","))
    boxes = pd.read_parquet(paths.tracking_dir / f"{stem(paths, a.play)}.boxes.parquet")
    top = pd.read_parquet(paths.tracking_dir / f"{stem(paths, a.play)}.topdown.parquet")
    with FileSource(best_source(paths)) as source:
        images = {fr.index: fr.image for fr in source.iter_frames(a.release - 2, a.catch + 3, step=1)}
    hits = follow(images, boxes, int(f0), np.array([u0, v0]), np.array([-30.0, 0]))
    ground, centres = {}, []
    for f, image in images.items():
        fit = hg.register(image, ANGLE)["fit"]
        if fit is None or not hg.plausible(fit["camera"], None):
            continue
        b = boxes[boxes.frame == f].merge(top[top.frame == f][["track_id", "x_yd"]], on="track_id").dropna(subset=["x_yd"])
        foot = hg.apply(np.array(fit["G"]), np.c_[(b.x1 + b.x2) / 2, b.y2])
        shift = float(np.median(b.x_yd.to_numpy() - foot[:, 0]))            # to absolute yards, frame by frame
        ground[f] = {"G": np.asarray(fit["G"]).tolist(), "shift": shift}
        centres.append(np.asarray(fit["camera"]["C"]) + [shift, 0, 0])
    camera = np.median(centres, axis=0)                                     # the camera does not move
    f, u, v = (float(x) for x in a.spot.split(","))                         # the ball down, before the snap
    b = boxes[boxes.frame == f].merge(top[top.frame == f][["track_id", "x_yd"]], on="track_id").dropna(subset=["x_yd"])
    g = np.array(json.loads((take_dir(paths, a.play) / "camera.json").read_text())[str(int(f))]["G"])
    shift = float(np.median(b.x_yd.to_numpy() - hg.apply(g, np.c_[(b.x1 + b.x2) / 2, b.y2])[:, 0]))
    down = np.r_[hg.apply(g, np.array([[u, v]]))[0] + [shift, 0], 0.0]
    spot = camera + (down - camera) * (1 - BALL_Z / camera[2])
    out = {"hits": {str(k): np.round(v, 2).tolist() for k, v in sorted(hits.items())},
           "ground": {str(k): v for k, v in ground.items()}, "camera": camera.tolist(), "spot": spot.tolist(), "snapped": a.snapped,
           "release": a.release, "catch": a.catch, "tuck": a.tuck, "thrower": a.thrower, "catcher": a.catcher}
    (take_dir(paths, a.play) / "ball.json").write_text(json.dumps(out))
    print(f"{len(hits)} sightings, {min(hits)} to {max(hits)}; {len(ground)} of {len(images)} frames on the ground map; "
          f"camera at {np.round(out['camera'], 1)}")
    return 0


def rotations(paths: GamePaths, play: int):
    """frame -> field-to-camera rotation: a running median through the readings, slerped between them."""
    got = json.loads((take_dir(paths, play) / "camera.json").read_text())
    x = np.array(sorted(map(int, got)))
    rv = np.stack([Rotation.from_matrix(np.array(got[str(k)]["R"])).as_rotvec() for k in x])     # nearest true rotation
    rv = np.stack([np.median(rv[max(0, i - 2):i + 3], axis=0) for i in range(len(rv))])
    sl = Slerp(x, Rotation.from_rotvec(rv))
    return lambda f: sl(np.clip(f, x[0], x[-1])).as_matrix()


def thin(rest: np.ndarray, faces: np.ndarray, cell: float = CELL_CM) -> tuple[np.ndarray, np.ndarray]:
    """Keep the point nearest the middle of each patch of the rest pose and rebuild the faces. A patch is
    the part of a grid cell joined by the mesh's own edges, so two thighs in one cell stay two thighs.
    Every body has the same mesh, so one choice serves them all."""
    key = np.floor(rest / cell).astype(np.int64)
    _, cells = np.unique(key, axis=0, return_inverse=True)
    cells = cells.ravel()
    a, c = np.r_[faces[:, 0], faces[:, 1], faces[:, 2]], np.r_[faces[:, 1], faces[:, 2], faces[:, 0]]
    same = cells[a] == cells[c]
    n = len(rest)
    _, inv = connected_components(coo_matrix((np.ones(same.sum()), (a[same], c[same])), shape=(n, n)), directed=False)
    middle = np.stack([np.bincount(inv, rest[:, i]) / np.bincount(inv) for i in range(3)], axis=1)
    gap = np.linalg.norm(rest - middle[inv], axis=1)
    keep = np.lexsort((gap, inv))
    keep = keep[np.r_[True, np.diff(inv[keep]) != 0]]
    f = inv[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    once = np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]
    return keep, f[np.sort(once)]


def rest_mesh(paths: GamePaths) -> np.ndarray:
    import torch

    m = torch.jit.load(paths.film_root / "models" / "sam-3d-body-vith" / "assets" / "mhr_model.pt", map_location="cpu")
    return m.character_torch.mesh.rest_vertices.numpy()


def load(paths: GamePaths, play: int, keep: np.ndarray) -> pd.DataFrame:
    """One row a fit: the thinned vertices about the pelvis in field axes (m), and what placing needs."""
    d = take_dir(paths, play)
    R = rotations(paths, play)
    W, H = 1920, 1080
    rows = []
    for p in sorted((d / "fits").glob("*.npz"), key=lambda q: int(q.stem)):
        z = np.load(p)
        f = int(p.stem)
        r = R(f)
        for i, t in enumerate(z["track"]):
            k = z["keypoints"][i]
            pelvis = (k[9] + k[10]) / 2
            full = z["vertices"][i].astype(np.float32) / 1000 - pelvis
            hips = k[9] - k[10]
            x1, y1, x2, y2 = z["box"][i]
            at = pelvis + z["cam_t"][i]                     # where the fit puts the pelvis in the picture ...
            f_px = float(z["focal"])
            off = np.array([(x1 + x2) / 2 - W / 2 - f_px * at[0] / at[2], y2 - H / 2 - f_px * at[1] / at[2], 0]) * at[2] / f_px
            rows.append({"frame": f, "track": int(t), "team": int(z["team"][i]), "filled": bool(z["filled"][i]),
                         "V": (r.T @ full[keep].T).T, "up": r.T @ ((k[5] + k[6]) / 2 - pelvis),
                         "arms": (r.T @ (k[ARMS] - pelvis).T).T,
                         "off": r.T @ off, "view": r.T @ np.array([0.0, 0, 1]),   # ... and the box's foot point from it
                         "hips": r.T @ hips, "range": float(np.linalg.norm(z["cam_t"][i])),
                         "u": float((x1 + x2) / 2), "v": float(y2)})
    return pd.DataFrame(rows).sort_values(["track", "frame"]).reset_index(drop=True)


def real_scale(paths: GamePaths, play: int, t: pd.DataFrame) -> float:
    """Real distance from the camera over the model's, median over the readings."""
    from formation_zero.perception.field import homography as hg

    got = json.loads((take_dir(paths, play) / "camera.json").read_text())
    out = []
    for k, c in got.items():
        g = t[t.frame == int(k)]
        if len(g):
            yd = hg.apply(np.array(c["G"]), g[["u", "v"]].to_numpy(float))
            real = np.linalg.norm(np.c_[yd, np.zeros(len(yd))] - np.array(c["C"]), axis=1) * YD_M
            out.append(real / g["range"].to_numpy())
    return float(np.median(np.concatenate(out)))


def bad_frames(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(flips, giants) along one track's rows."""
    hips = np.stack(g.hips.to_numpy())
    face = np.arctan2(hips[:, 1], hips[:, 0])
    c = pd.Series(np.cos(face)).rolling(FACE_WINDOW, center=True, min_periods=5).median()
    s = pd.Series(np.sin(face)).rolling(FACE_WINDOW, center=True, min_periods=5).median()
    off = np.degrees(np.abs((face - np.arctan2(s, c) + np.pi) % (2 * np.pi) - np.pi))
    run = g["range"].rolling(51, center=True, min_periods=8).median()
    return off > FLIP, (g["range"] / run < GIANT).to_numpy()


def bridge(y: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace the bad rows of `y` (n, ...) by a straight line between the good rows either side."""
    n = len(y)
    good = np.flatnonzero(~bad)
    if len(good) < 2 or not bad.any():
        return y
    out = y.copy()
    i = np.searchsorted(good, np.flatnonzero(bad))
    lo, hi = good[np.clip(i - 1, 0, len(good) - 1)], good[np.clip(i, 0, len(good) - 1)]
    w = (np.flatnonzero(bad) - lo) / np.maximum(hi - lo, 1)
    w = w.reshape((-1,) + (1,) * (y.ndim - 1))
    out[bad] = (1 - w) * y[lo] + w * y[hi]
    return out


def ticks(frames, phase: int):
    """Frame -> true time in ticks: one more for every dropped frame passed."""
    f = np.asarray(frames)
    return f + (f - phase) // DROP + 1


def even(tick: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n, ...) values at `tick` -> one row for every tick from first to last, straight between."""
    grid = np.arange(tick[0], tick[-1] + 1)
    i = np.clip(np.searchsorted(tick, grid, side="right") - 1, 0, len(tick) - 2)
    w = ((grid - tick[i]) / (tick[i + 1] - tick[i])).reshape((-1,) + (1,) * (y.ndim - 1))
    return y[i] * (1 - w) + y[i + 1] * w


def scene(paths: GamePaths, a) -> dict:
    d = take_dir(paths, a.play)
    phase = json.loads((d / "cadence.json").read_text())["phase"]
    faces = np.load(d / "faces.npy")
    keep, tri = thin(rest_mesh(paths), faces)
    t = load(paths, a.play, keep)
    scale = real_scale(paths, a.play, t) / YD_M                    # model metres -> real yards
    top = pd.read_parquet(paths.tracking_dir / f"{stem(paths, a.play)}.topdown.parquet")[["frame", "track_id", "x_yd", "y_yd"]]
    t = t.merge(top.rename(columns={"track_id": "track"}), on=["frame", "track"], how="left")
    tracks, flips, giants, tilt = [], 0, 0, []
    left, steps, apart = 0, 0, []
    for tr, g in t.groupby("track"):
        g = g.sort_values("frame").reset_index(drop=True)
        if g.x_yd.notna().sum() < 5:
            continue
        flip, giant = bad_frames(g)
        flips, giants = flips + int(flip.sum()), giants + int(giant.sum())
        bad = flip | giant
        tick = ticks(g.frame.to_numpy(), phase)
        grid = np.arange(tick[0], tick[-1] + 1)
        live = np.isin(grid, tick) | np.isin(grid - 1, tick)        # a dropped frame, not a track's own gap

        def clean(name: str) -> np.ndarray:
            return gaussian_filter1d(even(tick, bridge(np.stack(g[name].to_numpy()), bad)), SMOOTH, axis=0, mode="nearest")

        hips = clean("hips")
        face = np.degrees(np.arctan2(hips[:, 1], hips[:, 0]))
        step = np.abs((np.diff(face) + 180) % 360 - 180)
        left, steps = left + int((step > SPIN / HZ).sum()), steps + len(step)
        V = clean("V") * scale
        off = clean("off") * scale
        z = -V[:, :, 2].min(axis=1)                                 # the pelvis over the lowest point; field z is up
        feet = even(tick, bridge(g[["x_yd", "y_yd"]].astype(float).interpolate(limit_direction="both").to_numpy(), bad))
        view = even(tick, np.stack(g.view.to_numpy()))
        along = -(z + off[:, 2]) / view[:, 2]                       # back along the ray to the pelvis's height
        xy = gaussian_filter1d(feet - off[:, :2] - along[:, None] * view[:, :2], ROOT_SMOOTH, axis=0, mode="nearest")
        apart.append(np.linalg.norm(xy - feet, axis=1))
        up = np.stack(g.up.to_numpy())
        tilt.append(np.degrees(np.arccos(up[:, 2] / np.linalg.norm(up, axis=1))))
        root = np.c_[xy, z]
        tracks.append({"id": int(tr), "team": int(g.team.iloc[0]), "frames": grid[live], "root": root[live], "V": V[live],
                       "arms": (clean("arms") * scale + root[:, None])[live]})
    done = np.concatenate([k[-LEAN_LAST:] for k, g in zip(tilt, tracks) if g["frames"][-1] == ticks(t.frame.max(), phase)])
    print(f"{len(t)} fits on {len(tracks)} tracks, {len(keep)} points, {len(tri)} faces, scale {scale * YD_M:.2f}; "
          f"dropped and bridged: {flips} flips, {giants} giants; steps turning over {SPIN:.0f} deg/s after that: {left} of {steps}; "
          f"pelvis from the tracked feet, median {np.median(np.concatenate(apart)):.2f} yd; "
          f"lean of the back off vertical: all {np.median(np.concatenate(tilt)):.0f} deg, "
          f"last {LEAN_LAST} frames {np.median(done):.0f}")
    last = int(ticks(a.last or t.frame.max(), phase))
    for k in tracks:
        ok = k["frames"] <= last
        k.update({name: k[name][ok] for name in ("root", "V", "arms", "frames")})
    return {"keep": keep, "tri": tri, "tracks": [k for k in tracks if len(k["frames"])], "first": int(ticks(t.frame.min(), phase)),
            "last": last, "fps": HZ, "phase": phase}


def flight(b: dict, phase: int, g_free: bool, p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float, dict[int, np.ndarray]]:
    """The arc from the release to the catch: (start and velocity, gravity, pixels off at each sighting).
    The film fixes it across the picture; it says little about depth, which the two ends fix."""
    from formation_zero.perception.field import homography as hg

    C = np.array(b["camera"])
    seen = sorted(f for f in map(int, b["hits"]) if b["release"] < f <= b["catch"] and str(f) in b["ground"])
    t = (ticks(seen, phase) - ticks(b["release"], phase)) / HZ
    uv = np.array([b["hits"][str(f)] for f in seen])
    back = [np.linalg.inv(np.array(b["ground"][str(f)]["G"])) for f in seen]
    shift = [b["ground"][str(f)]["shift"] for f in seen]
    span = (ticks(b["catch"], phase) - ticks(b["release"], phase)) / HZ

    def picture(q: np.ndarray, g: float) -> np.ndarray:
        x = q[:3] + q[3:6] * t[:, None]
        x[:, 2] -= g / 2 * t**2
        on = C[:2] + (x[:, :2] - C[:2]) * (C[2] / (C[2] - x[:, 2]))[:, None]        # where the ray lands on the field
        return np.concatenate([hg.apply(h, o[None] - [s, 0]) for h, o, s in zip(back, on, shift)])

    def res(q: np.ndarray) -> np.ndarray:
        g = q[6] if g_free else G_YD
        end = q[:3] + q[3:6] * span - [0, 0, g / 2 * span**2]
        return np.r_[((picture(q, g) - uv) / 3).ravel(), (q[:3] - p0) / 0.35, (end - p1) / 0.35]

    v = (p1 - p0) / span + [0, 0, G_YD * span / 2]
    q = least_squares(res, np.r_[p0, v, G_YD] if g_free else np.r_[p0, v], loss="soft_l1", f_scale=2.0).x
    g = float(q[6]) if g_free else G_YD
    return q[:6], g, dict(zip(seen, np.linalg.norm(picture(q, g) - uv, axis=1)))


def ball_path(paths: GamePaths, a, s: dict, play: dict) -> dict | None:
    """Where the ball is on every tick: on the ground, on its way back, held, in the air, held."""
    f = take_dir(paths, a.play) / "ball.json"
    if not f.exists():
        return None
    b = json.loads(f.read_text())
    phase = s["phase"]
    tracks = {k["id"]: k for k in s["tracks"]}
    qb, rec = tracks[b["thrower"]], tracks[b["catcher"]]
    arms = lambda k, t: k["arms"][np.clip(np.searchsorted(k["frames"], t), 0, len(k["frames"]) - 1)]
    t0, t1, tuck = (int(ticks(b[k], phase)) for k in ("release", "catch", "tuck"))
    q, _, off = flight(b, phase, False, arms(qb, t0)[0], arms(rec, t1)[:2].mean(axis=0))
    _, g, _ = flight(b, phase, True, arms(qb, t0)[0], arms(rec, t1)[:2].mean(axis=0))
    errs = np.array(list(off.values()))
    p0, v0 = q[:3], q[3:]
    span = (t1 - t0) / HZ
    end = p0 + v0 * span - [0, 0, G_YD / 2 * span**2]
    print(f"ball: {len(errs)} sightings on the arc, median {np.median(errs):.1f} px off, worst {errs.max():.1f}; "
          f"{np.linalg.norm(v0):.1f} yd/s, {np.linalg.norm((end - p0)[:2]):.1f} yd in {span:.2f} s, peak {p0[2] + v0[2]**2 / 2 / G_YD:.2f} yd; "
          f"gravity if left free {g * YD_M:.1f} m/s/s; start {np.linalg.norm(p0 - arms(qb, t0)[0]):.2f} yd from the wrist, "
          f"end {np.linalg.norm(end - arms(rec, t1)[:2].mean(axis=0)):.2f} yd from the hands")
    snap, took, spot = int(ticks(b["snapped"], phase)), int(ticks(b["snapped"], phase)) + SNAP_TICKS, np.array(b["spot"])
    T = np.arange(s["first"], s["last"] + 1)
    pos, axis = np.tile(spot, (len(T), 1)), np.tile([1.0, 0, 0], (len(T), 1))
    fade = lambda x: (lambda w: w * w * (3 - 2 * w))(np.clip(x, 0, 1))
    forearm = lambda A: A[:, 0] - A[:, 2]
    A = arms(qb, T)
    hands = A[:, :2].mean(axis=1)
    cocked = A[:, 0] + (p0 - arms(qb, t0)[0])
    w = fade((T - (t0 - WIND)) / WIND)[:, None]
    held = (T >= took) & (T <= t0)
    pos[held], axis[held] = (hands * (1 - w) + cocked * w)[held], forearm(A)[held]
    go = (T > snap) & (T < took)
    w = ((T - snap) / SNAP_TICKS)[:, None]
    pos[go] = (spot * (1 - w) + hands[np.searchsorted(T, took)] * w + [0, 0, 0.35] * (4 * w * (1 - w)))[go]
    air = (T > t0) & (T <= t1)
    dt = ((T - t0) / HZ)[:, None]
    pos[air] = (p0 + v0 * dt - [0, 0, G_YD / 2] * dt**2)[air]
    axis[air] = (v0 - [0, 0, G_YD] * dt)[air]
    after = T > t1
    A = arms(rec, T)
    w = fade((T - tuck) / TUCK)[:, None]
    carry = A[:, :2].mean(axis=1) * (1 - w) + (0.6 * A[:, 0] + 0.4 * A[:, 2]) * w
    pos[after] = (carry + (end - carry[np.searchsorted(T, t1)]) * (1 - fade((T - t1) / 5))[:, None])[after]
    axis[after] = forearm(A)[after]
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    axis[T <= t0] = gaussian_filter1d(axis[T <= t0], 2.0, axis=0, mode="nearest")
    axis[after] = gaussian_filter1d(axis[after], 2.0, axis=0, mode="nearest")
    pos[after] = gaussian_filter1d(pos[after], 1.2, axis=0, mode="nearest")
    return {"pos": np.round(pos, 3).ravel().tolist(), "axis": np.round(axis, 3).ravel().tolist(), "snap": snap, "held": took, "release": t0, "catch": t1,
            "names": [qb["name"], rec["name"]]}


def facts(paths: GamePaths, a, s: dict) -> dict:
    """What the page says and draws about the play: the snap, the lines, the camera in absolute yards."""
    from formation_zero.perception.field import homography as hg

    td = json.loads((paths.tracking_dir / f"{stem(paths, a.play)}.topdown.json").read_text())
    pbp = pd.read_parquet(paths.pbp_path)
    row = pbp[pbp.play_uid == f"{paths.game_key}_p{a.play:03d}"].iloc[0]
    attack = 1 if td["anchor"].get("attack", 1) > 0 else -1
    got = json.loads((take_dir(paths, a.play) / "camera.json").read_text())
    top = pd.read_parquet(paths.tracking_dir / f"{stem(paths, a.play)}.topdown.parquet")
    boxes = pd.read_parquet(paths.tracking_dir / f"{stem(paths, a.play)}.boxes.parquet")
    k = sorted(got, key=int)[len(got) // 2]
    b = boxes[boxes.frame == int(k)].merge(top[top.frame == int(k)][["track_id", "x_yd"]], on="track_id").dropna(subset=["x_yd"])
    foot = hg.apply(np.array(got[k]["G"]), np.c_[(b.x1 + b.x2) / 2, b.y2])
    shift = float(np.median(b.x_yd.to_numpy() - foot[:, 0]))
    cam = np.array(got[k]["C"]) + [shift, 0, 0]
    snap = int(ticks(td["snap_frame"], s["phase"]))
    return {"snap": snap, "los": td["los_x"], "attack": attack, "first_down": td["los_x"] + attack * float(row.ydstogo),
            "camera": np.round(cam, 2).tolist(), "start": max(0, snap - s["first"] - 24), "gain": float(row.yards_gained),
            "title": a.title or f"Play {a.play}", "note": a.note or str(row.desc)[:160], "teams": (a.teams or "Team 0,Team 1").split(",")}


def pack(s: dict, marks: dict[int, str], play: dict, ball: dict | None) -> str:
    """The scene as a script tag's worth of JS: one gzip blob of mm int16 deltas over time, track after
    track, since a page opened from disk cannot fetch a file beside it."""
    meta, blobs = [], []
    for k in s["tracks"]:
        v = np.round(k["V"] * 1000).astype(np.int32)                # thousandths of a yard
        d = np.diff(v, axis=0, prepend=np.zeros_like(v[:1]))
        blobs.append(np.clip(d, -32767, 32767).astype(np.int16).tobytes())
        meta.append({"id": k["id"], "team": k["team"], "frames": k["frames"].tolist(),
                     "root": np.round(k["root"], 3).ravel().tolist(), "name": marks.get(k["id"])})
    raw = gzip.compress(b"".join(blobs), 6)
    head = {"first": s["first"], "last": s["last"], "fps": s["fps"], "nv": len(s["keep"]), "tracks": meta,
            "faces": s["tri"].astype(int).ravel().tolist(), "ball": ball, **play}
    return f"window.SCENE={json.dumps(head, separators=(',', ':'))};window.BODIES='{base64.b64encode(raw).decode()}';"


def cmd_build(paths: GamePaths, a) -> int:
    s = scene(paths, a)
    marks = {int(k): v for k, v in (m.split("=") for m in a.mark or [])}
    out = paths.raw_dir / "bodies" / f"p{a.play:03d}.room"
    out.mkdir(parents=True, exist_ok=True)
    for k in s["tracks"]:
        k["name"] = marks.get(k["id"])
    play = facts(paths, a, s)
    ball = ball_path(paths, a, s, play)
    if ball:
        print(f"ball down at x {ball['pos'][0]:.2f}, the official line {play['los']:.1f}; "
              f"ends {ball['pos'][-3]:.1f}, the official spot {play['los'] + play['attack'] * play['gain']:.1f}")
        play["snap"] = ball["snap"]                                 # the film's, a frame or two better than the tracker's guess
    js = pack(s, marks, play, ball)
    (out / "scene.js").write_text(js)
    (out / "index.html").write_text(HAND.read_text())
    print(f"{len(js) / 1e6:.1f} MB -> {out / 'index.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"camera": cmd_camera, "ball": cmd_ball, "build": cmd_build}
    sps = {name: sub.add_parser(name) for name in cmds}
    for sp in sps.values():
        sp.add_argument("--season", type=int, default=2026)
        sp.add_argument("--week", type=int, default=6)
        sp.add_argument("--away", default="LAC")
        sp.add_argument("--home", default="KC")
        sp.add_argument("--play", type=int, required=True)
        sp.add_argument("--first", type=int)
    sps["ball"].add_argument("--seed", required=True, help="frame,u,v of a sighting late in the flight")
    sps["ball"].add_argument("--spot", required=True, help="frame,u,v of the ball down before the snap, on a camera reading")
    for k in ("snapped", "release", "catch", "tuck", "thrower", "catcher"):
        sps["ball"].add_argument(f"--{k}", type=int, required=True)
    sps["build"].add_argument("--mark", nargs="*", help="track=name to label")
    sps["build"].add_argument("--last", type=int, help="frame to end the scene on")
    sps["build"].add_argument("--title")
    sps["build"].add_argument("--note")
    sps["build"].add_argument("--teams", help="names of team 0 and team 1, comma between")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    paths = GamePaths("data", a.season, a.week, a.away, a.home)
    if not paths.film_root_available():
        raise SystemExit(f"film drive for {paths.film_root} is not mounted")
    return cmds[a.cmd](paths, a)


if __name__ == "__main__":
    raise SystemExit(main())
