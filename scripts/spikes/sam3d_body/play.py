"""One whole play with a SAM 3D Body on every player on every frame: do the bodies hold steady?

The spike next to this file graded still crops. This runs the same model down a tracked take, from
before the snap to the end, on the 1080p film. Runs in the spike's environment:

    PY=.venv-sam3d/bin/python; S=scripts/spikes/sam3d_body/play.py
    $PY $S focal --play 40                     # the lens through the take, from the painted lines
    $PY $S fit --play 40 --first 35878         # resumable: one file a frame on the drive
    $PY $S video --play 40                     # bodies drawn over the film
    $PY $S steady --play 40 --snap 35938 --dead 36048     # the numbers in the README

Fits go to <game>/bodies/p<play>.<angle>/ on the film drive, and the video beside them. The game
defaults to Chargers at Chiefs, the one tracked with the round-1 student.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from spike import ROOT, focal_of, load_estimator, project, splat, vertex_normals  # noqa: E402

sys.path.insert(0, str(ROOT))

from formation_zero.data.layout import GamePaths  # noqa: E402
from formation_zero.perception.track.stitch import WIDE  # noqa: E402

ANGLE = "sideline"
BATCH = 8                   # people through the model at once
MIN_TRACK = 5               # frames: a blip is not worth a body that pops in and out
FOCAL_EVERY = 10            # frames between lens readings
TEAM_RGB = {0: (255, 214, 90), 1: (235, 70, 70)}
ALPHA = 0.75
FLIP = 60.0                 # degrees of hip turn in one frame
GIANT = 0.8                 # distance from the camera over its track's running median: 12 of 12 random fits below it wrong
YD_M = 0.9144


def take_dir(paths: GamePaths, play: int) -> Path:
    return paths.raw_dir / "bodies" / f"p{play:03d}.{ANGLE}"


def stem(paths: GamePaths, play: int) -> str:
    return f"{paths.game_key}_p{play:03d}.{ANGLE}"


def people(paths: GamePaths, play: int, first: int | None) -> pd.DataFrame:
    """The take's players on the field: not officials, not a track with no team, not a track far wider
    than tall (the painted shield), not a blip."""
    b = pd.read_parquet(paths.tracking_dir / f"{stem(paths, play)}.boxes.parquet")
    b = b[(b.label == "player") & ~b.off_field.fillna(False).astype(bool) & b.team.isin((0, 1))].copy()
    if first is not None:
        b = b[b.frame >= first]
    b["filled"] = b.filled.fillna(False).astype(bool)
    shape = ((b.x2 - b.x1) / (b.y2 - b.y1)).groupby(b.track_id).median()
    size = b.groupby("track_id").frame.nunique()
    keep = shape.index[(shape < WIDE) & (size >= MIN_TRACK)]
    return b[b.track_id.isin(keep)].reset_index(drop=True)


def cmd_focal(paths: GamePaths, a) -> int:
    """The camera zooms during a take, so the lens is read from the painted lines every FOCAL_EVERY
    frames and held between readings; frames that do not register are bridged."""
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source
    from formation_zero.perception.field import homography as hg

    b = people(paths, a.play, a.first)
    span = int(b.frame.min()), int(b.frame.max()) + 1
    out = {}
    with FileSource(best_source(paths)) as source:
        for fr in source.iter_frames(*span, step=FOCAL_EVERY):
            fit = hg.register(fr.image, ANGLE)["fit"]
            if fit is not None and hg.plausible(fit["camera"], None):
                out[fr.index] = float(fit["camera"]["f"])
    d = take_dir(paths, a.play)
    d.mkdir(parents=True, exist_ok=True)
    (d / "focal.json").write_text(json.dumps({str(k): v for k, v in out.items()}))
    f = np.array(list(out.values()))
    print(f"{len(f)} of {(span[1] - span[0]) // FOCAL_EVERY + 1} readings: focal {f.min():.0f} to {f.max():.0f} px, "
          f"upstream's default would be {np.hypot(source.width, source.height):.0f}")
    return 0


def lens(paths: GamePaths, play: int):
    """frame -> focal px, a running median through the readings so one bad registration cannot show."""
    got = json.loads((take_dir(paths, play) / "focal.json").read_text())
    x = np.array(sorted(map(int, got)))
    y = np.array([got[str(k)] for k in x])
    y = np.array([np.median(y[max(0, i - 2):i + 3]) for i in range(len(y))])
    return lambda f: float(np.interp(f, x, y))


def cmd_fit(paths: GamePaths, a) -> int:
    import torch

    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source

    ckpt = paths.film_root / "models" / "sam-3d-body-vith"
    b = people(paths, a.play, a.first)
    frames = sorted(int(f) for f in b.frame.unique())
    if a.every > 1:
        frames = frames[::a.every]
    if a.limit:
        frames = frames[:a.limit]
    d = take_dir(paths, a.play)
    (d / "fits").mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames if not (d / "fits" / f"{f}.npz").exists()]
    print(f"{len(b[b.frame.isin(frames)])} boxes on {len(frames)} frames, {len(todo)} frames to do", flush=True)
    if not todo:
        return 0
    focal = lens(paths, a.play) if a.focal == "real" else None
    est = load_estimator(ckpt, a.device)
    np.save(d / "faces.npy", est.faces)
    by_frame = dict(tuple(b.groupby("frame")))
    wanted, t0, fits = set(todo), time.perf_counter(), 0
    with FileSource(best_source(paths)) as source:
        W, H = source.width, source.height
        for fr in source.iter_frames(todo[0], todo[-1] + 1):
            if fr.index not in wanted:
                continue
            g = by_frame[fr.index]
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            cam = None
            if focal is not None:
                f = focal(fr.index)
                cam = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]], dtype=torch.float32)
            outs = []
            for i in range(0, len(boxes), a.batch):
                outs += est.process_one_image(fr.image, bboxes=boxes[i:i + a.batch], cam_int=cam, inference_type="body")
            np.savez_compressed(
                d / "fits" / f"{fr.index}.npz",
                track=g.track_id.to_numpy(np.int32), team=g.team.to_numpy(np.int8), filled=g.filled.to_numpy(),
                box=boxes, focal=np.float32(focal_of(outs[0]["focal_length"])),
                cam_t=np.stack([o["pred_cam_t"] for o in outs]).astype(np.float32),
                keypoints=np.stack([o["pred_keypoints_3d"] for o in outs]).astype(np.float32),
                vertices=np.stack([np.round(o["pred_vertices"] * 1000) for o in outs]).astype(np.int16))   # mm
            fits += len(outs)
            s = time.perf_counter() - t0
            print(f"frame {fr.index}: {len(outs)} people, {s / fits:.2f} s a body, "
                  f"{s / 60:.1f} min so far, {fits / max(len(b[b.frame.isin(frames)]), 1):.0%}", flush=True)
    return 0


def body_layer(V: np.ndarray, faces: np.ndarray, focal: float, W: int, H: int, rgb: tuple[int, int, int]):
    """One body's pixels: (x0, y0, colours, mask) of its own small canvas, or None off screen."""
    xy = project(V, focal, W, H)
    lo = np.floor(xy.min(axis=0)).astype(int) - 2
    hi = np.ceil(xy.max(axis=0)).astype(int) + 2
    size = int(max(hi - lo)) + 1
    if hi[0] < 0 or hi[1] < 0 or lo[0] >= W or lo[1] >= H or size > 900:
        return None
    n = vertex_normals(V, faces)
    light = 0.35 + 0.65 * np.abs(n[:, 2:3])
    colours = (np.array(rgb, np.float32) * light).clip(0, 255).astype(np.uint8)
    canvas, mask = splat(xy - lo, V[:, 2], colours, size)
    return int(lo[0]), int(lo[1]), canvas, mask


def paint(img: np.ndarray, z: dict, faces: np.ndarray, W: int, H: int) -> np.ndarray:
    """Bodies far to near by their feet on screen: each fit has its own depth, so depths between bodies
    mean little. A filled box is an estimate, so it is drawn fainter."""
    out = img.copy()
    for k in np.argsort(z["box"][:, 3]):
        V = z["vertices"][k].astype(np.float32) / 1000 + z["cam_t"][k]
        layer = body_layer(V, faces, float(z["focal"]), W, H, TEAM_RGB[int(z["team"][k])])
        if layer is None:
            continue
        x0, y0, canvas, mask = layer
        ax, ay = max(x0, 0), max(y0, 0)
        bx, by = min(x0 + mask.shape[1], W), min(y0 + mask.shape[0], H)
        m = mask[ay - y0:by - y0, ax - x0:bx - x0]
        c = canvas[ay - y0:by - y0, ax - x0:bx - x0]
        alpha = 0.4 if z["filled"][k] else ALPHA
        roi = out[ay:by, ax:bx]
        roi[m] = ((1 - alpha) * roi[m] + alpha * c[m]).astype(np.uint8)
    return out


def cmd_video(paths: GamePaths, a) -> int:
    from formation_zero.ingest.encode import VideoWriter
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source

    d = take_dir(paths, a.play)
    faces = np.load(d / "faces.npy")
    frames = sorted(int(p.stem) for p in (d / "fits").glob("*.npz"))
    if a.only:
        frames = [f for f in frames if f in set(a.only)]
    dst = paths.raw_dir / "bodies" / f"p{a.play:03d}.{ANGLE}.mp4"
    with FileSource(best_source(paths)) as source:
        W, H = source.width, source.height
        want = set(frames)
        if a.only:
            for fr in source.iter_frames(frames[0], frames[-1] + 1):
                if fr.index in want:
                    z = np.load(d / "fits" / f"{fr.index}.npz")
                    from PIL import Image
                    Image.fromarray(paint(fr.image, z, faces, W, H)).save(d / f"still_{fr.index}.jpg", quality=92)
            return 0
        with VideoWriter(dst, source.fps, W, H) as out:
            for fr in source.iter_frames(frames[0], frames[-1] + 1):
                if fr.index in want:
                    out.write(paint(fr.image, np.load(d / "fits" / f"{fr.index}.npz"), faces, W, H))
    print(f"{len(frames)} frames -> {dst}")
    return 0


def real_range(paths: GamePaths, t: pd.DataFrame, every: int = 20) -> np.ndarray:
    """Each body's distance from the camera over its real one, from the camera the painted lines give."""
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source
    from formation_zero.perception.field import homography as hg

    out = []
    with FileSource(best_source(paths)) as source:
        for fr in source.iter_frames(int(t.frame.min()), int(t.frame.max()) + 1, step=every):
            fit = hg.register(fr.image, ANGLE)["fit"]
            if fit is None or not hg.plausible(fit["camera"], None):
                continue
            g = t[t.frame == fr.index]
            yd = hg.apply(fit["G"], g[["foot_x", "foot_y"]].to_numpy(float))
            real = np.linalg.norm(np.c_[yd, np.zeros(len(yd))] - np.asarray(fit["camera"]["C"]), axis=1) * YD_M
            out.append(g["range"].to_numpy() / real)
    return np.concatenate(out)


def cmd_steady(paths: GamePaths, a) -> int:
    """How much a body changes from one frame to the next, with the player's own travel taken out:
    joints about the pelvis (a set player barely moves, so before the snap this is the noise), the way
    the hips face (a turn past FLIP in one frame is the fit flipping, not the player), and bone lengths,
    which no pose can change. A body drawn too big comes out too near the camera, so a fit well short of
    its track's running distance is a giant. Box height is no test for that: a crouched lineman's box
    often holds only his back, and the body rightly runs past it."""
    d = take_dir(paths, a.play)
    rows = []
    for p in sorted((d / "fits").glob("*.npz"), key=lambda q: int(q.stem)):
        z = np.load(p)
        for i, t in enumerate(z["track"]):
            k = z["keypoints"][i]
            pelvis = (k[9] + k[10]) / 2
            hips = k[9] - k[10]
            x1, y1, x2, y2 = z["box"][i]
            rows.append({"frame": int(p.stem), "track": int(t), "filled": bool(z["filled"][i]), "k": k[:21] - pelvis,
                         "face": np.degrees(np.arctan2(hips[2], hips[0])),
                         "thigh": float(np.linalg.norm(k[[9, 10]] - k[[11, 12]], axis=1).mean()),
                         "shin": float(np.linalg.norm(k[[11, 12]] - k[[13, 14]], axis=1).mean()),
                         "range": float(np.linalg.norm(z["cam_t"][i])), "box_h": float(y2 - y1),
                         "foot_x": float((x1 + x2) / 2), "foot_y": float(y2)})
    t = pd.DataFrame(rows).sort_values(["track", "frame"]).reset_index(drop=True)
    t["step"], t["turn"] = np.nan, np.nan
    for _, g in t.groupby("track"):
        same = np.diff(g.frame.to_numpy()) == 1
        k = np.stack(g.k.to_numpy())
        t.loc[g.index[1:][same], "step"] = np.linalg.norm(np.diff(k, axis=0), axis=2).mean(axis=1)[same] * 100
        t.loc[g.index[1:][same], "turn"] = np.abs((np.diff(g.face.to_numpy()) + 180) % 360 - 180)[same]
    t["flip"] = t.turn > FLIP
    run = t.groupby("track")["range"].transform(lambda s: s.rolling(51, center=True, min_periods=8).median())
    t["giant"] = t["range"] / run < GIANT
    t["phase"] = np.select([t.frame < a.snap, t.frame < a.dead], ["before the snap", "in play"], "after")
    print(f"{len(t)} fits, {t.track.nunique()} tracks, {t.filled.sum()} on filled boxes")
    for phase in ("before the snap", "in play", "after"):
        g = t[t.phase == phase]
        print(f"{phase}: joints a frame median {g.step.median():.1f} cm, p90 {g.step.quantile(.9):.1f}; flips past {FLIP:.0f} deg "
              f"{g.flip.sum()} of {g.turn.notna().sum()}; giants {g.giant.sum()} of {len(g)}")
    live = t[(t.phase == "in play") & t.turn.notna()]
    print("flips in play by box height:", ", ".join(
        f"{lo}-{hi} px {g.flip.mean():.1%} of {len(g)}" for (lo, hi) in ((0, 55), (55, 70), (70, 85), (85, 110), (110, 999))
        for g in [live[(live.box_h > lo) & (live.box_h <= hi)]]))
    print(f"median box height: before the snap {t[t.phase == 'before the snap'].box_h.median():.0f} px, "
          f"in play {t[t.phase == 'in play'].box_h.median():.0f}")
    print(f"thigh {t.thigh.median():.3f} m, shin {t.shin.median():.3f} m; within a track, std/median "
          f"{(t.groupby('track').thigh.std() / t.groupby('track').thigh.median()).median():.3f}, "
          f"{(t.groupby('track').shin.std() / t.groupby('track').shin.median()).median():.3f}")
    r = real_range(paths, t)
    print(f"distance from the camera, model over real, {len(r)} bodies: median {np.median(r):.2f}, "
          f"p5 {np.percentile(r, 5):.2f}, p95 {np.percentile(r, 95):.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"focal": cmd_focal, "fit": cmd_fit, "video": cmd_video, "steady": cmd_steady}
    sps = {name: sub.add_parser(name) for name in cmds}
    for sp in sps.values():
        sp.add_argument("--season", type=int, default=2026)
        sp.add_argument("--week", type=int, default=6)
        sp.add_argument("--away", default="LAC")
        sp.add_argument("--home", default="KC")
        sp.add_argument("--play", type=int, required=True)
        sp.add_argument("--first", type=int, help="first frame; default the start of the take")
    sps["fit"].add_argument("--device", default="mps")
    sps["fit"].add_argument("--batch", type=int, default=BATCH)
    sps["fit"].add_argument("--limit", type=int, help="stop after this many frames")
    sps["fit"].add_argument("--every", type=int, default=1, help="every nth frame")
    sps["fit"].add_argument("--focal", choices=["real", "default"], default="real")
    sps["video"].add_argument("--only", type=int, nargs="+", help="write stills of these frames, no video")
    sps["steady"].add_argument("--snap", type=int, required=True, help="the snap's frame")
    sps["steady"].add_argument("--dead", type=int, required=True, help="the frame the play is over")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    paths = GamePaths("data", a.season, a.week, a.away, a.home)
    if not paths.film_root_available():
        raise SystemExit(f"film drive for {paths.film_root} is not mounted")
    return cmds[a.cmd](paths, a)


if __name__ == "__main__":
    raise SystemExit(main())
