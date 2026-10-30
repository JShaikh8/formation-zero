"""Spike GRD-95: do SAM 3D Body meshes hold up on All-22 players (small, padded, helmeted)?

Runs in the spike's own environment (setup.sh next to this file), not the project venv:

    PY=.venv-sam3d/bin/python; S=scripts/spikes/sam3d_body/spike.py
    $PY $S pick            # full-resolution frames from labelled sideline and end-zone takes
    $PY $S run             # YOLO11 person boxes -> crops stratified by angle, size, crowding -> SAM 3D Body
    $PY $S sheet           # review sheets: crop | mesh on crop | side | top
    $PY $S grade ggrb...   # one letter per crop in sheet order: g good, r rough, b broken
    $PY $S summary         # grades by angle x size x crowding, seconds per person, cost of a game

The game defaults to the clear game (2026 week 1, SF at LA); --season/--week/--away/--home
override it. Work files go to data/derived/spikes/sam3d_body/<game_key>/. Grades are hand-made
labels, so they go to data/groundtruth/<game_key>.sam3d_body_grades.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Ops MPS lacks fall back to the CPU instead of failing; read when torch initialises.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from formation_zero.data.layout import GamePaths

VIEWS = ("SIDELINE", "ENDZONE")
FRACS = (0.3, 0.7)              # where in a take to grab a frame: near the snap, then live play
MIN_HEIGHT_PX = 20              # smaller boxes are crowd and noise, not players
ISOLATED_MAX = 0.05             # overlap fraction at or below this: nobody touching
CROWDED_MIN = 0.25              # at or above this: a quarter of the box is someone else
HEIGHT_EDGES = (60, 90, 120)    # px buckets for the summary: <60, 60-90, 90-120, 120+
PEOPLE_ON_FIELD = 29            # 22 players and 7 officials
PANEL = 240
GRADES = {"g": "good", "r": "rough", "b": "broken", "x": "not a player"}  # x: off-field staff; left out


def work_dir(paths: GamePaths) -> Path:
    return ROOT / "data" / "derived" / "spikes" / "sam3d_body" / paths.game_key


def grades_path(paths: GamePaths) -> Path:
    return paths.groundtruth_dir / f"{paths.game_key}.sam3d_body_grades.json"


def require_drive(paths: GamePaths) -> None:
    if not paths.film_root_available():
        raise SystemExit(f"film drive for {paths.film_root} is not mounted")


def load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(ROOT / path).convert("RGB"))


# ---------------------------------------------------------------- pick


def cmd_pick(paths: GamePaths, a) -> int:
    """Frames at 30% and 70% through takes spread over the game, per angle, at source resolution."""
    from PIL import Image

    from formation_zero.ingest.sources import FileSource

    require_drive(paths)
    gt = paths.groundtruth_dir / f"{paths.game_key}.shot_views.json"
    labels = [l for l in json.loads(gt.read_text())["labels"]
              if l.get("confident") and l["end_s"] - l["start_s"] >= a.min_take_s]
    wanted = []
    for view in VIEWS:
        takes = [l for l in labels if l["view"] == view]
        if not takes:
            print(f"warning: no labelled {view} takes of {a.min_take_s}s or more")
            continue
        for i in sorted(set(np.linspace(0, len(takes) - 1, min(a.takes, len(takes))).round().astype(int))):
            t = takes[i]
            for f in FRACS:
                wanted.append({"view": view, "take_start_s": t["start_s"], "frac": f,
                               "time_s": t["start_s"] + f * (t["end_s"] - t["start_s"])})

    out = work_dir(paths) / "frames"
    out.mkdir(parents=True, exist_ok=True)
    source = paths.source_dir / "film.mp4"
    frames = []
    with FileSource(source) as src:
        if src.height < 1000:
            print(f"warning: source is {src.width}x{src.height}; the question is about 1080p film")
        for w in sorted(wanted, key=lambda w: w["time_s"]):
            fi = round(w["time_s"] * src.fps)
            fr = next(src.iter_frames(fi, fi + 1), None)
            if fr is None:
                continue
            p = out / f"f{fr.index:06d}.png"
            Image.fromarray(fr.image).save(p)
            frames.append({**w, "frame": fr.index,
                           "path": str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p),
                           "width": fr.width, "height": fr.height})
    (work_dir(paths) / "frames.json").write_text(
        json.dumps({"game_key": paths.game_key, "source": str(source), "frames": frames}, indent=1))
    counts = Counter(f["view"] for f in frames)
    print(f"{len(frames)} frames ({', '.join(f'{v} {n}' for v, n in counts.items())}) -> {out}")
    return 0


# ---------------------------------------------------------------- run: boxes


def overlap_fraction(boxes: np.ndarray) -> np.ndarray:
    """For each box, the largest share of its area covered by any one other box."""
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    np.fill_diagonal(inter, 0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (inter / area[:, None]).max(axis=1)


def on_turf(img: np.ndarray, box, min_green: float = 0.35) -> bool:
    """Feet on green: keeps players and officials, drops the stands and most of the bench.

    Players standing on a painted end zone are dropped too; the spike can live with that.
    """
    import cv2

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    patch = img[int(y2):int(min(H, y2 + max(4, 0.15 * h))),
                int(max(0, x1 - 0.25 * w)):int(min(W, x2 + 0.25 * w))]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2HSV)
    green = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 30)
    return float(green.mean()) >= min_green


def detect(frames: list[dict], weights: Path, device: str, conf: float) -> list[dict]:
    """Every on-field person box in every frame, with its height and how crowded it is."""
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    cands = []
    for f in frames:
        img = load_rgb(f["path"])
        H, W = img.shape[:2]
        r = yolo.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=W, conf=conf, classes=[0],
                         device=device, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        for b, s, o in zip(boxes, scores, overlap_fraction(boxes)):
            h = float(b[3] - b[1])
            edge = b[0] <= 2 or b[1] <= 2 or b[2] >= W - 3 or b[3] >= H - 3
            if h < MIN_HEIGHT_PX or edge or not on_turf(img, b):
                continue
            cands.append({"frame": f["frame"], "view": f["view"],
                          "bbox": [round(float(v), 1) for v in b], "conf": round(float(s), 3),
                          "height_px": round(h, 1), "overlap": round(float(o), 3)})
    return cands


def select(cands: list[dict], per_cell: int, max_per_frame: int) -> list[dict]:
    """Per angle, `per_cell` isolated and `per_cell` crowded people spread evenly over height.

    Middling overlaps are left out so the two groups are clean. At most `max_per_frame` from
    one frame, so one busy frame cannot fill a group.
    """
    chosen: list[dict] = []
    per_frame: Counter = Counter()
    for view in VIEWS:
        for crowded in (False, True):
            pool = sorted((c for c in cands if c["view"] == view and
                           (c["overlap"] >= CROWDED_MIN if crowded else c["overlap"] <= ISOLATED_MAX)),
                          key=lambda c: c["height_px"])
            taken: set[int] = set()
            targets = np.linspace(0, len(pool) - 1, min(per_cell, len(pool))) if pool else []
            for t in targets:
                for i in sorted(range(len(pool)), key=lambda i: abs(i - t)):
                    if i in taken or per_frame[pool[i]["frame"]] >= max_per_frame:
                        continue
                    taken.add(i)
                    per_frame[pool[i]["frame"]] += 1
                    chosen.append({**pool[i], "crowded": crowded})
                    break
    for k, c in enumerate(chosen, 1):
        c["id"] = f"c{k:02d}"
    return chosen


# ---------------------------------------------------------------- run: bodies


def redirect_cuda(device: str) -> None:
    """Upstream hard-codes CUDA on the inference path (the batch transfer in the estimator and
    `.cuda()` on the ray grid); send both to `device`. Body-only inference touches nothing else.
    MPS has no float64, so doubles become floats on the way."""
    import sam_3d_body.models.meta_arch.sam3d_body as arch
    import sam_3d_body.sam_3d_body_estimator as est
    import torch
    from sam_3d_body.utils.dist import recursive_to

    def move(t):
        return (t.float() if device == "mps" and t.dtype == torch.float64 else t).to(device)

    def to_device(x, target):
        if target != "cuda":
            return recursive_to(x, target)
        if isinstance(x, dict):
            return {k: to_device(v, target) for k, v in x.items()}
        if isinstance(x, list):
            return [to_device(v, target) for v in x]
        return move(x) if isinstance(x, torch.Tensor) else x

    est.recursive_to = arch.recursive_to = to_device
    torch.Tensor.cuda = lambda self, *args, **kwargs: move(self)


def load_estimator(ckpt: Path, device: str):
    """Upstream's load_sam_3d_body, with flash attention off (CUDA only) and the device honoured."""
    import torch
    from sam_3d_body import SAM3DBodyEstimator
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.checkpoint import load_state_dict
    from sam_3d_body.utils.config import get_config

    if device != "cuda":
        redirect_cuda(device)
    cfg = get_config(str(ckpt / "model_config.yaml"))
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(ckpt / "assets" / "mhr_model.pt")
    if "FLASH_ATTN" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.FLASH_ATTN = False
    cfg.freeze()
    model = SAM3DBody(cfg)
    state = torch.load(ckpt / "model.ckpt", map_location="cpu", weights_only=False)
    load_state_dict(model, state.get("state_dict", state), strict=False)
    model = model.to(device).eval()
    if device == "mps":
        keep_mhr_on_cpu(model)
    return SAM3DBodyEstimator(model, cfg)


def keep_mhr_on_cpu(model) -> None:
    """The MHR body model (TorchScript) casts to float64 inside, which MPS lacks.

    Run it on CPU and everything else (the ViT-H backbone and decoder, nearly all the compute)
    on MPS. It is stored unregistered so a later `model.to()` cannot move it back.
    """
    import torch

    class OnCpu(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.__dict__["mod"] = mod.cpu()

        def forward(self, *args):
            dev = next(x.device for x in args if isinstance(x, torch.Tensor))
            out = self.mod(*(x.cpu() if isinstance(x, torch.Tensor) else x for x in args))
            return tuple((o.float() if o.dtype == torch.float64 else o).to(dev) for o in out)

    heads = [m for m in model.modules() if isinstance(getattr(m, "mhr", None), torch.jit.ScriptModule)]
    for m in heads:
        m.mhr = OnCpu(m.mhr)


def intrinsics(width: int, height: int, hfov_deg: float | None):
    """None keeps upstream's default (focal = image diagonal, about 47 degrees across at 16:9)."""
    if hfov_deg is None:
        return None
    import torch

    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return torch.tensor([[[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]], dtype=torch.float32)


def project(points_cam: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    """Pinhole projection with the principal point at the image centre, as upstream assumes."""
    z = points_cam[:, 2:3]
    return np.concatenate([focal * points_cam[:, :1] / z + width / 2,
                           focal * points_cam[:, 1:2] / z + height / 2], axis=1)


def focal_of(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def reprojection_px(out: dict, width: int, height: int) -> float:
    """Our projection of the 3D keypoints against the model's own 2D keypoints. Near zero means
    the sheet's mesh overlay uses the same camera the model did."""
    p = project(out["pred_keypoints_3d"] + out["pred_cam_t"], focal_of(out["focal_length"]), width, height)
    return round(float(np.median(np.linalg.norm(p - out["pred_keypoints_2d"][:, :2], axis=1))), 2)


def cmd_run(paths: GamePaths, a) -> int:
    import torch

    require_drive(paths)
    wd = work_dir(paths)
    frames = json.loads((wd / "frames.json").read_text())["frames"]
    models = paths.film_root / "models"
    ckpt = models / a.checkpoint
    if not (ckpt / "model.ckpt").exists():
        raise SystemExit(f"no checkpoint at {ckpt}: run scripts/spikes/sam3d_body/setup.sh")
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    cands = detect(frames, models / "yolo11m.pt", device, a.conf)
    crops = select(cands, a.per_cell, a.max_per_frame)
    if not crops:
        raise SystemExit(f"no usable person boxes in {len(frames)} frames")
    cells = Counter((c["view"], "crowded" if c["crowded"] else "isolated") for c in crops)
    print(f"{len(cands)} on-field people in {len(frames)} frames; {len(crops)} crops: "
          + ", ".join(f"{v} {k} {n}" for (v, k), n in sorted(cells.items())))

    est = load_estimator(ckpt, device)
    fmeta = {f["frame"]: f for f in frames}
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for c in crops:
        by_frame[c["frame"]].append(c)
    out_dir = wd / "crops"
    out_dir.mkdir(exist_ok=True)
    timing = []
    for k, (frame, cs) in enumerate(sorted(by_frame.items())):
        img = load_rgb(fmeta[frame]["path"])
        H, W = img.shape[:2]
        boxes = np.array([c["bbox"] for c in cs], dtype=np.float32)
        cam = intrinsics(W, H, a.hfov)
        if k == 0:  # warm-up, untimed; also where a gap in MPS support shows up
            try:
                est.process_one_image(img, bboxes=boxes[:1], cam_int=cam, inference_type="body")
            except Exception as e:
                if device != "mps":
                    raise
                print(f"MPS failed ({type(e).__name__}: {e}); reloading on CPU")
                del est
                device = "cpu"
                est = load_estimator(ckpt, device)
                est.process_one_image(img, bboxes=boxes[:1], cam_int=cam, inference_type="body")
        t0 = time.perf_counter()
        outs = est.process_one_image(img, bboxes=boxes, cam_int=cam, inference_type="body")
        timing.append({"frame": frame, "people": len(cs), "seconds": round(time.perf_counter() - t0, 3)})
        for c, o in zip(cs, outs):
            np.savez_compressed(out_dir / f"{c['id']}.npz",
                                vertices=o["pred_vertices"].astype(np.float32),
                                keypoints_3d=o["pred_keypoints_3d"], keypoints_2d=o["pred_keypoints_2d"],
                                cam_t=o["pred_cam_t"], focal_length=np.float32(focal_of(o["focal_length"])))
            c["reproj_px"] = reprojection_px(o, W, H)
        print(f"  frame {frame}: {len(cs)} people in {timing[-1]['seconds']:.2f} s")
    np.save(wd / "faces.npy", est.faces)

    import ultralytics

    per_person = sum(t["seconds"] for t in timing) / sum(t["people"] for t in timing)
    reproj = float(np.median([c["reproj_px"] for c in crops]))
    (wd / "crops.json").write_text(json.dumps({
        "game_key": paths.game_key, "checkpoint": a.checkpoint, "device": device, "hfov": a.hfov,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__, "detector": "yolo11m",
        "candidates": len(cands), "seconds_per_person": round(per_person, 3),
        "median_reproj_px": round(reproj, 2), "timing": timing, "crops": crops}, indent=1))
    print(f"{len(crops)} bodies on {device}: {per_person:.2f} s per person; "
          f"keypoint reprojection median {reproj:.2f} px")
    if reproj > 3:
        print("warning: the sheet's projection disagrees with the model's 2D keypoints; "
              "mesh overlays will be offset")
    return 0


# ---------------------------------------------------------------- sheet


def vertex_normals(V: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def splat(xy: np.ndarray, depth: np.ndarray, colors: np.ndarray, size: int, radius: int = 1):
    """Point z-buffer: each pixel takes the nearest vertex that lands on it. Dense enough for a
    ~18k-vertex body at panel size, and needs no OpenGL. Returns (rgb, mask)."""
    offs = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)]
    xs = np.concatenate([np.round(xy[:, 0]).astype(int) + dx for dx, _ in offs])
    ys = np.concatenate([np.round(xy[:, 1]).astype(int) + dy for _, dy in offs])
    zs = np.tile(depth, len(offs))
    cs = np.tile(colors, (len(offs), 1))
    ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    xs, ys, zs, cs = xs[ok], ys[ok], zs[ok], cs[ok]
    order = np.argsort(zs, kind="stable")
    pix, first = np.unique(ys[order] * size + xs[order], return_index=True)
    rgb = np.zeros((size * size, 3), np.uint8)
    mask = np.zeros(size * size, bool)
    rgb[pix] = cs[order][first]
    mask[pix] = True
    return rgb.reshape(size, size, 3), mask.reshape(size, size)


MESH_RGB = np.array([166, 189, 219], dtype=np.float32)
R_SIDE = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)   # 90 degrees about vertical
R_TOP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)   # looking down, far side up


def shaded(normals: np.ndarray) -> np.ndarray:
    """Light from the viewer; abs() so face winding does not matter."""
    return (MESH_RGB * (0.3 + 0.7 * np.abs(normals[:, 2:3]))).clip(0, 255).astype(np.uint8)


def crop_tiles(img: np.ndarray, crop: dict, mesh: dict, faces: np.ndarray) -> np.ndarray:
    """crop | mesh on crop | side | top, one row of square panels."""
    import cv2

    H, W = img.shape[:2]
    x1, y1, x2, y2 = crop["bbox"]
    side = max(x2 - x1, y2 - y1) * 1.7
    x0, y0 = (x1 + x2 - side) / 2, (y1 + y2 - side) / 2
    s = PANEL / side
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    # Nearest-neighbour on purpose: the grader should see how few pixels the model had.
    panel = cv2.warpAffine(img, M, (PANEL, PANEL), flags=cv2.INTER_NEAREST, borderValue=0)

    V = mesh["vertices"] + mesh["cam_t"]
    n = vertex_normals(V, faces)
    uv = (project(V, float(mesh["focal_length"]), W, H) - [x0, y0]) * s
    rgb, mask = splat(uv, V[:, 2], shaded(n), PANEL)
    over = panel.copy()
    over[mask] = (0.35 * panel[mask] + 0.65 * rgb[mask]).astype(np.uint8)

    C = V - V.mean(axis=0)
    scale = 0.85 * PANEL / max(float(np.ptp(C, axis=0).max()), 1e-6)
    views = []
    for R in (R_SIDE, R_TOP):
        P = C @ R.T
        rgb, mask = splat(P[:, :2] * scale + PANEL / 2, P[:, 2], shaded(n @ R.T), PANEL)
        v = np.full((PANEL, PANEL, 3), 38, np.uint8)
        v[mask] = rgb[mask]
        views.append(v)

    bx = [round(t) for t in ((x1 - x0) * s, (y1 - y0) * s, (x2 - x0) * s, (y2 - y0) * s)]
    cv2.rectangle(panel, (bx[0], bx[1]), (bx[2], bx[3]), (80, 220, 80), 1)
    label = (f"{crop['id']} {crop['view'][0]} {crop['height_px']:.0f}px "
             f"{'crowd' if crop['crowded'] else 'iso'}")
    for p, text in ((panel, label), (over, f"reproj {crop.get('reproj_px', 0):.1f}px"),
                    (views[0], "side"), (views[1], "top")):
        cv2.rectangle(p, (0, 0), (8 * len(text) + 6, 16), (0, 0, 0), -1)
        cv2.putText(p, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 230, 80), 1, cv2.LINE_AA)
    row = np.hstack([panel, over, *views])
    row[:, PANEL - 1::PANEL] = 255
    return row


def cmd_sheet(paths: GamePaths, a) -> int:
    from PIL import Image

    wd = work_dir(paths)
    data = json.loads((wd / "crops.json").read_text())
    faces = np.load(wd / "faces.npy")
    fmeta = {f["frame"]: f for f in json.loads((wd / "frames.json").read_text())["frames"]}
    images: dict[int, np.ndarray] = {}
    rows = []
    for c in data["crops"]:
        if c["frame"] not in images:
            images[c["frame"]] = load_rgb(fmeta[c["frame"]]["path"])
        rows.append(crop_tiles(images[c["frame"]], c, np.load(wd / "crops" / f"{c['id']}.npz"), faces))

    out = wd / "sheets"
    out.mkdir(exist_ok=True)
    cols, per_sheet = 2, 16
    blank = np.zeros_like(rows[0])
    written = []
    for k in range(0, len(rows), per_sheet):
        chunk = rows[k:k + per_sheet]
        chunk += [blank] * (-len(chunk) % cols)
        grid = np.vstack([np.hstack(chunk[r:r + cols]) for r in range(0, len(chunk), cols)])
        grid[PANEL - 1::PANEL, :] = 255
        p = out / f"sheet_{k // per_sheet:02d}.png"
        Image.fromarray(grid).save(p)
        written.append(p)
    print(f"{len(written)} sheets, {len(rows)} crops in id order, left to right then down -> {out}")
    return 0


# ---------------------------------------------------------------- grade and summary


def parse_grades(args: list[str], ids: list[str]) -> dict[str, str]:
    """Either one string with a letter per crop in id order ('ggrb...'), or id=letter pairs."""
    if len(args) == 1 and re.fullmatch(r"[grbx]+", args[0]):
        if len(args[0]) != len(ids):
            raise SystemExit(f"{len(args[0])} letters for {len(ids)} crops")
        return dict(zip(ids, args[0]))
    grades = {}
    for arg in args:
        m = re.fullmatch(r"(c\d+)=([grbx])", arg)
        if not m or m.group(1) not in ids:
            raise SystemExit(f"bad grade {arg!r}: use c07=g, or one letter (g/r/b/x) per crop")
        grades[m.group(1)] = m.group(2)
    return grades


def cmd_grade(paths: GamePaths, a) -> int:
    crops = json.loads((work_dir(paths) / "crops.json").read_text())["crops"]
    new = parse_grades(a.grades, [c["id"] for c in crops])
    out = grades_path(paths)
    letter = {word: k for k, word in GRADES.items()}
    old = ({g["id"]: letter[g["grade"]] for g in json.loads(out.read_text())["grades"]}
           if out.exists() else {})
    merged = {**old, **new}
    keep = ("id", "frame", "view", "bbox", "height_px", "overlap", "crowded")
    rows = [{**{k: c[k] for k in keep}, "grade": GRADES[merged[c["id"]]]}
            for c in crops if c["id"] in merged]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"game_key": paths.game_key, "scale": GRADES, "grades": rows}, indent=1))
    print(f"{len(new)} graded, {len(rows)}/{len(crops)} total -> {out}")
    return 0


def bucket(height_px: float) -> str:
    lo = 0
    for hi in HEIGHT_EDGES:
        if height_px < hi:
            return f"{lo}-{hi}" if lo else f"<{hi}"
        lo = hi
    return f"{lo}+"


def cmd_summary(paths: GamePaths, a) -> int:
    data = json.loads((work_dir(paths) / "crops.json").read_text())
    gp = grades_path(paths)
    graded = {g["id"]: g["grade"] for g in json.loads(gp.read_text())["grades"]} if gp.exists() else {}
    table: dict[tuple, Counter] = defaultdict(Counter)
    for c in data["crops"]:
        key = (c["view"], "crowded" if c["crowded"] else "isolated", bucket(c["height_px"]))
        table[key][graded.get(c["id"], "ungraded")] += 1
    order = {b: i for i, b in enumerate([bucket(e - 1) for e in HEIGHT_EDGES] + [bucket(1e9)])}
    print(f"{'angle':9} {'context':9} {'height':7} {'n':>3} {'good':>5} {'rough':>6} {'broken':>7} {'x':>3} {'ungraded':>9}")
    for (view, ctx, b), n in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]])):
        print(f"{view:9} {ctx:9} {b:7} {sum(n.values()):>3} {n['good']:>5} {n['rough']:>6} "
              f"{n['broken']:>7} {n['not a player']:>3} {n['ungraded']:>9}")
    spp = data["seconds_per_person"]
    per_play = 2 * a.keyframes * PEOPLE_ON_FIELD * spp
    print(f"\n{spp:.2f} s per person on {data['device']} ({data['checkpoint']}). "
          f"{a.keyframes} keyframes x 2 angles x {PEOPLE_ON_FIELD} people = {per_play / 60:.1f} min per play, "
          f"{a.plays * per_play / 3600:.1f} h for {a.plays} plays.")
    if len(graded) < len(data["crops"]):
        print(f"{len(data['crops']) - len(graded)} crops not graded yet: spike.py grade")
    return 0


# ---------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"pick": cmd_pick, "run": cmd_run, "sheet": cmd_sheet, "grade": cmd_grade, "summary": cmd_summary}
    sps = {name: sub.add_parser(name) for name in cmds}
    for sp in sps.values():
        sp.add_argument("--season", type=int, default=2026)
        sp.add_argument("--week", type=int, default=1)
        sp.add_argument("--away", default="SF")
        sp.add_argument("--home", default="LA")
    sps["pick"].add_argument("--takes", type=int, default=6, help="takes per angle (two frames each)")
    sps["pick"].add_argument("--min-take-s", type=float, default=6.0)
    sps["run"].add_argument("--device", choices=["mps", "cpu", "cuda"])
    sps["run"].add_argument("--checkpoint", default="sam-3d-body-vith",
                            help="folder under <film root>/models")
    sps["run"].add_argument("--hfov", type=float,
                            help="horizontal field of view in degrees; default is upstream's (~47)")
    sps["run"].add_argument("--conf", type=float, default=0.35)
    sps["run"].add_argument("--per-cell", type=int, default=8,
                            help="crops per angle x isolated/crowded group")
    sps["run"].add_argument("--max-per-frame", type=int, default=3)
    sps["grade"].add_argument("grades", nargs="+")
    sps["summary"].add_argument("--keyframes", type=int, default=6, help="per play per angle")
    sps["summary"].add_argument("--plays", type=int, default=141, help="filmable plays in a game")
    a = ap.parse_args(argv)
    # GamePaths reads FZ_FILM_ROOT from the repo's .env relative to the working directory.
    os.chdir(ROOT)
    paths = GamePaths("data", a.season, a.week, a.away, a.home)
    return cmds[a.cmd](paths, a)


if __name__ == "__main__":
    raise SystemExit(main())
