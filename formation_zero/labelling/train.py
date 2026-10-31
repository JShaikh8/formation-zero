"""Train the student detector on approved labels, then find where it and the teacher disagree (GRD-31).

Architecture 7.1, step 4: fine-tune a small detector on the frames a person approved, run it on
teacher-labelled frames nobody reviewed, and send the frames where the two disagree most to the
next review sheet. Two or three rounds is the expectation.

    fz-train export --games 2025:20:LA:CHI 2026:1:SF:LA                       # YOLO dataset from approved labels
    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.labelling.train fit     # fine-tune (needs ultralytics)
    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.labelling.train disagree --games 2025:20:LA:CHI

Round 0, before any review (fit with `--base <film root>/models/yolo/yolo11s.pt` to keep
downloads off the laptop's disk):

    fz-train export --source teacher --fold 0                                  # and --fold 1
    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.labelling.train fit --source teacher --fold 0
    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.labelling.train heldout

Splits are by play, never by frame: two frames of one play are near copies, and one in training
with the other in validation would flatter every number.

The student learns players and referees, not the ball. The teacher finds the ball in about one
frame in eight (GRD-29), so most training frames would show a ball with no box, which teaches the
detector that balls are background. The ball gets its own model from the frames where it is
labelled (section 7.1).

Round 0 (`--source teacher`) trains on the teacher's boxes before anyone has reviewed them, in two
folds by play (`--fold 0`, `--fold 1`): each student learns half the plays and scores the half it
never saw, so every frame gets an honest disagreement score (`heldout`), which orders the review
page with the teacher's likely mistakes first (`fz-review build --order`). Two half-size runs cost
about one full one, and either student serves the tracking work.

Round 1 (2026-10-31): the 371 frames the owner kept, in the same two folds, so neither round's
student has seen a play it is scored on. Continuing from round 0 for 20 epochs beat a fresh COCO
start at 40 on both folds and both classes (player mAP50 0.958 against 0.954, referee 0.871 against
0.860), so rounds continue. Against round 0 on those frames player mAP50 went from 0.837 to 0.958,
referee from 0.677 to 0.871, but those frames were picked because round 0 disputed them (median 8 and
17 disputed boxes against 2 and 4 on the rest), so the gain is flattered. Player recall did not rise
(0.917 to 0.911 snow, 0.956 to 0.945 clear): round 1 learned what not to box.

Frames nobody chose say the same thing more soberly. On 101 random unreviewed sideline frames of
49ers-Rams that register, boxes with feet off the field fell from 4.2 players and 4.1 referees a
frame to 0.1 and 0.4, while referees on the field held (1.6 to 1.4); frames with 20-24 players on
the field went from 58% to 65%. Most of the referees lost were chain crew and staff
(`labelling.field_filter`). On the snow game, 148 random sideline frames did not move (64% to 65%).

The extra boxes left are mostly stacks on linemen at the line, which a stricter merge settles without
any labelling: NMS IoU 0.7 to 0.5 (NMS_IOU) takes overlapping pairs from 2.93 a frame to 0.61, where
approved labels have 0.65, and player precision from 0.836 to 0.897 for recall 0.927 to 0.917.

Is this many frames enough? Half the training plays (98 and 113 frames, double the epochs, so the
same steps), scored on the same frames with NMS 0.5: player mAP50 0.950 to 0.956 for twice the frames,
mAP50-95 0.722 to 0.737, referee mAP50 0.850 to 0.871. For players in these two games the curve is
nearly flat. Disagreements left at the top of the next sheet are mostly piles and tight zooms. A third
game is unmeasured. `runs/student` is the round-1 student on all 371 frames (the fold-0 export's train
and val together, from round 0 fold 0, 20 epochs), so the validation it prints is on training frames.

Round 1 in tracking (plays 1-60 of 49ers-Rams, each play tracked by the fold student that never saw
it, NMS 0.5, against the round-0 run): sideline tracks 5,584 to 3,144, tracks off the field 1,304 to
16, fronts right at the snap 52 to 54 of 56, anchor error p90 0.44 to 0.36 yd. The price is about 0.6
fewer players a frame before the snap: exactly 22 on 33% of frames against 43%, 21-23 on 75% against
85%. Where the boxes differ, the ones lost are real linemen buried in the clump. Tracking stays on
round 0 until the Chargers-Chiefs test frames can say how many are really missed.

The first game it never saw (Chargers-Chiefs, 2026-10-31, no training on it): on 300 random frames
the student at NMS 0.5 boxes no one extra on the sideline camera (over 26 players, 15% to 0%), and
frames with 20-24 players go from 88% to 94% before the snap and 69% to 83% in the play. After the
whistle they fall from 28% to 22%, unexplained until the test frames are in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from formation_zero.perception.detect.yolo import NMS_IOU

CLASSES = ("player", "referee")
VAL_FRACTION = 0.15
DISAGREE_IOU = 0.5
DATASET = Path("data/derived/labels/student")
DATASETS = {"approved": DATASET, "teacher": Path("data/derived/labels/student_round0")}
RUN_NAMES = {"approved": "student", "teacher": "student_round0"}


def val_play(play_uid: str, fraction: float = VAL_FRACTION) -> bool:
    """A stable split by play: the same play is always on the same side, whatever else changes."""
    return int(hashlib.sha1(play_uid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def fold_of(play_uid: str, folds: int = 2) -> int:
    """A play's fold, from other bits of the same hash, so folds and the validation split are independent."""
    return int(hashlib.sha1(play_uid.encode()).hexdigest()[8:16], 16) % folds


def yolo_lines(boxes: list[dict], width: int, height: int) -> list[str]:
    """Boxes in pixels -> YOLO label lines (class cx cy w h, normalised), clipped to the frame."""
    out = []
    for b in boxes:
        if b["label"] not in CLASSES:
            continue
        x1, y1, x2, y2 = (max(0.0, min(v, lim)) for v, lim in zip(b["box"], (width, height, width, height)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(f"{CLASSES.index(b['label'])} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                   f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return out


def export_dataset(games, out_dir: Path = DATASET, *, source: str = "approved", fold: int | None = None) -> dict:
    """Labelled frames of each game -> <out_dir>/{images,labels}/{train,val}/ and data.yaml.

    `source` is "approved" (the kept frames of the human review) or "teacher" (every frame the
    teacher labelled, unreviewed: round 0). With `fold`, that fold's plays are the validation set
    and the rest train; otherwise VAL_FRACTION of plays validate. Images are symlinks to the
    extracted frames on the film drive. `games` is a list of GamePaths.
    """
    from PIL import Image

    counts = {"train": 0, "val": 0}
    for split in counts:
        for kind in ("images", "labels"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
    for paths in games:
        rows = labelled_frames(paths, source)
        if rows is None:
            continue
        play_of = {s["frame"]: s["play_uid"] for s in json.loads(paths.samples_path.read_text())["samples"]}
        for row in rows:
            frame = row["frame"]
            uid = play_of[frame]
            split = "val" if (val_play(uid) if fold is None else fold_of(uid) == fold) else "train"
            src = paths.frames_dir / f"{frame:06d}.jpg"
            name = f"{paths.game_key}_{frame:06d}"
            with Image.open(src) as im:
                w, h = im.size
            link = out_dir / "images" / split / f"{name}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            (out_dir / "labels" / split / f"{name}.txt").write_text("\n".join(yolo_lines(row["boxes"], w, h)) + "\n")
            counts[split] += 1
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return counts


def labelled_frames(paths, source: str) -> list[dict] | None:
    """[{"frame", "boxes"}] from the review's kept frames or the teacher's file; None if not there yet."""
    if source == "approved":
        f = paths.groundtruth_dir / f"{paths.game_key}.detections.json"
        return json.loads(f.read_text())["kept"] if f.exists() else None
    if source == "teacher":
        f = paths.labels_dir / f"{paths.game_key}.teacher_sam3.json"
        return [{"frame": int(k), "boxes": v} for k, v in json.loads(f.read_text())["frames"].items()] if f.exists() else None
    raise ValueError(f"unknown source {source!r}")


def disagreement(teacher: list[dict], student: list[dict], iou_min: float = DISAGREE_IOU) -> int:
    """Boxes one model has and the other lacks, matched greedily by IoU within each class."""
    from formation_zero.labelling.teacher_sam3 import iou

    missing = 0
    for label in CLASSES:
        t = [np.array(b["box"], float) for b in teacher if b["label"] == label]
        s = [np.array(b["box"], float) for b in student if b["label"] == label]
        free = list(range(len(s)))
        for tb in t:
            if free:
                scores = iou(tb, np.array([s[i] for i in free]))
                k = int(np.argmax(scores))
                if scores[k] >= iou_min:
                    free.pop(k)
                    continue
            missing += 1
        missing += len(free)
    return missing


def fit(data_yaml: Path, *, base: str = "yolo11s.pt", epochs: int = 60, imgsz: int = 1280, batch: int = 8,
        val: bool = True, device: str = "mps", name: str = "student", project: Path = Path("data/derived/labels/runs")):
    """Fine-tune a COCO-pretrained YOLO11. Large input: far-side players are small, but on the 16 GB
    laptop 1280 px at batch 8 swapped itself to a standstill (2026-10-30); 960 at batch 4 fits.
    `val=False` validates only after the last epoch (a round-0 fold validates on half the frames)."""
    from ultralytics import YOLO

    model = YOLO(base)
    return model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, val=val, device=device,
                       project=str(project.resolve()), name=name, exist_ok=True, plots=False)


def weights_of(run: Path) -> Path:
    """A run's best.pt, or last.pt when it validated only at the end and kept no best."""
    best = run / "weights" / "best.pt"
    return best if best.exists() else run / "weights" / "last.pt"


def score_frames(paths, weights: Path, frames: list[str], *, imgsz: int = 1280, device: str = "mps",
                 removed: dict[str, list[int]] | None = None) -> dict[str, int]:
    """Teacher-student disagreement on each of `frames` (keys of the teacher's file). `removed` is
    {frame: teacher box indices} to leave out, the off-field boxes a reviewed student rightly lacks."""
    from ultralytics import YOLO

    teacher = json.loads((paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").read_text())["frames"]
    gone = removed or {}
    model = YOLO(str(weights))
    out = {}
    for f in frames:
        r = model.predict(str(paths.frames_dir / f"{int(f):06d}.jpg"), imgsz=imgsz, iou=NMS_IOU, device=device, verbose=False)[0]
        sboxes = [{"label": CLASSES[int(c)], "box": b.tolist()} for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]
        out[f] = disagreement([b for i, b in enumerate(teacher[f]) if i not in gone.get(f, ())], sboxes)
    return out


def heldout(paths, weights_by_fold: dict[int, Path], **kw) -> Path:
    """Round 0: each fold's student scores the frames of the plays it never saw. Writes
    <game>.round0_disagreement.json, {frame: unmatched boxes}, for `fz-review build --order`."""
    teacher = json.loads((paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").read_text())["frames"]
    play_of = {str(s["frame"]): s["play_uid"] for s in json.loads(paths.samples_path.read_text())["samples"]}
    scores: dict[str, int] = {}
    for k, w in weights_by_fold.items():
        scores |= score_frames(paths, w, [f for f in teacher if fold_of(play_of[f], len(weights_by_fold)) == k], **kw)
    out = paths.labels_dir / f"{paths.game_key}.round0_disagreement.json"
    out.write_text(json.dumps({"scores": scores, "folds": {k: str(w) for k, w in weights_by_fold.items()}}))
    return out


def next_sheet(paths, weights: Path, *, top: int = 300, imgsz: int = 1280, device: str = "mps") -> Path:
    """Run the student on teacher-labelled frames nobody reviewed; the worst disagreements become
    the next review sheet's detections file (teacher boxes, ordered for review by `labelling.review`).
    Boxes `labelling.field_filter` calls off the field don't count: a student that learned the
    review no longer finds benches, and without this they would fill the sheet."""
    teacher = json.loads((paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").read_text())
    approved = paths.groundtruth_dir / f"{paths.game_key}.detections.json"
    seen = set()
    if approved.exists():
        a = json.loads(approved.read_text())
        seen = {r["frame"] for r in a["kept"]} | set(a["flagged"]) | set(a.get("dropped", []))
    suggestions = paths.labels_dir / f"{paths.game_key}.field_suggestions.json"
    removed = json.loads(suggestions.read_text())["suggestions"] if suggestions.exists() else None
    todo = [f for f in teacher["frames"] if int(f) not in seen]
    scored = sorted(((d, f) for f, d in score_frames(paths, weights, todo, imgsz=imgsz, device=device, removed=removed).items()),
                    reverse=True)
    out = paths.labels_dir / f"{paths.game_key}.round_next.json"
    out.write_text(json.dumps({"teacher": teacher["teacher"] + " (student disagreements)",
                               "frames": {f: teacher["frames"][f] for d, f in scored[:top] if d > 0}}))
    return out


def _games(specs: list[str], data_root: str):
    from formation_zero.data.layout import GamePaths

    out = []
    for spec in specs:
        season, week, away, home = spec.split(":")
        out.append(GamePaths(data_root, int(season), int(week), away, home))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["export", "fit", "disagree", "heldout"])
    ap.add_argument("--games", nargs="+", default=["2025:20:LA:CHI", "2026:1:SF:LA"], help="season:week:away:home")
    ap.add_argument("--source", choices=sorted(DATASETS), default="approved",
                    help="approved review labels, or the unreviewed teacher boxes (round 0)")
    ap.add_argument("--base", default="yolo11s.pt", help="starting weights (fit)")
    ap.add_argument("--weights", help="student weights (disagree); default: this source's best.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fold", type=int, choices=[0, 1], help="round 0: train on the other fold's plays")
    ap.add_argument("--no-val", action="store_true", help="validate only after the last epoch (fit)")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    out, name = DATASETS[a.source], RUN_NAMES[a.source]
    if a.fold is not None:
        out, name = out.with_name(f"{out.name}_f{a.fold}"), f"{name}_f{a.fold}"
    if a.action == "export":
        print(f"{export_dataset(_games(a.games, a.data_root), out, source=a.source, fold=a.fold)} frames -> {out}")
    elif a.action == "fit":
        fit(out / "data.yaml", base=a.base, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, val=not a.no_val, name=name)
    elif a.action == "heldout":
        runs = Path("data/derived/labels/runs")
        weights = {k: weights_of(runs / f"{RUN_NAMES['teacher']}_f{k}") for k in (0, 1)}
        for paths in _games(a.games, a.data_root):
            print(f"held-out disagreement -> {heldout(paths, weights, imgsz=a.imgsz)}")
    else:
        weights = Path(a.weights or f"data/derived/labels/runs/{name}/weights/best.pt")
        for paths in _games(a.games, a.data_root):
            print(f"next review sheet -> {next_sheet(paths, weights, imgsz=a.imgsz)}")
    return 0


__all__ = ["CLASSES", "disagreement", "export_dataset", "fit", "fold_of", "heldout", "labelled_frames", "next_sheet",
           "score_frames", "val_play", "weights_of", "yolo_lines"]


if __name__ == "__main__":
    raise SystemExit(main())
