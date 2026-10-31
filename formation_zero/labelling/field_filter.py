"""Suggest removing teacher boxes whose feet are off the field (architecture 7.1's field-mask filter).

    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.labelling.field_filter \\
        --season 2026 --week 1 --away SF --home LA

For each sampled frame, `perception.field.lines.field_mask` finds the field inside its visible
boundary lines; a player or referee box whose foot point (bottom middle) is outside it becomes a
suggested removal, written to derived/labels/<game>.field_suggestions.json as {frame: [box
indices into the teacher's detections file]}. The review page shows them already removed, so a
click restores a box the mask got wrong (`fz-review build --suggest`). Nothing is deleted.

Measured on the GRD-37 sample: on clear-game kickoffs the mask takes out both benches (16 of 52
people a frame). Of 274 removals checked by eye, 20 were players on the field, all on three frames
with a false boundary.

Checked again on the full runs (2026-10-30), and narrowed: suggestions are made on sideline-camera
frames only, and never for more than MAX_SHARE of a frame's boxes. On 12 random clear-game frames
with suggestions, 11 removed only sideline people (some benches stay); the twelfth, an end-zone
frame, removed three players in the end zone. End-zone frames often take the end-zone paint for a
boundary: 25 of 96 would have lost more than half their boxes, against 13 of 597 sideline frames
(kickoffs, where half really are benches, sit just under the cap). The snow game is left out: on 8
random frames, 4 had players on the field suggested away, one 13 of its 22.

Sideline frames that register are judged in yards instead (`feet_off`: past OFF_FIELD_YD outside a
sideline, as tracking judges tracks, with the head rule for boxes cut off by the frame), since the
mask's far boundary often sits beyond the far bench and its near one inside the field. MAX_SHARE
guards the mask only: half a kickoff frame really can be benches. On the first 40 frames of the
49ers-Rams page (2026-10-31), 34 registered, and the player boxes left a frame went from a median of
29 to 21. `--order` and `--first` redo only the frames a review will reach first, leaving the rest as
they were.

The owner's review of that page (210 frames, 2026-10-31) kept all but 35 of 2,724 pre-removals: 2
players, 33 officials. A removal stands unless clicked, so that is weaker evidence for officials
than for players. By eye about 6 of 48 random "referee" boxes left removed were officials; the rest
were chain crew, coaches and camera operators the teacher called referees, so they are mostly right
to go, and the few officials off the field go with them. The referee class therefore means officials
on the field, near enough: the round-1 student keeps those (1.4 a frame against 1.6, `labelling.train`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from formation_zero.labelling.review import frame_order

ANGLES = ("sideline",)      # where the far sideline is found reliably (GRD-37: 17 of 22)
MAX_SHARE = 0.5             # a frame that would lose more of its boxes to the mask than this gets no suggestions
OFF_FIELD_YD = 0.5          # feet this far past a sideline are off the field (as in perception.tracking)


def foot_outside(boxes: list[dict], mask: np.ndarray) -> list[int]:
    """Indices of boxes whose bottom-middle pixel is outside `mask` (feet past the frame edge count as inside)."""
    h, w = mask.shape
    out = []
    for i, b in enumerate(boxes):
        x1, _, x2, y2 = b["box"]
        x, y = int((x1 + x2) / 2), int(y2)
        if 0 <= x < w and 0 <= y < h and not mask[y, x]:
            out.append(i)
    return out


def feet_off(boxes: list[dict], shape: tuple[int, int], g: np.ndarray, centre: np.ndarray) -> list[int]:
    """Indices of boxes whose feet land past a sideline, through image -> field homography `g` with the
    camera at `centre`. A box cut off by the frame is off only past the near sideline."""
    from formation_zero.configs.nfl_field import FIELD_WIDTH_YD
    from formation_zero.perception.field import homography as hg

    pts, whole = hg.feet(np.array([b["box"] for b in boxes], float), shape, g, centre)
    y = hg.apply(g, pts)[:, 1]
    return np.flatnonzero((y < -OFF_FIELD_YD) | (whole & (y > FIELD_WIDTH_YD + OFF_FIELD_YD))).tolist()


def suggest(paths, detections: dict, samples: dict[int, dict], frames: list[str] | None = None) -> dict[str, list[int]]:
    """Suggested removals for `frames` (default all): in yards where a sideline frame registers, else by the mask."""
    from PIL import Image

    from formation_zero.perception.field import homography as hg
    from formation_zero.perception.field.lines import field_mask

    keys = list(detections["frames"]) if frames is None else frames
    out = {}
    for k, f in enumerate(keys, 1):
        boxes = detections["frames"][f]
        angle = samples[int(f)]["angle"]
        if not boxes or angle not in ANGLES:
            continue
        img = np.asarray(Image.open(paths.frames_dir / f"{int(f):06d}.jpg").convert("RGB"))
        fit = hg.register(img, angle)["fit"]
        if fit is not None and hg.plausible(fit["camera"], None):
            gone = feet_off(boxes, img.shape[:2], fit["G"], fit["camera"]["C"])
            if gone:
                out[f] = gone
        else:
            mask = field_mask(img, angle)
            if mask is not None:
                gone = foot_outside(boxes, mask)
                if gone and len(gone) <= MAX_SHARE * len(boxes):
                    out[f] = gone
        if k % 100 == 0:
            print(f"  {k}/{len(keys)}: {len(out)} frames with suggestions", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--order", help="held-out disagreement JSON: with --first, the frames a review reaches first")
    ap.add_argument("--first", type=int, help="redo only this many frames, in review order; the others keep their suggestions")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    det = json.loads((paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").read_text())
    samples = {s["frame"]: s for s in json.loads(paths.samples_path.read_text())["samples"]}
    path = paths.labels_dir / f"{paths.game_key}.field_suggestions.json"
    frames, kept = None, {}
    if a.first:
        disputed = json.loads(Path(a.order).read_text())["scores"] if a.order else None
        frames = [str(f) for f in frame_order(det, disputed)[:a.first]]
        if path.exists():
            kept = {f: v for f, v in json.loads(path.read_text())["suggestions"].items() if f not in set(frames)}
    t0 = time.perf_counter()
    out = {**kept, **suggest(paths, det, samples, frames)}
    path.write_text(json.dumps({"teacher": det["teacher"], "suggestions": out}))
    n = sum(map(len, out.values()))
    print(f"{n} boxes on {len(out)} frames suggested for removal in {time.perf_counter() - t0:.0f}s -> {path}")
    return 0


__all__ = ["ANGLES", "MAX_SHARE", "OFF_FIELD_YD", "feet_off", "foot_outside", "suggest"]


if __name__ == "__main__":
    raise SystemExit(main())
