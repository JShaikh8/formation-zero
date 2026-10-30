"""Suggest removing teacher boxes whose feet are off the field (architecture 7.1's field-mask filter).

    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.labelling.field_filter \\
        --season 2026 --week 1 --away SF --home LA

For each sampled frame, `perception.field.lines.field_mask` finds the field inside its visible
boundary lines; a player or referee box whose foot point (bottom middle) is outside it becomes a
suggested removal, written to derived/labels/<game>.field_suggestions.json as {frame: [box
indices into the teacher's detections file]}. The review page shows them already removed, so a
click restores a box the mask got wrong (`fz-review build --suggest`). Nothing is deleted.

Measured on the GRD-37 sample: on clear-game kickoffs the mask takes out both benches (16 of 52
people a frame); on the snow game it rarely has an answer, so it suggests little. Of 274 removals
checked by eye, 20 were players on the field, all on three frames with a false boundary.
"""

from __future__ import annotations

import json
import time

import numpy as np


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


def suggest(paths, detections: dict, samples: dict[int, dict]) -> dict[str, list[int]]:
    from PIL import Image

    from formation_zero.perception.field.lines import field_mask

    out = {}
    for k, (f, boxes) in enumerate(detections["frames"].items(), 1):
        if not boxes:
            continue
        img = np.asarray(Image.open(paths.frames_dir / f"{int(f):06d}.jpg").convert("RGB"))
        mask = field_mask(img, samples[int(f)]["angle"])
        if mask is not None:
            gone = foot_outside(boxes, mask)
            if gone:
                out[f] = gone
        if k % 250 == 0:
            print(f"  {k}/{len(detections['frames'])}: {len(out)} frames with suggestions", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    det = json.loads((paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").read_text())
    samples = {s["frame"]: s for s in json.loads(paths.samples_path.read_text())["samples"]}
    t0 = time.perf_counter()
    out = suggest(paths, det, samples)
    path = paths.labels_dir / f"{paths.game_key}.field_suggestions.json"
    path.write_text(json.dumps({"teacher": det["teacher"], "suggestions": out}))
    n = sum(map(len, out.values()))
    print(f"{n} boxes on {len(out)} frames suggested for removal in {time.perf_counter() - t0:.0f}s -> {path}")
    return 0


__all__ = ["foot_outside", "suggest"]


if __name__ == "__main__":
    raise SystemExit(main())
