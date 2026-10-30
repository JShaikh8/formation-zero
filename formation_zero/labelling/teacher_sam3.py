"""SAM 3 as the labelling teacher: text prompts to boxes on the sampled frames (GRD-29).

Runs in the SAM 3 environment (scripts/spikes/sam3_throughput/setup.sh), not the project venv:

    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.labelling.teacher_sam3 \\
        --season 2025 --week 20 --away LA --home CHI

Reads the frames `labelling.sample` picked and `fz-sample --extract` saved on the film drive. Writes
one JSON line per frame to derived/labels/<game>.teacher_sam3.jsonl as it goes, so a run that is
stopped resumes where it left off, and at the end the review tool's detections file,
derived/labels/<game>.teacher_sam3.json (see `labelling.review`).

Speed (GRD-32): about 6.3 s a frame once the fanless laptop is warm, so 2,000 frames take about
3.5 hours. The three prompts run as one batch, in bfloat16, on the GPU.

Filters (architecture 7.1), applied before review:
- duplicates: within a class, a box overlapping a better-scoring one by IoU > 0.7 goes;
- football: a "football" taller than a third of the frame's median player goes (SAM 3 called the
  ball painted in the NFL shield a football);
- field mask: not yet. Uniformed people on the sidelines come back as players (9 of 29 in one
  kickoff frame) until field registration exists (milestone 4); the reviewer sees them.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

PROMPTS = ("player", "referee", "football")
# What the run records, and what the detections file keeps. Balls go into the log down to 0.3 but
# the file keeps 0.4 and up: on 48 snow-game balls checked by eye, 21 of 24 at 0.4 or more were the
# ball (none clearly not), against 10 of 24 below it (8 hands, shoes or blur). The log keeps the
# rest, so the cut can move without running SAM 3 again (--limit 0 rewrites the file).
LOG_FLOOR = {"player": 0.5, "referee": 0.5, "football": 0.3}
THRESHOLDS = {"player": 0.5, "referee": 0.5, "football": 0.4}
DUPLICATE_IOU = 0.7
BALL_MAX_OF_PLAYER = 1 / 3
SAM3_COMMIT = "f1647e8"


# ---------------------------------------------------------------- filters (numpy only, tested)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box against many, boxes as [x1, y1, x2, y2]."""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def drop_duplicates(boxes: list[dict], threshold: float = DUPLICATE_IOU) -> tuple[list[dict], int]:
    """Within each label, keep a box only if no better-scoring kept box overlaps it by more than `threshold`."""
    kept, dropped = [], 0
    for label in {b["label"] for b in boxes}:
        mine = sorted((b for b in boxes if b["label"] == label), key=lambda b: -b["score"])
        chosen: list[dict] = []
        for b in mine:
            if chosen and iou(np.array(b["box"]), np.array([c["box"] for c in chosen])).max() > threshold:
                dropped += 1
            else:
                chosen.append(b)
        kept += chosen
    return kept, dropped


def drop_giant_balls(boxes: list[dict], ratio: float = BALL_MAX_OF_PLAYER) -> tuple[list[dict], int]:
    """A football is a fraction of a player's height; anything bigger is a logo or a mistake."""
    heights = [b["box"][3] - b["box"][1] for b in boxes if b["label"] == "player"]
    if not heights:
        return boxes, 0
    limit = ratio * float(np.median(heights))
    kept = [b for b in boxes if b["label"] != "football" or b["box"][3] - b["box"][1] <= limit]
    return kept, len(boxes) - len(kept)


def apply_filters(boxes: list[dict]) -> tuple[list[dict], dict]:
    boxes, dup = drop_duplicates(boxes)
    boxes, ball = drop_giant_balls(boxes)
    return sorted(boxes, key=lambda b: (b["label"], -b["score"])), {"duplicates": dup, "giant_balls": ball}


# ---------------------------------------------------------------- SAM 3 on this machine (see GRD-32)


def _shims(device: str) -> None:
    """The four things upstream needs that Apple silicon lacks, none of them real work (GRD-32 README)."""
    import importlib.resources

    import torch

    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda pkg, name: str(importlib.resources.files(pkg) / name)
        sys.modules["pkg_resources"] = shim
    try:
        import triton  # noqa: F401
    except ImportError:
        edt = types.ModuleType("sam3.model.edt")
        edt.edt_triton = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("video tracking needs triton"))
        sys.modules["sam3.model.edt"] = edt
    if device == "cuda":
        return
    target = torch.device(device)

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            d = kwargs.get("device")
            if (isinstance(d, str) and d.startswith("cuda")) or (isinstance(d, torch.device) and d.type == "cuda"):
                kwargs["device"] = target
            return fn(*args, **kwargs)
        return inner

    for name in ("zeros", "ones", "empty", "full", "arange", "tensor", "linspace"):
        setattr(torch, name, wrap(getattr(torch, name)))
    torch.Tensor.cuda = lambda self, *a, **k: self.to(target)
    torch.Tensor.pin_memory = lambda self, *a, **k: self


class Teacher:
    """SAM 3 image model with the three prompts batched, in bfloat16."""

    def __init__(self, checkpoint: Path, device: str = "mps"):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch

        _shims(device)
        import sam3.model_builder as builder
        from sam3.model import vitdet
        from sam3.model.data_misc import FindStage
        from sam3.model.sam3_image_processor import Sam3Processor
        from torch.nn import functional as F

        # The fused CUDA MLP kernel falls back to the CPU in bfloat16 and MPS aborts; run it unfused.
        vitdet.addmm_act = lambda act, linear, x: F.relu(linear(x)) if act in (F.relu, torch.nn.ReLU) else F.gelu(linear(x))
        bpe = Path(builder.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        model = builder.build_sam3_image_model(bpe_path=str(bpe), device="cpu", checkpoint_path=str(checkpoint),
                                               load_from_HF=False)
        self.torch, self.device = torch, device
        self.model = model.to(device).eval()
        self.proc = Sam3Processor(self.model, device=device, confidence_threshold=min(LOG_FLOOR.values()))
        n = len(PROMPTS)
        with torch.inference_mode(), self._autocast():
            self.text = self.model.backbone.forward_text(list(PROMPTS), device=device)
        self.find = FindStage(img_ids=torch.zeros(n, dtype=torch.long, device=device),
                              text_ids=torch.arange(n, dtype=torch.long, device=device),
                              input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                              input_points=None, input_points_mask=None)

    def _autocast(self):
        return self.torch.autocast(device_type=self.device, dtype=self.torch.bfloat16)

    def __call__(self, img: np.ndarray) -> list[dict]:
        """Boxes for every prompt on one RGB frame, in its own pixels, above each prompt's log floor."""
        from sam3.model import box_ops

        torch = self.torch
        h, w = img.shape[:2]
        with torch.inference_mode(), self._autocast():
            state = self.proc.set_image(torch.from_numpy(img).permute(2, 0, 1))
            bo = state["backbone_out"]
            bo.update(self.text)
            out = self.model.forward_grounding(backbone_out=bo, find_input=self.find, find_target=None,
                                               geometric_prompt=self.model._get_dummy_prompt(num_prompts=len(PROMPTS)))
            probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
            scale = torch.tensor([w, h, w, h], device=self.device)
            boxes = []
            for k, label in enumerate(PROMPTS):
                keep = probs[k] > LOG_FLOOR[label]
                xyxy = (box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][k][keep]) * scale).float().cpu().numpy()
                scores = probs[k][keep].float().cpu().numpy()
                boxes += [{"label": label, "box": [round(float(v), 1) for v in b], "score": round(float(s), 3)}
                          for b, s in zip(xyxy, scores)]
        return boxes


# ---------------------------------------------------------------- the run


def run(paths, *, device: str = "mps", limit: int | None = None) -> Path:
    from PIL import Image

    samples = json.loads(paths.samples_path.read_text())["samples"]
    log = paths.labels_dir / f"{paths.game_key}.teacher_sam3.jsonl"
    done = {json.loads(line)["frame"] for line in log.read_text().splitlines() if line.strip()} if log.exists() else set()
    todo = [s for s in samples if s["frame"] not in done][:limit]
    missing = [s["frame"] for s in todo if not (paths.frames_dir / f"{s['frame']:06d}.jpg").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} frames not extracted (fz-sample --extract), e.g. {missing[:3]}")
    print(f"{len(done)} frames already done, {len(todo)} to go")
    if todo:
        t0 = time.perf_counter()
        teacher = Teacher(paths.film_root / "models" / "sam3" / "sam3.pt", device)
        print(f"model loaded in {time.perf_counter() - t0:.0f}s")
        with log.open("a") as f:
            for i, s in enumerate(todo, 1):
                t = time.perf_counter()
                img = np.array(Image.open(paths.frames_dir / f"{s['frame']:06d}.jpg").convert("RGB"))
                boxes, dropped = apply_filters(teacher(img))
                f.write(json.dumps({"frame": s["frame"], "size": [img.shape[1], img.shape[0]], "boxes": boxes,
                                    "dropped": dropped, "s": round(time.perf_counter() - t, 2)}) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  {time.perf_counter() - t:.1f}s/frame  {len(boxes)} boxes", flush=True)
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    out = paths.labels_dir / f"{paths.game_key}.teacher_sam3.json"
    out.write_text(json.dumps(detections_file(rows)))
    return out


def detections_file(rows: list[dict]) -> dict:
    """The review tool's detections from the run's log, cut at THRESHOLDS. The thresholds are part of
    the teacher's name: the review page keys its saved decisions by it, and box indices change with them.
    So is the frame size when the log records it (runs from 2026-10-30 on), so the 576p and 1080p labels
    of one game never share saved decisions (Chrome gives every local page the same storage)."""
    cut = " ".join(f"{k}>{v}" for k, v in THRESHOLDS.items())
    sizes = {tuple(r["size"]) for r in rows if "size" in r}
    size = f" {'/'.join(f'{w}x{h}' for w, h in sorted(sizes))}" if sizes else ""
    return {"teacher": f"sam3@{SAM3_COMMIT} bf16 batched{size}, {cut}", "thresholds": THRESHOLDS,
            "frames": {str(r["frame"]): [b for b in r["boxes"] if b["score"] > THRESHOLDS[b["label"]]] for r in rows}}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, help="stop after this many new frames (a trial run; 0 rewrites the detections file from the log)")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    if not paths.film_root_available():
        raise SystemExit(f"film drive not mounted: {paths.film_root}")
    print(f"detections -> {run(paths, device=a.device, limit=a.limit)}")
    return 0


__all__ = ["LOG_FLOOR", "PROMPTS", "THRESHOLDS", "Teacher", "apply_filters", "detections_file", "drop_duplicates",
           "drop_giant_balls", "iou", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
