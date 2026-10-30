"""Spike GRD-32: how fast does SAM 3 (image model, text prompts) run on this laptop?

Runs in the spike venv (setup.sh next to this file), not the project venv:

    PY=.venv-sam3d/bin/python; B=scripts/spikes/sam3_throughput/bench.py
    $PY $B frames                # 40 frames per film from labelled takes, both angles, to the drive
    $PY $B sustained --rest 480  # rest, then 1080p back to back for 6 min: rested rate, then throttling
    $PY $B burst                 # 7 configs (resolution/prompt mode/precision) interleaved frame by frame
    $PY $B overlay --picks-1080p f068627 ... --picks-576p f048644 ...   # masks and boxes, counted
    $PY $B summary               # the numbers and the cost of a game

Numbers go to results.json next to this file; frames and overlays to <film root>/spikes/sam3/.
Every timed frame runs all three prompts and copies boxes, scores and full-resolution masks back
to the CPU, so a frame's time is what a labelling pass would actually wait for.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import platform
import subprocess
import sys
import time
import types
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ops MPS lacks fall back to the CPU instead of failing; read when torch initialises.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

SAM3_SHA = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"   # facebookresearch/sam3 (setup.sh)
HF_REV = "3c879f39826c281e95690f02c7821c4de09afae7"     # facebook/sam3 checkpoint
PROMPTS = ("player", "referee", "football")
CONF = 0.5                      # upstream's default score threshold
FILMS = {                       # film window: media seconds where the All-22 film runs
    "1080p": {"game": "2025_wk20_LA-CHI", "path": "2025/wk20/LA_at_CHI/source/film_1080p.mp4",
              "window": (2677, 8452)},
    "576p": {"game": "2026_wk01_SF-LA", "path": "2026/wk01/SF_at_LA/source/film.mp4",
             "window": (60, 4500)},
}
POOL = 40                       # frames per film: 10 takes per angle, two frames each
FRACS = (0.3, 0.7)              # where in a take: near the snap, then live play
FRAMES_PER_GAME = 135_000       # ARCHITECTURE 6.2: ~180 plays x 2 angles x ~15 s x 25 fps
LABEL_FRAMES = 2_000            # ARCHITECTURE 7.1: frames sampled for the teacher labels
COLORS = {"player": (255, 210, 40), "referee": (255, 60, 200), "football": (40, 255, 120)}
BURST_CONFIGS = ("1080p/seq/fp32", "1080p/batched/fp32", "1080p/batched/bf16", "1080p/batched/fp16",
                 "576p/seq/fp32", "576p/batched/fp32", "576p/batched/bf16")
RESULTS = HERE / "results.json"


def film_root() -> Path:
    from formation_zero.data.layout import film_root_from_env

    os.chdir(ROOT)  # film_root_from_env reads .env from the working directory
    return film_root_from_env(ROOT / "data")


def out_dir() -> Path:
    return film_root() / "spikes" / "sam3"


def load_results() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {"story": "GRD-32"}


def save_results(section: str, value) -> None:
    res = load_results()
    res[section] = value
    res["machine"], res["versions"] = machine(), versions()
    res["updated"] = now_iso()
    RESULTS.write_text(json.dumps(res, indent=1))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def machine() -> dict:
    batt = sh("pmset", "-g", "batt")
    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_gb": round(int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1),
            "gpu_cores": int(next((l.split(":")[1] for l in sh("system_profiler", "SPDisplaysDataType").splitlines()
                                   if "Total Number of Cores" in l), "0")),
            "macos": platform.mac_ver()[0], "power": batt.split("'")[1] if "'" in batt else batt}


def versions() -> dict:
    import torch
    import torchvision

    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "sam3_commit": sh("git", "-C", str(ROOT / ".vendor" / "sam3"), "rev-parse", "HEAD") or SAM3_SHA,
            "checkpoint": f"facebook/sam3@{HF_REV[:7]} sam3.pt",
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}


def thermal() -> str:
    """macOS's own thermal and performance warning levels, plus load average (other jobs share
    the machine); no sudo needed."""
    therm = "; ".join(l.strip() for l in sh("pmset", "-g", "therm").splitlines() if l.strip())
    return f"{therm}; load {' '.join(f'{x:.1f}' for x in os.getloadavg())}"


# ---------------------------------------------------------------- sam3 on this machine


def import_sam3():
    """Upstream imports two things this machine lacks, neither on the image path: pkg_resources
    (gone from setuptools 81+) only locates the tokenizer vocabulary, and triton (CUDA only) only
    backs the video tracker's distance transform. Shim the first, stub the module using the second."""
    import importlib.resources

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

        def edt_triton(*args, **kwargs):
            raise NotImplementedError("needs triton (CUDA); used by video tracking only")

        edt.edt_triton = edt_triton
        sys.modules["sam3.model.edt"] = edt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # timm deprecations, CUDA autocast decorators on video code
        import sam3.model_builder as builder
        from sam3.model.sam3_image_processor import Sam3Processor
    return builder, Sam3Processor


def redirect_cuda(device: str) -> None:
    """Upstream builds two caches with device="cuda" literals (sine position encodings and the
    decoder's box coordinates) and pins one host tensor. Send the literals to `device` and make
    pinning a no-op. The image path touches nothing else CUDA-only and has no float64."""
    import torch

    target = torch.device(device)

    def fix(d):
        cuda = (isinstance(d, str) and d.startswith("cuda")) or (isinstance(d, torch.device) and d.type == "cuda")
        return target if cuda else d

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = fix(kwargs["device"])
            return fn(*args, **kwargs)
        return inner

    for name in ("zeros", "ones", "empty", "full", "arange", "tensor", "linspace"):
        setattr(torch, name, wrap(getattr(torch, name)))
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(target)
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self


def unfuse_vit_mlp() -> None:
    """The ViT's MLP calls a fused CUDA kernel (aten::_addmm_activation) that also forces
    bfloat16. MPS lacks the kernel, so it falls back to the CPU and hands bfloat16 to the next
    float32 layer, and MPS aborts the process on the mixed-dtype matmul. Run it unfused:
    act(fc1(x)), in whatever precision the caller chose."""
    import sam3.model.vitdet as vitdet
    from torch import nn
    from torch.nn import functional as F

    def addmm_act(activation, linear, x):
        return F.relu(linear(x)) if activation in (F.relu, nn.ReLU) else F.gelu(linear(x))

    vitdet.addmm_act = addmm_act


def load_model(device: str):
    builder, Sam3Processor = import_sam3()
    ckpt = film_root() / "models" / "sam3" / "sam3.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}: run scripts/spikes/sam3_throughput/setup.sh")
    if device != "cuda":
        redirect_cuda(device)
        unfuse_vit_mlp()
    t0 = time.perf_counter()
    bpe = Path(builder.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = builder.build_sam3_image_model(bpe_path=str(bpe), device="cpu", checkpoint_path=str(ckpt),
                                           load_from_HF=False)
    model = model.to(device).eval()
    proc = Sam3Processor(model, device=device, confidence_threshold=CONF)
    return model, proc, round(time.perf_counter() - t0, 1)


def precision(device: str, dtype: str):
    import torch

    if dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[dtype])


class Runner:
    """All three prompts on one frame, two ways.

    seq: upstream's documented API. set_image once, then set_text_prompt per prompt, which
         re-encodes the prompt's text every call.
    batched: set_image once, then one grounding pass with the three prompts as a batch of three
         queries on the same image. Text features are encoded once and reused for every frame.
    """

    def __init__(self, model, proc, device: str, mode: str, dtype: str):
        import torch
        from sam3.model.data_misc import FindStage

        self.torch, self.model, self.proc = torch, model, proc
        self.device, self.mode, self.dtype = device, mode, dtype
        self.mem = {"current_gb": 0.0, "driver_gb": 0.0, "rss_gb": 0.0}
        if mode == "batched":
            n = len(PROMPTS)
            with torch.inference_mode(), precision(device, dtype):
                self.text = model.backbone.forward_text(list(PROMPTS), device=device)
            self.find = FindStage(img_ids=torch.zeros(n, dtype=torch.long, device=device),
                                  text_ids=torch.arange(n, dtype=torch.long, device=device),
                                  input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                                  input_points=None, input_points_mask=None)

    def sync(self) -> None:
        if self.device == "mps":
            self.torch.mps.synchronize()

    def sample_memory(self) -> None:
        import psutil

        gb = 2**30
        if self.device == "mps":
            self.mem["current_gb"] = max(self.mem["current_gb"], self.torch.mps.current_allocated_memory() / gb)
            self.mem["driver_gb"] = max(self.mem["driver_gb"], self.torch.mps.driver_allocated_memory() / gb)
        self.mem["rss_gb"] = max(self.mem["rss_gb"], psutil.Process().memory_info().rss / gb)

    def __call__(self, img: np.ndarray) -> tuple[dict, dict]:
        torch = self.torch
        chw = torch.from_numpy(img).permute(2, 0, 1)  # the processor reads (H, W) from shape[-2:]
        with torch.inference_mode(), precision(self.device, self.dtype):
            t0 = time.perf_counter()
            state = self.proc.set_image(chw)
            self.sync()
            t1 = time.perf_counter()
            self.sample_memory()
            dets = self._seq(state) if self.mode == "seq" else self._batched(state, *img.shape[:2])
            t2 = time.perf_counter()
        return dets, {"image_s": t1 - t0, "prompts_s": t2 - t1, "total_s": t2 - t0}

    @staticmethod
    def fetch(boxes, scores, masks) -> dict:
        return {"boxes": boxes.float().cpu().numpy(), "scores": scores.float().cpu().numpy(),
                "masks": masks.squeeze(1).cpu().numpy()}

    def _seq(self, state) -> dict:
        dets = {}
        for p in PROMPTS:
            s = self.proc.set_text_prompt(prompt=p, state=state)
            self.sample_memory()
            dets[p] = self.fetch(s["boxes"], s["scores"], s["masks"])
        return dets

    def _batched(self, state, h: int, w: int) -> dict:
        """Upstream's _forward_grounding, for a batch of prompts instead of one."""
        torch = self.torch
        from sam3.model import box_ops
        from sam3.model.data_misc import interpolate

        bo = state["backbone_out"]
        bo.update(self.text)
        out = self.model.forward_grounding(backbone_out=bo, find_input=self.find, find_target=None,
                                           geometric_prompt=self.model._get_dummy_prompt(num_prompts=len(PROMPTS)))
        probs = out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)
        probs = probs.squeeze(-1)
        scale = torch.tensor([w, h, w, h], device=self.device)
        dets = {}
        for b, p in enumerate(PROMPTS):
            keep = probs[b] > CONF
            boxes = box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][b][keep]) * scale
            masks = interpolate(out["pred_masks"][b][keep].unsqueeze(1), (h, w), mode="bilinear",
                                align_corners=False).sigmoid() > 0.5
            self.sample_memory()
            dets[p] = self.fetch(boxes, probs[b][keep], masks)
        return dets


# ---------------------------------------------------------------- frames


def takes_for(game: str, window: tuple[float, float]) -> list[dict]:
    """Labelled takes (data/groundtruth, read only), or even slices of the window without them."""
    gt = ROOT / "data" / "groundtruth" / f"{game}.shot_views.json"
    try:
        labels = [l for l in json.loads(gt.read_text())["labels"]
                  if l.get("confident") and l["end_s"] - l["start_s"] >= 4
                  and window[0] <= l["start_s"] and l["end_s"] <= window[1]]
    except (OSError, ValueError, KeyError):
        labels = []
    if not labels:
        edges = np.linspace(*window, POOL // 2 + 1)
        return [{"start_s": a, "end_s": b, "view": "unlabelled"} for a, b in zip(edges[:-1], edges[1:])]
    picked = []
    for view in ("SIDELINE", "ENDZONE"):
        takes = [l for l in labels if l["view"] == view]
        idx = sorted(set(np.linspace(0, len(takes) - 1, min(POOL // 4, len(takes))).round().astype(int)))
        picked += [takes[i] for i in idx]
    return picked


def cmd_frames(a) -> int:
    from PIL import Image

    from formation_zero.ingest.sources import FileSource

    info = {}
    for res, film in FILMS.items():
        src_path = film_root() / film["path"]
        dest = out_dir() / "frames" / res
        dest.mkdir(parents=True, exist_ok=True)
        wanted = sorted((t["start_s"] + f * (t["end_s"] - t["start_s"]), t["view"])
                        for t in takes_for(film["game"], film["window"]) for f in FRACS)
        frames = []
        with FileSource(src_path) as src:
            for time_s, view in wanted:
                fi = round(time_s * src.fps)
                fr = next(src.iter_frames(fi, fi + 1), None)
                if fr is None:
                    continue
                name = f"f{fr.index:06d}.png"
                Image.fromarray(fr.image).save(dest / name)
                frames.append({"file": name, "frame": fr.index, "time_s": round(fr.time_s, 2), "view": view})
            # Sequential decode speed, for comparison with the model: 10 s of film from mid-window.
            start = round(sum(film["window"]) / 2 * src.fps)
            t0 = time.perf_counter()
            n = sum(1 for _ in src.iter_frames(start, start + 250))
            decode_fps = n / (time.perf_counter() - t0)
            size = (src.width, src.height)
        (dest / "frames.json").write_text(json.dumps({"source": str(src_path), "frames": frames}, indent=1))
        info[res] = {"game": film["game"], "source": film["path"], "size": size, "frames": len(frames),
                     "views": {v: sum(f["view"] == v for f in frames) for v in {f["view"] for f in frames}},
                     "decode_fps": round(decode_fps, 1)}
        print(f"{res}: {len(frames)} frames {size[0]}x{size[1]} -> {dest}; decode {decode_fps:.0f} fps")
    save_results("frames", info)
    return 0


def load_pool(res: str) -> list[tuple[dict, np.ndarray]]:
    from PIL import Image

    d = out_dir() / "frames" / res
    if not (d / "frames.json").exists():
        raise SystemExit(f"no frames in {d}: run bench.py frames")
    meta = json.loads((d / "frames.json").read_text())["frames"]
    return [(m, np.array(Image.open(d / m["file"]).convert("RGB"))) for m in meta]


# ---------------------------------------------------------------- timing


def warm_up(run: Runner, pool, n: int) -> list[str]:
    """Untimed frames: kernel compilation, allocator growth. Returns the ops that fell back to the
    CPU. PyTorch reports those once per process, from C++ to stderr, so stderr is captured here."""
    import re
    import tempfile

    with tempfile.TemporaryFile() as tmp:
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            for _, img in pool[:n]:
                run(img)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            text = tmp.read().decode(errors="replace")
    sys.stderr.write(text)
    return sorted(set(re.findall(r"operator '([^']+)' is not currently supported on the MPS", text)))


def summarise(times: list[dict], dets: list[dict]) -> dict:
    total = np.array([t["total_s"] for t in times])
    return {"frames": len(times), "fps": round(len(total) / total.sum(), 3),
            "s_per_frame_mean": round(float(total.mean()), 3), "s_per_frame_median": round(float(np.median(total)), 3),
            "s_per_frame_p90": round(float(np.percentile(total, 90)), 3),
            "image_s_mean": round(float(np.mean([t["image_s"] for t in times])), 3),
            "prompts_s_mean": round(float(np.mean([t["prompts_s"] for t in times])), 3),
            "detections_mean": {p: round(float(np.mean([d[p] for d in dets])), 1) for p in PROMPTS}}


def cmd_burst(a) -> int:
    """Every config on the same frames, round-robin. The machine is fanless and slows as it heats
    up, so running configs one after another would favour whichever ran first; interleaved, the
    heat slows all of them alike. Numbers from a rested machine come from `sustained`."""
    model, proc, load_s = load_model(a.device)
    configs = [tuple(c.split("/")) for c in a.configs]
    pools = {r: load_pool(r) for r in {c[0] for c in configs}}
    runs = {c: Runner(model, proc, a.device, c[1], c[2]) for c in configs}
    fallbacks = sorted({op for c, run in runs.items() for op in warm_up(run, pools[c[0]], a.warmup)})
    times = {c: [] for c in configs}
    dets = {c: [] for c in configs}
    started = now_iso()
    for i in range(a.warmup, a.warmup + a.frames):
        for c, run in runs.items():
            d, t = run(pools[c[0]][i][1])
            times[c].append(t)
            dets[c].append({p: len(d[p]["scores"]) for p in PROMPTS})
        print(f"  round {i - a.warmup + 1}/{a.frames}: " + "  ".join(f"{times[c][-1]['total_s']:.1f}" for c in configs))
    out = {}
    for c in configs:
        s = summarise(times[c], dets[c])
        img = pools[c[0]][0][1]
        s.update({"input": f"{img.shape[1]}x{img.shape[0]}", "model_input": "1008x1008",
                  "memory_current_peak_gb": round(runs[c].mem["current_gb"], 2),
                  "per_frame": [round(t["total_s"], 3) for t in times[c]]})
        out["/".join(c)] = s
        print(f"{'/'.join(c):22s} {s['fps']:.3f} fps  {s['s_per_frame_mean']:.2f} s/frame "
              f"(image {s['image_s_mean']:.2f} + prompts {s['prompts_s_mean']:.2f})  dets {s['detections_mean']}")
    peak = {k: round(max(r.mem[k] for r in runs.values()), 2) for k in ("current_gb", "driver_gb", "rss_gb")}
    save_results("burst", {"order": "interleaved, one frame per config per round", "device": a.device,
                           "warmup_frames": a.warmup, "started": started, "finished": now_iso(),
                           "load_s": load_s, "memory_peak": peak, "mps_cpu_fallback_ops": fallbacks,
                           "thermal_after": thermal(), "configs": out})
    return 0


def cmd_sustained(a) -> int:
    """Warm up (kernel compilation), rest so the machine cools, then run frames back to back.
    The first `--burst` frames are the rested burst number; the rest shows the throttling."""
    model, proc, load_s = load_model(a.device)
    pool = load_pool(a.res)
    run = Runner(model, proc, a.device, a.mode, a.dtype)
    fallbacks = warm_up(run, pool, a.warmup)
    if a.rest:
        print(f"  resting {a.rest}s")
        time.sleep(a.rest)
    before, started = thermal(), now_iso()
    log, t_start, k = [], time.perf_counter(), 0
    while time.perf_counter() - t_start < a.seconds:
        _, img = pool[k % len(pool)]
        d, t = run(img)
        log.append({"t": round(time.perf_counter() - t_start, 2), "s": round(t["total_s"], 3),
                    "image_s": round(t["image_s"], 3), "players": len(d["player"]["scores"])})
        k += 1
        if k % 10 == 0:
            print(f"  {log[-1]['t']:6.0f}s  {k} frames  last {t['total_s']:.2f} s/frame")
    elapsed = log[-1]["t"]

    def fps(sel):
        return round(len(sel) / sum(sel), 3) if sel else None

    windows = [{"from_s": lo, "fps": fps([e["s"] for e in log if lo <= e["t"] - e["s"] and e["t"] <= lo + a.window])}
               for lo in range(0, int(elapsed), a.window)]
    out = {"config": f"{a.res}/{a.mode}/{a.dtype}", "device": a.device, "rest_s": a.rest,
           "seconds": round(elapsed, 1), "frames": len(log), "fps": fps([e["s"] for e in log]),
           f"fps_first_{a.burst}_frames": fps([e["s"] for e in log[:a.burst]]),
           "fps_first_minute": fps([e["s"] for e in log if e["t"] <= 60]),
           "fps_last_minute": fps([e["s"] for e in log if e["t"] >= elapsed - 60]),
           "image_s_first_minute": round(float(np.mean([e["image_s"] for e in log if e["t"] <= 60])), 3),
           "image_s_last_minute": round(float(np.mean([e["image_s"] for e in log if e["t"] >= elapsed - 60])), 3),
           "windows": windows, "memory_peak": {k: round(v, 2) for k, v in run.mem.items()},
           "mps_cpu_fallback_ops": fallbacks, "load_s": load_s, "thermal_before": before,
           "thermal_after": thermal(), "started": started, "per_frame": log}
    save_results("sustained", out)
    print(f"{out['config']}: {out['frames']} frames in {elapsed:.0f}s, {out['fps']:.3f} fps overall; "
          f"first {a.burst} frames {out[f'fps_first_{a.burst}_frames']:.3f}, first minute "
          f"{out['fps_first_minute']:.3f}, last minute {out['fps_last_minute']:.3f}; mem {out['memory_peak']}")
    return 0


# ---------------------------------------------------------------- overlays


def draw(img: np.ndarray, dets: dict, title: str) -> np.ndarray:
    import cv2

    out = img.astype(np.float32)
    for p in PROMPTS:
        c = np.array(COLORS[p], np.float32)
        for m in dets[p]["masks"]:
            out[m] = 0.5 * out[m] + 0.5 * c
    out = out.astype(np.uint8)
    lw = 2 if img.shape[0] > 700 else 1
    for p in PROMPTS:
        for (x1, y1, x2, y2), s in zip(dets[p]["boxes"], dets[p]["scores"]):
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[p], lw)
            if p != "player":
                cv2.putText(out, f"{s:.2f}", (int(x1), max(10, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 * lw, COLORS[p], lw, cv2.LINE_AA)
    bar = np.zeros((26 * lw, out.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 18 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * lw, (255, 255, 255), lw, cv2.LINE_AA)
    return np.vstack([bar, out])


def cmd_overlay(a) -> int:
    from PIL import Image

    model, proc, _ = load_model(a.device)
    run = Runner(model, proc, a.device, a.mode, a.dtype)
    dest = out_dir() / "overlays"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for res, names in (("1080p", a.picks_1080p), ("576p", a.picks_576p)):
        pool = {m["file"]: (m, img) for m, img in load_pool(res)}
        for name in names:
            m, img = pool[name if name.endswith(".png") else f"{name}.png"]
            dets, t = run(img)
            counts = {p: len(dets[p]["scores"]) for p in PROMPTS}
            hs = np.sort(dets["player"]["boxes"][:, 3] - dets["player"]["boxes"][:, 1]) if counts["player"] else []
            title = (f"{FILMS[res]['game']} {res} {m['view']} t={m['time_s']:.0f}s  " +
                     "  ".join(f"{p} {n}" for p, n in counts.items()) + f"  ({t['total_s']:.2f}s)")
            out = dest / f"{res}_{m['file'].removesuffix('.png')}.png"
            Image.fromarray(draw(img, dets, title)).save(out)
            rows.append({"image": str(out), "res": res, "frame": m["frame"], "time_s": m["time_s"],
                         "view": m["view"], "counts": counts,
                         "player_box_height_px": [round(float(h), 1) for h in hs],
                         "referee_scores": [round(float(s), 2) for s in dets["referee"]["scores"]],
                         "football_scores": [round(float(s), 2) for s in dets["football"]["scores"]]})
            print(f"{out.name}: {counts}; player heights {rows[-1]['player_box_height_px'][:3]}..."
                  f"{rows[-1]['player_box_height_px'][-3:]}")
    old = load_results().get("overlays", {})
    save_results("overlays", {"config": f"{a.mode}/{a.dtype}/{a.device}", "threshold": CONF,
                              "images": rows, "notes": old.get("notes", "")})
    return 0


# ---------------------------------------------------------------- summary


def cmd_summary(a) -> int:
    res = load_results()
    sus = res.get("sustained")
    if sus:
        fps = sus["fps"]
        first = next(v for k, v in sus.items() if k.startswith("fps_first_") and k.endswith("_frames"))
        print(f"sustained {sus['config']} after {sus['rest_s']} s rest: {sus['frames']} frames in "
              f"{sus['seconds']:.0f} s = {fps:.3f} fps ({1 / fps:.2f} s/frame); first frames {first:.3f} fps, "
              f"first minute {sus['fps_first_minute']:.3f}, last minute {sus['fps_last_minute']:.3f}")
        # A game runs for days, so the settled (throttled) rate is the one that applies to it.
        for label, rate in (("rested", first), (f"{sus['seconds']:.0f} s average", fps),
                            ("settled (last minute)", sus["fps_last_minute"])):
            print(f"  at the {label} rate, {rate:.3f} fps: one game ({FRAMES_PER_GAME:,} frames) "
                  f"{FRAMES_PER_GAME / rate / 3600:.0f} h; labelling set ({LABEL_FRAMES:,} frames) "
                  f"{LABEL_FRAMES / rate / 3600:.1f} h")
    burst = res.get("burst", {})
    if burst.get("configs"):
        print(f"\ninterleaved comparison ({burst['order']}):")
        print(f"{'config':22s} {'fps':>6} {'s/frame':>8} {'image':>6} {'prompts':>8}  detections")
        for key, s in burst["configs"].items():
            print(f"{key:22s} {s['fps']:6.3f} {s['s_per_frame_mean']:8.2f} {s['image_s_mean']:6.2f} "
                  f"{s['prompts_s_mean']:8.2f}  {s['detections_mean']}")
        print(f"peak memory {burst['memory_peak']}")
    return 0


# ---------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"frames": cmd_frames, "burst": cmd_burst, "sustained": cmd_sustained,
            "overlay": cmd_overlay, "summary": cmd_summary}
    sps = {name: sub.add_parser(name) for name in cmds}
    for name in ("burst", "sustained", "overlay"):
        sps[name].add_argument("--device", default="mps", choices=["mps", "cpu"])
        sps[name].add_argument("--warmup", type=int, default=3)
    sps["burst"].add_argument("--configs", nargs="+", default=list(BURST_CONFIGS),
                              help="resolution/mode/precision, e.g. 1080p/batched/bf16")
    sps["burst"].add_argument("--frames", type=int, default=15, help="timed frames per config")
    for name in ("sustained", "overlay"):
        sps[name].add_argument("--mode", default="batched", choices=["seq", "batched"])
        sps[name].add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    sps["sustained"].add_argument("--res", default="1080p", choices=list(FILMS))
    sps["sustained"].add_argument("--seconds", type=int, default=360)
    sps["sustained"].add_argument("--rest", type=int, default=0, help="idle seconds after warm-up")
    sps["sustained"].add_argument("--burst", type=int, default=20, help="frames in the rested burst figure")
    sps["sustained"].add_argument("--window", type=int, default=30, help="seconds per fps window")
    sps["overlay"].add_argument("--picks-1080p", nargs="+", default=[])
    sps["overlay"].add_argument("--picks-576p", nargs="+", default=[])
    a = ap.parse_args(argv)
    return cmds[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
