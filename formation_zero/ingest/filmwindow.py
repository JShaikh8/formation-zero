"""Find where the coaches film is inside a recording, without anyone typing times.

A screen recording of NFL+ starts with menus and other content and may end on a frozen frame.
Coaches film is continuously *moving* pictures at a stable brightness: every second differs
from the last by a moderate amount (a moving camera, moving players, a cut every dozen seconds),
never by nothing (a frozen frame, an idle menu) and rarely by everything (a scrolling menu).

Measured on the demo game (a snow game, so the field is white, which is why this does not use
colour): per-second luma difference is never below 8 during the film, about 1 on the frozen tail
(compression noise), and 0 for 28% of the menu seconds; film brightness stays in a tight band
(0.48 to 0.60) while everything before it is dim (0.08 to 0.37). Thirty-second windows are
flagged as film when at least 80% of their seconds move, the median difference is moderate, the
brightness is stable inside the window and sits in the recording's bright mode. The film window
is the longest run of flagged windows, tolerating short gaps. On the demo game that lands within
one window of the hand-found answer at both ends.

`fieldness` (fraction of turf-coloured pixels) is kept for later stages; it is not used here
because snow, night games and dome turf all move it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_zero.ingest.sources import FrameSource


@dataclass(frozen=True)
class FilmWindow:
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    samples: int
    threshold: float          # brightness threshold used (relative to the recording)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def fieldness(image: np.ndarray) -> float:
    """Fraction of pixels that read as turf: green dominant over red and blue, not too dark."""
    rgb = image.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    turf = (g > r + 12) & (g > b + 12) & (g > 40)
    return float(turf.mean())


def luma_small(image: np.ndarray, size: tuple[int, int] = (64, 36)) -> np.ndarray:
    """Cheap luma thumbnail by striding (no resize dependency)."""
    h, w = image.shape[:2]
    ys = (np.arange(size[1]) * h / size[1]).astype(int).clip(0, h - 1)
    xs = (np.arange(size[0]) * w / size[0]).astype(int).clip(0, w - 1)
    return image[ys][:, xs].astype(np.float32).mean(axis=2)


def motion_samples(source: FrameSource, step_s: float = 1.0, max_frames: int | None = None) -> list[tuple[int, float, float, float]]:
    """(frame, time, luma difference from the previous sample, mean brightness 0..1) every `step_s`."""
    step = max(1, int(round(step_s * source.fps)))
    out = []
    prev = None
    for frame in source.iter_frames(0, max_frames, step):
        g = luma_small(frame.image)
        d = float(np.abs(g - prev).mean()) if prev is not None else 0.0
        out.append((frame.index, frame.time_s, d, float(g.mean() / 255.0)))
        prev = g
    return out


def longest_run(flags: list[bool], max_gap: int) -> tuple[int, int] | None:
    """Longest stretch of True allowing up to `max_gap` consecutive False inside it. Returns [i, j]."""
    best = None
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        last_true = i
        gap = 0
        while j < n:
            if flags[j]:
                last_true = j
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        if best is None or (last_true - i) > (best[1] - best[0]):
            best = (i, last_true)
        i = last_true + 1
    return best


def find(source: FrameSource, *, step_s: float = 1.0, window_s: float = 30.0, max_gap_s: float = 60.0,
         min_duration_s: float = 60.0, moving_min: float = 3.0, moving_max: float = 80.0,
         moving_fraction: float = 0.8, brightness_std_max: float = 0.10) -> FilmWindow | None:
    """The film window of a recording, or None if nothing looks like film for long enough."""
    samples = motion_samples(source, step_s)
    per_window = max(1, int(round(window_s / step_s)))
    if len(samples) < per_window:
        return None
    windows = []
    for k in range(0, len(samples) - per_window + 1, per_window):
        chunk = samples[k:k + per_window]
        diffs = np.array([c[2] for c in chunk[1:]] if k == 0 else [c[2] for c in chunk])
        bright = np.array([c[3] for c in chunk])
        moving = float(((diffs >= moving_min) & (diffs < moving_max)).mean()) if diffs.size else 0.0
        windows.append({"start": chunk[0], "end": chunk[-1], "moving": moving,
                        "median": float(np.median(diffs)) if diffs.size else 0.0,
                        "bstd": float(bright.std()), "bmean": float(bright.mean())})
    bmeans = np.array([w["bmean"] for w in windows])
    lo, hi = np.percentile(bmeans, 20), np.percentile(bmeans, 80)
    # The film is the recording's bright, stable mode. Relative, with a margin, so a recording
    # that is all film (lo ~ hi) still passes everywhere.
    bright_threshold = float((lo + hi) / 2 - 0.05)
    flags = [w["moving"] >= moving_fraction and moving_min <= w["median"] <= moving_max
             and w["bstd"] < brightness_std_max and w["bmean"] >= bright_threshold for w in windows]
    run = longest_run(flags, max_gap=int(max_gap_s / window_s))
    if run is None:
        return None
    i, j = run
    start_f, start_t = windows[i]["start"][0], windows[i]["start"][1]
    end_f = windows[j]["end"][0] + int(round(step_s * source.fps))
    end_t = windows[j]["end"][1] + step_s
    if end_t - start_t < min_duration_s:
        return None
    return FilmWindow(start_s=round(start_t, 3), end_s=round(end_t, 3), start_frame=start_f, end_frame=end_f,
                      samples=len(samples), threshold=round(bright_threshold, 3))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.ingest.sources import FileSource

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", help="the source or the proxy; both are frame-aligned")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--window", type=float, default=30.0)
    a = ap.parse_args(argv)
    with FileSource(a.video) as src:
        w = find(src, step_s=a.step, window_s=a.window)
    if w is None:
        print("no film window found")
        return 1
    print(f"film {w.start_s / 60:.1f}-{w.end_s / 60:.1f} min (frames {w.start_frame}-{w.end_frame}), "
          f"{w.duration_s / 60:.1f} min, brightness threshold {w.threshold}, {w.samples} samples")
    return 0


__all__ = ["FilmWindow", "fieldness", "luma_small", "motion_samples", "longest_run", "find"]

if __name__ == "__main__":
    raise SystemExit(main())
