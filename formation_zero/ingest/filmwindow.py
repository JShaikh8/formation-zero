"""Find where the coaches film is inside a recording, without anyone typing times.

A screen recording of NFL+ starts with menus and may end on a frozen frame. The film itself is
the long stretch where frames look like a football field: mostly green, with strong straight
edges. Sampled every couple of seconds on the proxy, that "fieldness" is a clean step function,
and the film window is its longest high run, tolerating short dips (replays, close-ups, a cut to
black between angles).

The score is deliberately simple and dependency-free. Snow, night games and dome turf shift the
green; the threshold is relative to the recording's own score distribution, not absolute.
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
    threshold: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def fieldness(image: np.ndarray) -> float:
    """Fraction of pixels that read as turf: green dominant over red and blue, not too dark."""
    rgb = image.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    turf = (g > r + 12) & (g > b + 12) & (g > 40)
    return float(turf.mean())


def scores(source: FrameSource, step_s: float = 2.0, max_frames: int | None = None) -> list[tuple[int, float, float]]:
    """(frame, time, fieldness) every `step_s` seconds."""
    step = max(1, int(round(step_s * source.fps)))
    out = []
    for frame in source.iter_frames(0, max_frames, step):
        out.append((frame.index, frame.time_s, fieldness(frame.image)))
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


def find(source: FrameSource, *, step_s: float = 2.0, max_gap_s: float = 30.0,
         min_duration_s: float = 60.0) -> FilmWindow | None:
    """The film window of a recording, or None if nothing looks like a field for long enough."""
    samples = scores(source, step_s)
    if not samples:
        return None
    vals = np.array([s[2] for s in samples])
    # Bimodal by construction: menus near 0, field well above. Split at the midpoint between the
    # low and high modes, with a floor so a recording that is all menu never "finds" a window.
    lo, hi = np.percentile(vals, 10), np.percentile(vals, 90)
    threshold = max(0.12, float((lo + hi) / 2))
    flags = [v >= threshold for v in vals]
    run = longest_run(flags, max_gap=int(max_gap_s / step_s))
    if run is None:
        return None
    i, j = run
    start_f, start_t = samples[i][0], samples[i][1]
    end_f = samples[j][0] + int(round(step_s * source.fps))
    end_t = samples[j][1] + step_s
    if end_t - start_t < min_duration_s:
        return None
    return FilmWindow(start_s=round(start_t, 3), end_s=round(end_t, 3), start_frame=start_f,
                      end_frame=end_f, samples=len(samples), threshold=round(threshold, 3))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.ingest.sources import FileSource

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", help="use the proxy rendition; it is faster and frame-aligned")
    ap.add_argument("--step", type=float, default=2.0)
    a = ap.parse_args(argv)
    with FileSource(a.video) as src:
        w = find(src, step_s=a.step)
    if w is None:
        print("no film window found")
        return 1
    print(f"film {w.start_s / 60:.1f}-{w.end_s / 60:.1f} min (frames {w.start_frame}-{w.end_frame}), "
          f"{w.duration_s / 60:.1f} min, threshold {w.threshold}, {w.samples} samples")
    return 0


__all__ = ["FilmWindow", "fieldness", "scores", "longest_run", "find"]

if __name__ == "__main__":
    raise SystemExit(main())
