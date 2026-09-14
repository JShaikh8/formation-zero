"""Segment continuous coaches film into shots (the camera's cuts between angles).

Why this exists
---------------
NFL+ All-22 is delivered as one long file, but it is not one long take: each play is shown
twice, sideline then endzone, as two shots separated by a hard cut. Recovering those cuts turns
96 minutes of video into a few hundred bounded segments, which is what makes every later step
tractable — snap detection becomes "find one frame inside a known 15-second shot" instead of
"find the snaps in an hour and a half of film".

What it does and does not decide
--------------------------------
It finds cut boundaries and records a cheap geometric feature (`tilt`) per shot. It deliberately
does **not** label a shot sideline vs. endzone. Two hand-rolled classifiers were tried on the
demo game; both looked convincing on a small sample and then landed near chance across the full
film (52% adjacent-shot alternation, where a strictly alternating film should approach 100%).
The failure mode is that snow begins falling around media 65-70 min and the field turns white,
which changes the yard-line contrast the feature depends on. Labelling therefore waits for
hand-labelled ground truth sampled from *both* halves and a measured held-out accuracy. `tilt`
is stored as raw signal for that work, not as an answer.

Method
------
Cuts: mean absolute luma difference between consecutive sampled frames, downscaled to 64x36.
The threshold wants calibrating per source against its own score distribution; on the demo game
the scores run median 4, p90 10, p99 19, and 15 separates cleanly. A threshold of 28 — which
looks reasonable if you have not looked at the distribution — finds almost nothing.

Tilt: the magnitude-weighted circular mean of edge orientation, reported as 0-90 degrees, where
0 means the dominant lines run horizontally across the image and 90 means they run vertically.
On the demo game sideline shots measure around 88 and endzone shots around 47 — so what actually
dominates the sideline frames is upright players and the vertical LOS/first-down graphics, not
the yard lines. Worth knowing before reading any physical meaning into the number.

Demo game film window
---------------------
`2025_wk20_LA-CHI` is a screen recording: media 0:00-44:37 is unrelated content and the NFL+ menu,
the film runs 44:37-140:51, and 140:51 to the end is a single frozen frame. Scan 2677.0-8452.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Calibrated on 2025_wk20_LA-CHI; see module docstring before reusing on another source.
CUT_THRESHOLD = 15.0

# Shots this short are camera whip or motion blur inside a take, not real cuts. Absorbed into
# the preceding shot so a pan across the field does not fragment into a dozen segments.
MIN_DURATION = 3.0

SAMPLE_STEP = 0.4       # seconds between sampled frames
CUT_SIZE = (64, 36)     # luma difference resolution
TILT_SIZE = (300, 168)  # gradient/orientation resolution

# Fraction of edge magnitudes treated as "strong" — the stripes and player edges, not turf noise.
_EDGE_PERCENTILE = 85


@dataclass(frozen=True)
class Shot:
    """One camera take: a bounded segment of film between two cuts."""

    start: float          # seconds into the media file
    duration: float
    tilt: float | None    # dominant edge-line orientation, 0-90 deg; None if too few edges
    samples: int          # frames sampled inside this shot
    cut_diff: float | None = None   # luma difference at this shot's opening cut (None for the first)

    @property
    def end(self) -> float:
        return self.start + self.duration


def line_tilt(gray: np.ndarray) -> float:
    """Dominant edge-line orientation of a grayscale frame, in degrees from 0 to 90.

    0 means the dominant lines run horizontally across the image, 90 that they run vertically,
    45 that they rake diagonally. Orientation is a mod-180 quantity, so the mean is taken in
    double-angle space — averaging raw angles would cancel lines at +85 and -85 degrees into
    nonsense despite them being nearly parallel.
    """
    gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.hypot(gx, gy)
    strong = magnitude > np.percentile(magnitude, _EDGE_PERCENTILE)
    if strong.sum() < 40:
        return float("nan")
    # A line's orientation is perpendicular to its gradient, hence the quarter turn.
    angles = 2 * (np.arctan2(gy[strong], gx[strong]) + np.pi / 2)
    weights = magnitude[strong]
    mean = np.arctan2((weights * np.sin(angles)).sum(),
                      (weights * np.cos(angles)).sum()) / 2
    return abs(float(np.degrees(mean)))


SPIKE_RATIO = 2.5        # a cut is a difference this many times its neighbours' median...
SPIKE_MIN = 8.0          # ...and at least this large in absolute terms
SPIKE_NEIGHBOURS = 4     # samples on each side that define "neighbours"


def is_cut(differences, *, mode: str = "absolute", cut_threshold: float = CUT_THRESHOLD,
           spike_ratio: float = SPIKE_RATIO, spike_min: float = SPIKE_MIN, neighbours: int = SPIKE_NEIGHBOURS):
    """Boolean per sample. `absolute`: difference above a per-source threshold. `spike`: difference
    is an isolated jump against the median of its neighbours, which is what a cut is and a pan
    is not; this transfers between recordings whose difference levels differ."""
    d = np.asarray(differences, dtype=float)
    if mode == "absolute":
        return d > cut_threshold
    out = np.zeros(len(d), dtype=bool)
    for i in range(1, len(d)):
        nb = np.concatenate([d[max(0, i - neighbours):i], d[i + 1:i + 1 + neighbours]])
        base = float(np.median(nb)) if nb.size else 0.0
        out[i] = d[i] >= spike_min and d[i] >= spike_ratio * max(base, 1.0)
    return out


def group(samples: list[tuple[float, float, float]], *,
          cut_threshold: float = CUT_THRESHOLD,
          min_duration: float = MIN_DURATION,
          step: float = SAMPLE_STEP,
          mode: str = "absolute") -> list[Shot]:
    """Group `(time, difference, tilt)` samples into shots, splitting where difference spikes.

    Each shot's tilt is the median over its middle half. The opening and closing samples of a
    shot straddle the cut and are contaminated by the outgoing or incoming angle, so trimming
    them keeps a short shot from taking its neighbour's orientation.
    """
    if not samples:
        return []

    cuts = is_cut([s[1] for s in samples], mode=mode, cut_threshold=cut_threshold)
    runs: list[list[tuple[float, float]]] = []
    run_cut: list[float | None] = [None]
    current: list[tuple[float, float]] = []
    for (time, difference, tilt), cut in zip(samples, cuts):
        if cut and current:
            runs.append(current)
            run_cut.append(float(difference))
            current = []
        current.append((time, tilt))
    runs.append(current)

    shots = [
        Shot(start=run[0][0],
             duration=run[-1][0] - run[0][0] + step,
             tilt=_median_tilt(run),
             samples=len(run),
             cut_diff=run_cut[i])
        for i, run in enumerate(runs)
    ]
    return _absorb_fragments(shots, min_duration)


def _median_tilt(run: list[tuple[float, float]]) -> float | None:
    """Median tilt over the middle half of a run, ignoring frames with too few edges."""
    lo = len(run) // 4
    core = run[lo:max(lo + 1, 3 * len(run) // 4)]
    values = [tilt for _, tilt in core if tilt == tilt]  # NaN != NaN
    return round(float(np.median(values)), 1) if values else None


def _absorb_fragments(shots: list[Shot], min_duration: float) -> list[Shot]:
    """Fold sub-`min_duration` shots into the preceding shot, extending its duration."""
    kept: list[Shot] = []
    for shot in shots:
        if kept and shot.duration < min_duration:
            previous = kept[-1]
            kept[-1] = Shot(start=previous.start,
                            duration=shot.end - previous.start,
                            tilt=previous.tilt,
                            samples=previous.samples + shot.samples,
                            cut_diff=previous.cut_diff)
        else:
            kept.append(shot)
    return kept


def sample(path, start: float, end: float, step: float = SAMPLE_STEP):
    """Decode `path` from `start` to `end`, yielding `(time, luma_difference, tilt)` per sample.

    Decodes sequentially and drops frames between sample points rather than seeking to each one:
    seeking to every sample would be far slower, and the difference signal needs a fixed spacing
    to stay comparable.
    """
    import av  # optional [cv] dependency — keep the import off the module import path

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    try:
        container.seek(int(start / stream.time_base), stream=stream)
        previous, next_at = None, start
        for frame in container.decode(stream):
            time = float(frame.pts * stream.time_base)
            if time < start:
                continue
            if time > end:
                break
            if time < next_at:
                continue
            next_at = time + step
            small = frame.reformat(width=CUT_SIZE[0], height=CUT_SIZE[1],
                                   format="gray").to_ndarray().astype(np.int16)
            difference = float(np.abs(small - previous).mean()) if previous is not None else 0.0
            previous = small
            yield time, difference, line_tilt(
                frame.reformat(width=TILT_SIZE[0], height=TILT_SIZE[1],
                               format="gray").to_ndarray())
    finally:
        container.close()


def scan(path, start: float, end: float, *,
         step: float = SAMPLE_STEP,
         cut_threshold: float = CUT_THRESHOLD,
         min_duration: float = MIN_DURATION,
         mode: str = "absolute") -> list[Shot]:
    """Decode a film window and return its shots."""
    return group(list(sample(path, start, end, step)),
                 cut_threshold=cut_threshold, min_duration=min_duration, step=step, mode=mode)


def to_frame(shots: list[Shot], game_key: str, fps: float):
    """Shots -> DataFrame, with frame numbers alongside media times.

    Frame numbers matter because the 1080p and proxy renditions of a game are frame-aligned, so
    an index found on the cheap file addresses the same frame in the expensive one.
    """
    import pandas as pd

    return pd.DataFrame([{
        "game_key": game_key,
        "shot_index": index,
        "start_s": round(shot.start, 3),
        "end_s": round(shot.end, 3),
        "duration_s": round(shot.duration, 3),
        "start_frame": int(round(shot.start * fps)),
        "end_frame": int(round(shot.end * fps)),
        "tilt": shot.tilt,
        "samples": shot.samples,
        "cut_diff": shot.cut_diff,
    } for index, shot in enumerate(shots)])


def video_fps(path) -> float:
    """Average frame rate of a video file."""
    import av

    container = av.open(str(path))
    try:
        return float(container.streams.video[0].average_rate)
    finally:
        container.close()


def main(argv: list[str] | None = None) -> int:
    """CLI: scan a film window and write the shot table for a game."""
    import argparse

    from formation_zero.data.layout import GamePaths

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", help="path to the film (use the proxy rendition; it is faster "
                                      "and frame-aligned with the full-resolution file)")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--start", type=float, required=True,
                        help="media seconds where the film begins (excludes any pregame content)")
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cut-threshold", type=float, default=CUT_THRESHOLD)
    args = parser.parse_args(argv)

    paths = GamePaths(args.data_root, args.season, args.week, args.away, args.home).ensure_dirs()
    shots = scan(args.video, args.start, args.end, cut_threshold=args.cut_threshold)
    table = to_frame(shots, paths.game_key, video_fps(args.video))
    table.to_parquet(paths.shots_path, index=False)

    print(f"{len(shots)} shots over {(args.end - args.start) / 60:.1f} min "
          f"-> {paths.shots_path}")
    print(f"median {table['duration_s'].median():.1f}s  "
          f"shortest {table['duration_s'].min():.1f}s  longest {table['duration_s'].max():.1f}s")
    return 0


__all__ = ["Shot", "line_tilt", "group", "is_cut", "sample", "scan", "to_frame", "video_fps",
           "CUT_THRESHOLD", "MIN_DURATION", "SAMPLE_STEP"]


if __name__ == "__main__":
    raise SystemExit(main())
