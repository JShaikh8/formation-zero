"""Pick frames for the teacher to label, spread over camera angle, phase of play and half of the game.

Architecture section 7.1: about 2,000 frames per game, stratified so the student detector meets every
condition it will see. Halves matter on the snow game: the snow starts mid-game and turns the field
white, the shift that broke the summer's camera-angle feature.

Phase comes from where a frame sits inside its take until snap detection exists (milestone 4). A
take opens on the formation before the snap, the middle is the play, and the camera holds a few
seconds after the whistle, so the opening fifth, the middle third and the last fifth stand in for
pre-snap, live and post-whistle. It is a guess and is stored as `phase_guess`.

Frames are keyed by source frame number. The proxy and any other rendition are frame-aligned with the
source, so the same list can be extracted from whichever file is best (the 1080p copy when there is
one).

    fz-sample --season 2025 --week 20 --away LA --home CHI            # list: derived/labels/<game>.samples.json
    fz-sample --season 2025 --week 20 --away LA --home CHI --extract  # and the JPEGs, on the film drive
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ANGLES = ("sideline", "endzone")
PHASES = {"pre_snap": (0.04, 0.20), "live": (0.35, 0.65), "post_whistle": (0.80, 0.96)}
HALVES = ("first", "second")
GOLDEN = 0.6180339887498949       # offsets inside a phase window: each round lands between earlier ones


def half_of(qtr) -> str:
    return "first" if int(qtr) <= 2 else "second"


def sample_frames(index, pbp, n: int = 2000, fps: float = 25.0) -> list[dict]:
    """About `n` frames from the play index, the same number in every angle x phase x half stratum.

    Plays are spread evenly through each stratum; when a stratum needs more frames than it has plays,
    each further round takes every play again at a new offset inside the phase window. Deterministic.
    """
    qtr = dict(zip(pbp["play_uid"], pbp["qtr"]))
    plays = index.sort_values("film_play_index").to_dict("records")
    strata = [(a, ph, h) for h in HALVES for a in ANGLES for ph in PHASES]
    quota = [n // len(strata) + (1 if k < n % len(strata) else 0) for k in range(len(strata))]
    out, seen = [], set()
    for (angle, phase, half), want in zip(strata, quota):
        pool = [p for p in plays if half_of(qtr[p["play_uid"]]) == half]
        if not pool or want == 0:
            continue
        lo, hi = PHASES[phase]
        for k in range(want):
            rnd, pos = divmod(k, len(pool))
            p = pool[int(np.floor(pos * len(pool) / min(want, len(pool))))] if want < len(pool) else pool[pos]
            u = (0.5 + rnd * GOLDEN) % 1.0
            start, end = p[f"{angle}_start_s"], p[f"{angle}_end_s"]
            q = lo + u * (hi - lo)
            frame = round((start + q * (end - start)) * fps)
            if frame in seen:
                continue
            seen.add(frame)
            out.append({"play_uid": p["play_uid"], "film_play_index": int(p["film_play_index"]), "angle": angle,
                        "phase_guess": phase, "half": half, "frame": frame, "time_s": round(frame / fps, 3),
                        "q": round(q, 3)})
    return sorted(out, key=lambda s: s["frame"])


def extract(samples: list[dict], video: str | Path, out_dir: str | Path, *, quality: int = 92) -> int:
    """Save each sampled frame of `video` as <out_dir>/<frame:06d>.jpg, in one sequential decode.

    Seeking per frame would be slow on a source with a keyframe every few seconds; decoding straight
    through and converting only the wanted frames is faster. Frames already on disk are skipped.
    """
    import av
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s["frame"] for s in samples if not (out_dir / f"{s['frame']:06d}.jpg").exists()}
    if not wanted:
        return 0
    last, saved = max(wanted), 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 25)
        for frame in container.decode(stream):
            i = round(float(frame.pts * stream.time_base) * fps)
            if i in wanted:
                Image.fromarray(frame.to_ndarray(format="rgb24")).save(out_dir / f"{i:06d}.jpg", quality=quality)
                saved += 1
            if i >= last:
                break
    return saved


def main(argv: list[str] | None = None) -> int:
    import argparse

    import pandas as pd

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--extract", action="store_true", help="also save the frames as JPEGs on the film drive")
    ap.add_argument("--video", help="file to extract from (default: the best source rendition)")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)

    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    samples = sample_frames(pd.read_parquet(paths.play_index_path), pd.read_parquet(paths.pbp_path), a.n)
    paths.samples_path.parent.mkdir(parents=True, exist_ok=True)
    paths.samples_path.write_text(json.dumps({"game_key": paths.game_key, "phases": PHASES, "samples": samples}, indent=1))
    print(f"{len(samples)} frames -> {paths.samples_path}")
    if a.extract:
        if not paths.film_root_available():
            raise SystemExit(f"film drive not mounted: {paths.film_root}")
        video = Path(a.video) if a.video else best_source(paths)
        print(f"extracting from {video} -> {paths.frames_dir}")
        print(f"{extract(samples, video, paths.frames_dir)} frames saved")
    return 0


def best_source(paths) -> Path:
    """The highest-resolution source rendition: film_1080p.mp4 when it exists, else film.mp4."""
    hd = paths.source_dir / "film_1080p.mp4"
    return hd if hd.exists() else paths.source_dir / "film.mp4"


__all__ = ["ANGLES", "HALVES", "PHASES", "best_source", "extract", "half_of", "sample_frames"]


if __name__ == "__main__":
    raise SystemExit(main())
