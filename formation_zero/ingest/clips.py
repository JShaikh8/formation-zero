"""Cut one clip per play per angle from the continuous film, using the play index.

    fz-clips --season 2025 --week 20 --away LA --home CHI --video source.mp4

Reads derived/shots/<game>.plays.parquet (sideline and end-zone spans per play), writes
raw/<season>/wk<NN>/<AWAY>_at_<HOME>/<angle>/pNNN.mp4 with a pad on both sides, re-encoded so
each clip starts on a keyframe and seeking inside it is exact, and a manifest mapping clip
frame 0 back to the source frame so nothing loses the film's own frame numbers.
"""

from __future__ import annotations

from pathlib import Path

from formation_zero.ingest.encode import VideoWriter
from formation_zero.ingest.sources import FileSource

ANGLES = ("sideline", "endzone")


def cut(source: FileSource, dst: str | Path, start_s: float, end_s: float, *, pad_s: float = 1.5) -> dict:
    """Write frames from start_s - pad to end_s + pad into dst. Returns the clip's manifest row."""
    start_f = max(0, int(round((start_s - pad_s) * source.fps)))
    end_f = int(round((end_s + pad_s) * source.fps))
    if source.frames is not None:
        end_f = min(end_f, source.frames)
    writer = None
    n = 0
    for frame in source.iter_frames(start_f, end_f):
        if writer is None:
            writer = VideoWriter(dst, source.fps, frame.width, frame.height)
        writer.write(frame.image)
        n += 1
    if writer is not None:
        writer.close()
    return {"clip": str(dst), "source_start_frame": start_f, "source_end_frame": start_f + n,
            "frames": n, "fps": source.fps, "pad_s": pad_s}


def cut_game(video: str | Path, plays, paths, *, pad_s: float = 1.5, only: set[str] | None = None) -> list[dict]:
    """Cut every play in the play index table. `paths` is a GamePaths; `only` limits to play_uids."""
    rows = []
    with FileSource(video) as source:
        for r in plays.itertuples():
            if only and r.play_uid not in only:
                continue
            for angle in ANGLES:
                start_s, end_s = getattr(r, f"{angle}_start_s"), getattr(r, f"{angle}_end_s")
                dst = paths.clip_path(angle, int(r.play_index))
                row = cut(source, dst, float(start_s), float(end_s), pad_s=pad_s)
                row.update({"play_uid": r.play_uid, "play_index": int(r.play_index), "angle": angle,
                            "play_start_s": float(start_s), "play_end_s": float(end_s)})
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    import pandas as pd

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--video", required=True, help="the full-resolution source film")
    ap.add_argument("--pad", type=float, default=1.5)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)

    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home).ensure_dirs()
    plays = pd.read_parquet(paths.play_index_path)
    rows = cut_game(a.video, plays, paths, pad_s=a.pad)
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(paths.clips_manifest_path, index=False)
    print(f"{len(rows)} clips ({len(plays)} plays x {len(ANGLES)} angles) -> {paths.clips_manifest_path}")
    return 0


__all__ = ["ANGLES", "cut", "cut_game"]

if __name__ == "__main__":
    raise SystemExit(main())
