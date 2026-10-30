"""Detect and track both takes of each play, straight from the source film (milestone 3).

    PYTHONPATH=. .venv-sam3d/bin/python -m formation_zero.perception.tracking \\
        --season 2025 --week 20 --away LA --home CHI --plays 1-10 --weights <best.pt> --overlay

Each take's span comes from the play index and its frames from the full-resolution film, so every
row keeps the film's own frame number (no clips to cut first). Writes, per play and angle,
derived/tracking/<game>_pNNN.<angle>.boxes.parquet (one row per tracked box, in image pixels)
and a .summary.json beside it (`track.bytetrack.summarise`); with --overlay, an MP4 of the take
on the film drive under overlays/.

Take spans from the play index run a few frames past their cut (the index reads the film every
0.4 s; on 49ers-Rams plays 1-10, 19 of 20 takes ran 3 to 17 frames into the next shot). So each take
is cut-checked at full rate (`Cuts`): a cut within TRIM_S of either end trims the take there, and a
cut further in ends every track (the tracker's `cut`). The 576p copy of 49ers-Rams also freezes and
jumps a second at a time in places; a jump reads as a cut, which is right for tracking.

Player tracks whose feet stay off the field (benches, staff) are marked `off_field` on sideline
takes (`vote_field`); they take no team, and the overlay draws them grey without an id.

Teams (`perception.team`): the colour split is fitted once per game on the teacher's player boxes
in the sampled frames, then each player track takes the median of its confident margins over
crops every TEAM_EVERY frames. Team 0 is the lighter-shirted side.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from formation_zero.perception.base import Detector, Tracker

ANGLES = ("sideline", "endzone")
COLUMNS = ["frame", "track_id", "label", "conf", "x1", "y1", "x2", "y2"]
TEAM_EVERY = 5          # frames between the crops a track votes with
TEAM_FIT_FRAMES = 400   # sampled frames whose teacher player boxes fit the game's split
CUT_CHANGE = 12.0       # a cut changes a 64x36 grey thumbnail by at least this much a pixel (on 0-255)...
CUT_RATIO = 4.0         # ...and by this many times the median change over the frames before it
TRIM_S = 1.0            # a cut this close to a take's end trims the take there
FIELD_ANGLES = ("sideline",)    # where the field mask finds the far sideline reliably (GRD-37); end-zone paint fools it
FIELD_EVERY = 10        # frames between the field masks a track votes with
FIELD_MAX_SHARE = 0.5   # a mask that would put more of a frame's boxes off the field than this is not trusted
OFF_FIELD_SHARE = 0.5   # a track whose feet are off the field in at least this share of its votes is off it...
OFF_FIELD_MIN = 2       # ...given at least this many off-field votes


class Cuts:
    """Fed one frame at a time, says whether a hard cut (or a freeze-and-jump) comes just before it.
    Measured on 20 takes: cuts change the thumbnail by 16 to 32, ordinary frames by a median of 0.3 to 2.6."""

    def __init__(self, window: int = 15):
        from collections import deque

        self._last, self._recent = None, deque(maxlen=window)

    def __call__(self, image) -> bool:
        import numpy as np

        grey = image @ np.array([0.299, 0.587, 0.114], np.float32) if image.ndim == 3 else image.astype(np.float32)
        bh, bw = max(1, grey.shape[0] // 36), max(1, grey.shape[1] // 64)
        thumb = grey[:bh * (grey.shape[0] // bh), :bw * (grey.shape[1] // bw)]
        thumb = thumb.reshape(thumb.shape[0] // bh, bh, thumb.shape[1] // bw, bw).mean(axis=(1, 3))   # block means
        if self._last is None:
            self._last = thumb
            return False
        change = float(np.abs(thumb - self._last).mean())
        self._last = thumb
        cut = change > max(CUT_CHANGE, CUT_RATIO * (float(np.median(self._recent)) if self._recent else 0.0))
        self._recent.append(change)
        return cut


def trim_span(start: int, end: int, cuts: list[int], fps: float, edge_s: float = TRIM_S) -> tuple[int, int]:
    """The take [start, end) without the stretch beyond a cut near either end; a cut trims only the
    end it is nearer to, so a short take never trims to nothing."""
    edge = edge_s * fps
    return (max([c for c in cuts if c - start <= edge and c - start < end - c], default=start),
            min([c for c in cuts if end - c <= edge and end - c <= c - start], default=end))


def take_span(play, angle: str, fps: float) -> tuple[int, int]:
    """Source frames [start, end) of one take, from a play index row."""
    return round(getattr(play, f"{angle}_start_s") * fps), round(getattr(play, f"{angle}_end_s") * fps)


def track_take(source, start: int, end: int, detector: Detector, tracker: Tracker, *,
               batch: int = 8) -> tuple[pd.DataFrame, list[int]]:
    """Every tracked box in frames [start, end) of `source` (a FileSource), and the frames that
    open a new shot; the tracker's tracks end at each."""
    tracker.reset()
    rows, buf, cuts, is_cut = [], [], [], Cuts()

    def flush():
        for frame, det in zip(buf, detector([f.image for f in buf])):
            if frame.index in cuts:
                tracker.cut()
            t = tracker.update(det)
            rows.extend((frame.index, int(i), str(lab), round(float(c), 3), *(round(float(v), 1) for v in box))
                        for box, c, lab, i in zip(t.xyxy, t.confidence, t.label, t.track_id))
        buf.clear()

    for frame in source.iter_frames(start, end):
        if is_cut(frame.image):
            cuts.append(frame.index)
        buf.append(frame)
        if len(buf) == batch:
            flush()
    flush()
    from formation_zero.perception.track.bytetrack import vote_labels

    return vote_labels(pd.DataFrame(rows, columns=COLUMNS)), cuts


def fit_teams(paths, *, frames: int = TEAM_FIT_FRAMES, seed: int = 0):
    """The game's colour split, from the teacher's player boxes on a spread of sampled frames."""
    import numpy as np
    from PIL import Image

    from formation_zero.perception.team import ColourTeams

    det = json.loads((paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").read_text())["frames"]
    keys = sorted(det, key=int)
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(frames, len(keys))).astype(int)]
    crops = []
    for f in keys:
        img = np.asarray(Image.open(paths.frames_dir / f"{int(f):06d}.jpg").convert("RGB"))
        crops += [crop(img, b["box"]) for b in det[f] if b["label"] == "player"]
    return ColourTeams(seed).fit(crops)


def crop(image, box):
    x1, y1, x2, y2 = (round(v) for v in box)
    return image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]


def vote_teams(source, start: int, end: int, boxes: pd.DataFrame, teams, *, every: int = TEAM_EVERY) -> dict[int, int]:
    """Track id -> 0/1 for player tracks, from the median of confident margins; -1 where none were."""
    import numpy as np

    from formation_zero.perception.team import UNSURE

    players = boxes[boxes["label"] == "player"]
    margins: dict[int, list[float]] = {int(t): [] for t in players["track_id"].unique()}
    by_frame = dict(tuple(players.groupby("frame")))
    for f in source.iter_frames(start, end, step=every):
        rows = by_frame.get(f.index)
        if rows is None:
            continue
        m = teams.margin([crop(f.image, (r.x1, r.y1, r.x2, r.y2)) for r in rows.itertuples()])
        for t, v in zip(rows["track_id"], m):
            if np.isfinite(v) and abs(v) >= UNSURE:
                margins[int(t)].append(float(v))
    return {t: (int(np.median(v) > 0) if v else -1) for t, v in margins.items()}


def vote_field(source, start: int, end: int, boxes: pd.DataFrame, angle: str, *,
               every: int = FIELD_EVERY) -> set[int]:
    """Player track ids standing off the field (benches, staff, the chain crew): the bottom middle of
    the box outside `perception.field.lines.field_mask` in most of the frames the track is in, one mask
    every `every` frames. Feet past the frame edge count as on. Officials stay (some work the sideline).
    Empty for angles the mask cannot read.

    Checked on 49ers-Rams plays 1-10 (2026-10-30): of 165 tracks it put off the field, nearly all were
    bench players, staff, chain crew and cheerleaders. It misses the near bench whenever the near
    sideline is out of shot (no boundary to cut at); field registration (milestone 4) does that properly."""
    import numpy as np

    from formation_zero.perception.field.lines import field_mask

    if angle not in FIELD_ANGLES or boxes.empty:
        return set()
    votes: dict[int, list[bool]] = {}
    players = set(boxes.loc[boxes["label"] == "player", "track_id"].astype(int))
    by_frame = dict(tuple(boxes.groupby("frame")))
    for f in source.iter_frames(start, end, step=every):
        rows = by_frame.get(f.index)
        mask = field_mask(f.image, angle) if rows is not None else None
        if mask is None:
            continue
        h, w = mask.shape
        x = ((rows["x1"] + rows["x2"]) / 2).astype(int).to_numpy()
        y = rows["y2"].astype(int).to_numpy()
        seen = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        off = np.zeros(len(rows), bool)
        off[seen] = ~mask[y[seen], x[seen]]
        if off.mean() > FIELD_MAX_SHARE:
            continue
        for t, o in zip(rows["track_id"], off):
            votes.setdefault(int(t), []).append(bool(o))
    return {t for t, v in votes.items() if t in players and sum(v) >= OFF_FIELD_MIN and sum(v) >= OFF_FIELD_SHARE * len(v)}


def take_paths(paths, play_index: int, angle: str) -> tuple[Path, Path, Path]:
    stem = f"{paths.game_key}_p{play_index:03d}.{angle}"
    return (paths.tracking_dir / f"{stem}.boxes.parquet", paths.tracking_dir / f"{stem}.summary.json",
            paths.raw_dir / "overlays" / f"p{play_index:03d}.{angle}.mp4")


def run(paths, plays: pd.DataFrame, detector: Detector, *, overlay: bool = False, teams=None, batch: int = 8) -> list[dict]:
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source
    from formation_zero.perception.overlay import render
    from formation_zero.perception.track.bytetrack import ByteTracker, summarise

    paths.tracking_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with FileSource(best_source(paths)) as source:
        tracker = ByteTracker(source.fps)
        for play in plays.itertuples():
            for angle in ANGLES:
                index_span = take_span(play, angle, source.fps)
                t0 = time.perf_counter()
                boxes, cuts = track_take(source, *index_span, detector, tracker, batch=batch)
                secs = time.perf_counter() - t0
                start, end = trim_span(*index_span, cuts, source.fps)
                boxes = boxes[(boxes["frame"] >= start) & (boxes["frame"] < end)].reset_index(drop=True)
                off = vote_field(source, start, end, boxes, angle)
                boxes["off_field"] = boxes["track_id"].isin(off)
                tracks, take = summarise(boxes, source.fps)
                if len(tracks):
                    tracks["off_field"] = tracks["track_id"].isin(off)
                take["off_field"] = len(off)
                on = boxes[~boxes["off_field"]]
                team_of = vote_teams(source, start, end, on, teams) if teams is not None else None
                if team_of is not None:
                    boxes["team"] = boxes["track_id"].map(team_of).fillna(-1).astype(int)
                    tracks["team"] = tracks["track_id"].map(team_of).fillna(-1).astype(int)
                    take["teams"] = {str(k): int(v) for k, v in pd.Series(team_of).value_counts().items()}
                box_path, summary_path, video = take_paths(paths, int(play.play_index), angle)
                boxes.to_parquet(box_path, index=False)
                take.update({"play_uid": play.play_uid, "angle": angle, "start_frame": start, "end_frame": end,
                             "frames": end - start, "index_span": list(index_span),
                             "cuts": [c for c in cuts if start < c < end],
                             "fps_processed": round((index_span[1] - index_span[0]) / secs, 1),
                             "tracks_detail": tracks.to_dict("records")})
                summary_path.write_text(json.dumps(take, indent=1))
                if overlay:
                    render(source, start, end, boxes, video, team_of=team_of)
                out.append(take)
                print(f"  {play.play_uid} {angle}: {end - start} frames (trimmed {index_span[1] - index_span[0] - (end - start)}, "
                      f"{len(take['cuts'])} cuts inside), {take['tracks']} tracks "
                      f"({take['by_label']}, {len(off)} off the field), {take['fragments']} fragments, teams {take.get('teams')}, "
                      f"{take['fps_processed']} fps", flush=True)
    return out


def _plays(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths
    from formation_zero.perception.detect.yolo import YoloDetector
    from formation_zero.perception.track.bytetrack import DETECT_CONF

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--plays", required=True, help="play_index numbers, e.g. 1-10 or 3,7,12")
    ap.add_argument("--weights", required=True, help="YOLO weights: a student's best.pt, or COCO yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--overlay", action="store_true", help="also write an MP4 of each take with the tracks drawn")
    ap.add_argument("--no-teams", action="store_true", help="skip the team split")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    if not paths.film_root_available():
        raise SystemExit(f"film drive not mounted: {paths.film_root}")
    index = pd.read_parquet(paths.play_index_path)
    plays = index[index["play_index"].astype(int).isin(_plays(a.plays))]
    teams = None if a.no_teams else fit_teams(paths)
    run(paths, plays, YoloDetector(a.weights, imgsz=a.imgsz, conf=DETECT_CONF, device=a.device), overlay=a.overlay, teams=teams)
    return 0


__all__ = ["ANGLES", "COLUMNS", "Cuts", "crop", "fit_teams", "run", "take_paths", "take_span", "track_take", "trim_span",
           "vote_field", "vote_teams"]


if __name__ == "__main__":
    raise SystemExit(main())
