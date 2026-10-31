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
cut further in ends every track (the tracker's `cut`). The film itself freezes and jumps a second at a
time in places (49ers-Rams p006, the same in the 576p and 1080p copies), and the camera sometimes
zooms out in four frames (p004); both read as cuts, which is right, since box tracking cannot follow
either.

Player tracks whose feet stay off the field (benches, staff) are marked `off_field` on sideline
takes (`vote_field`); they take no team, and the overlay draws them grey without an id. Feet are
judged in yards where the frame registers (`perception.field.homography`), else by the field mask.

Teams (`perception.team`): the colour split is fitted once per game on the teacher's player boxes
in the sampled frames, then each player track takes the median of its confident margins over
crops every TEAM_EVERY frames. Team 0 is the lighter-shirted side. Opposing linemen locked together
often swap ids, so a track whose margins change team for good is first cut there (`swap_cuts`), the
part after the cut taking a new id. Checked 2026-10-31 on 49ers-Rams plays 1-60, sideline takes:
427 of 2,904 player tracks on the field cut (6 twice); by eye 29 of 30 random cuts are real swaps
(a 49er one side, a Ram the other; the other one a running Ram scored weakly, which SWAP_STRONG now
keeps whole). Rerun with the cuts (`field.topdown` holds the rest): players on the wrong side of
the line in the 5 frames before the snap 5 on 4 takes, against 25 on 20. End-zone takes got 299 cuts,
not checked. `--revote` redoes the teams of tracked takes without detecting again.
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
SWAP_MIN_RUN = 4        # an id swap needs this many confident crops on each side of it...
SWAP_AGREE = 0.75       # ...this share of them on that side's team...
SWAP_STRONG = 0.5       # ...and that side's median margin at least this far from zero
CUT_CHANGE = 12.0       # a cut changes a 64x36 grey thumbnail by at least this much a pixel (on 0-255)...
CUT_RATIO = 4.0         # ...and by this many times the median change over the frames before it
TRIM_S = 1.0            # a cut this close to a take's end trims the take there
FIELD_ANGLES = ("sideline",)    # where the field mask finds the far sideline reliably (GRD-37); end-zone paint fools it
FIELD_EVERY = 10        # frames between the field checks a track votes with
FIELD_MAX_SHARE = 0.5   # a mask that would put more of a frame's boxes off the field than this is not trusted
OFF_FIELD_YD = 0.5      # registered frames: feet this far past a sideline are off the field
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


def swap_cuts(frames, margins, *, min_run: int = SWAP_MIN_RUN, agree: float = SWAP_AGREE,
              strong: float = SWAP_STRONG) -> list[int]:
    """Where a track changes team for good, from its confident margins in frame order: the frames from
    which the track is taken to follow someone else (half-way between the last crop of one side and
    the first of the other). The split kept is the one most crops agree with, both sides at least
    `min_run` crops, `agree` of them on their side's team and a median margin `strong` from zero;
    each side is then split again the same way. A swap and a swap back with a long stretch between
    stays whole, since no one split leaves both sides agreeing."""
    import numpy as np

    f, m = np.asarray(frames), np.asarray(margins, float)
    s = np.sign(m)
    cuts: list[int] = []

    def split(lo: int, hi: int) -> None:
        best = None
        for k in range(lo + min_run, hi - min_run + 1):
            ma, mb = float(np.median(m[lo:k])), float(np.median(m[k:hi]))
            if ma * mb >= 0 or min(abs(ma), abs(mb)) < strong:
                continue
            ga, gb = int((s[lo:k] == np.sign(ma)).sum()), int((s[k:hi] == np.sign(mb)).sum())
            if ga < agree * (k - lo) or gb < agree * (hi - k):
                continue
            if best is None or ga + gb > best[0]:
                best = (ga + gb, k)
        if best is None:
            return
        k = best[1]
        split(lo, k)
        cuts.append(int((f[k - 1] + f[k]) // 2))
        split(k, hi)

    split(0, len(m))
    return cuts


def vote_teams(source, start: int, end: int, boxes: pd.DataFrame, teams, *,
               every: int = TEAM_EVERY) -> tuple[pd.DataFrame, dict[int, int]]:
    """Player tracks on the field -> 0/1, from the median of confident margins; -1 where none were.
    Tracks are first cut where they swap ids with an opponent (`swap_cuts`), each part after a cut
    under a new id. Returns the boxes with those ids, and the teams."""
    from itertools import pairwise

    import numpy as np

    from formation_zero.perception.team import UNSURE

    off = boxes["off_field"] if "off_field" in boxes else pd.Series(False, index=boxes.index)
    players = boxes[(boxes["label"] == "player") & ~off]
    votes: dict[int, list[tuple[int, float]]] = {int(t): [] for t in players["track_id"].unique()}
    by_frame = dict(tuple(players.groupby("frame")))
    for f in source.iter_frames(start, end, step=every):
        rows = by_frame.get(f.index)
        if rows is None:
            continue
        m = teams.margin([crop(f.image, (r.x1, r.y1, r.x2, r.y2)) for r in rows.itertuples()])
        for t, v in zip(rows["track_id"], m):
            if np.isfinite(v) and abs(v) >= UNSURE:
                votes[int(t)].append((f.index, float(v)))
    boxes = boxes.copy()
    next_id = int(boxes["track_id"].max()) + 1 if len(boxes) else 0
    team_of: dict[int, int] = {}
    for t, v in votes.items():
        if not v:
            team_of[t] = -1
            continue
        fs, ms = np.array([a for a, _ in v]), np.array([b for _, b in v])
        edges = [None, *swap_cuts(fs, ms), None]
        for i, (a, b) in enumerate(pairwise(edges)):
            part = (fs >= (a if a is not None else -np.inf)) & (fs < (b if b is not None else np.inf))
            tid = t
            if i:
                tid, next_id = next_id, next_id + 1
                rows = (boxes["track_id"] == t) & (boxes["frame"] >= a) & (boxes["frame"] < (b if b is not None else np.inf))
                boxes.loc[rows, "track_id"] = tid
            team_of[tid] = int(np.median(ms[part]) > 0)
    return boxes, team_of


def vote_field(source, start: int, end: int, boxes: pd.DataFrame, angle: str, *,
               every: int = FIELD_EVERY) -> set[int]:
    """Player track ids standing off the field (benches, staff, the chain crew): feet (the bottom
    middle of the box) off the field in most of the frames checked, one every `every` frames.
    Officials stay (some work the sideline). Empty for angles the mask cannot read.

    A frame that registers easily (`perception.field.homography.register`), with the camera where the
    take's other easy frames put it, judges feet in yards: off past OFF_FIELD_YD outside a sideline;
    a box cut off by the bottom of the frame is off only if its bottom edge is already past the near
    sideline. Other frames fall back on the field mask (feet past the frame edge count as on), and
    a mask that would put most of a frame's boxes off is skipped.

    Checked on 49ers-Rams plays 1-10 (2026-10-30). Mask only: of 165 tracks put off the field, nearly
    all were benches, staff, chain crew and cheerleaders, but it missed much of both benches: mapped
    through the homography, its far boundary lands up to 5.6 yd beyond the far sideline (median 1.1;
    often the outer edge of the white border or the team area) and its near one about 3 yd inside
    the near sideline, where the bench crowd hides the ends of the yard lines. In yards (80% of boxes
    on easy frames; about 0.7 s more a frame checked): 256 tracks off against 152, 118 added (50 far
    side, 68 near, none with a median foot inside the field; 116 seen by eye, all benches and sideline
    staff) and 14 dropped, all within 1.4 yd of the near sideline."""
    import numpy as np

    from formation_zero.configs.nfl_field import FIELD_WIDTH_YD
    from formation_zero.perception.field import hashes
    from formation_zero.perception.field import homography as hg

    if angle not in FIELD_ANGLES or boxes.empty:
        return set()
    players = set(boxes.loc[boxes["label"] == "player", "track_id"].astype(int))
    by_frame = dict(tuple(boxes.groupby("frame")))
    checked = []
    for f in source.iter_frames(start, end, step=every):
        rows = by_frame.get(f.index)
        if rows is None:
            continue
        found = hashes.find(f.image, angle)
        fit = hg.register(f.image, angle, found=found)["fit"]
        checked.append((rows, f.image.shape[:2], found["det"]["mask"], fit if fit is not None and fit["easy"] else None))
    cams = np.array([fit["camera"]["C"] for *_, fit in checked if fit is not None]).reshape(-1, 3)
    prior = (float(np.median(cams[:, 1])), float(np.median(cams[:, 2]))) if len(cams) else None
    votes: dict[int, list[bool]] = {}
    for rows, shape, mask, fit in checked:
        xy = rows[["x1", "y1", "x2", "y2"]].to_numpy(float)
        feet, whole = hg.feet(xy, shape)
        if fit is not None and hg.plausible(fit["camera"], prior):
            y = hg.apply(fit["G"], feet)[:, 1]
            off = (y < -OFF_FIELD_YD) | (whole & (y > FIELD_WIDTH_YD + OFF_FIELD_YD))
        elif mask is not None:
            h, w = mask.shape
            x, y = feet[:, 0].astype(int), feet[:, 1].astype(int)
            seen = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            off = np.zeros(len(rows), bool)
            off[seen] = ~mask[y[seen], x[seen]]
            if off.mean() > FIELD_MAX_SHARE:
                continue
        else:
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
    from formation_zero.perception.track.bytetrack import ByteTracker

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
                boxes, tracks, take, team_of = label_take(source, start, end, boxes, off, teams)
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
                      f"{take.get('swaps', 0)} id swaps cut, {take['fps_processed']} fps", flush=True)
    return out


def label_take(source, start: int, end: int, boxes: pd.DataFrame, off: set[int], teams):
    """Off-field flags, teams (after the id-swap cuts) and the per-track summary of one tracked take.
    Returns (boxes, tracks, take, team_of); team_of is None without `teams`. The take's fragments
    count the cuts too."""
    from formation_zero.perception.track.bytetrack import summarise

    boxes = boxes.assign(off_field=boxes["track_id"].isin(off))
    team_of = None
    if teams is not None:
        before = boxes["track_id"].nunique()
        boxes, team_of = vote_teams(source, start, end, boxes, teams)
        swaps = boxes["track_id"].nunique() - before
    tracks, take = summarise(boxes, source.fps)
    if len(tracks):
        tracks["off_field"] = tracks["track_id"].isin(off)
    take["off_field"] = len(off)
    if team_of is not None:
        boxes["team"] = boxes["track_id"].map(team_of).fillna(-1).astype(int)
        tracks["team"] = tracks["track_id"].map(team_of).fillna(-1).astype(int)
        take["teams"] = {str(k): int(v) for k, v in pd.Series(team_of).value_counts().items()}
        take["swaps"] = int(swaps)
    return boxes, tracks, take, team_of


def revote(paths, plays: pd.DataFrame, teams, *, overlay: bool = False) -> list[dict]:
    """The teams again, id-swap cuts included, on takes already tracked: boxes and summary rewritten
    in place (the take's span, cuts and speed kept), the overlay too with `overlay`. No detection."""
    from formation_zero.ingest.sources import FileSource
    from formation_zero.labelling.sample import best_source
    from formation_zero.perception.overlay import render

    out = []
    with FileSource(best_source(paths)) as source:
        for play in plays.itertuples():
            for angle in ANGLES:
                box_path, summary_path, video = take_paths(paths, int(play.play_index), angle)
                if not box_path.exists():
                    continue
                old = json.loads(summary_path.read_text())
                boxes = pd.read_parquet(box_path)
                off = set(boxes.loc[boxes["off_field"], "track_id"].astype(int)) if "off_field" in boxes else set()
                boxes = boxes.drop(columns=["team", "off_field"], errors="ignore")
                boxes, tracks, take, team_of = label_take(source, old["start_frame"], old["end_frame"], boxes, off, teams)
                boxes.to_parquet(box_path, index=False)
                take.update({k: old[k] for k in ("play_uid", "angle", "start_frame", "end_frame", "frames", "index_span",
                                                 "cuts", "fps_processed") if k in old})
                take["tracks_detail"] = tracks.to_dict("records")
                summary_path.write_text(json.dumps(take, indent=1))
                if overlay:
                    render(source, old["start_frame"], old["end_frame"], boxes, video, team_of=team_of)
                out.append(take)
                print(f"  {play.play_uid} {angle}: {take['tracks']} tracks, teams {take.get('teams')}, "
                      f"{take.get('swaps', 0)} id swaps cut", flush=True)
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
    ap.add_argument("--weights", help="YOLO weights: a student's best.pt, or COCO yolo11s.pt (not with --revote)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--overlay", action="store_true", help="also write an MP4 of each take with the tracks drawn")
    ap.add_argument("--no-teams", action="store_true", help="skip the team split")
    ap.add_argument("--revote", action="store_true", help="redo only the teams of takes already tracked")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)
    if not a.revote and not a.weights:
        ap.error("--weights is needed unless --revote")
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    if not paths.film_root_available():
        raise SystemExit(f"film drive not mounted: {paths.film_root}")
    index = pd.read_parquet(paths.play_index_path)
    plays = index[index["play_index"].astype(int).isin(_plays(a.plays))]
    teams = None if a.no_teams else fit_teams(paths)
    if a.revote:
        revote(paths, plays, teams, overlay=a.overlay)
        return 0
    run(paths, plays, YoloDetector(a.weights, imgsz=a.imgsz, conf=DETECT_CONF, device=a.device), overlay=a.overlay, teams=teams)
    return 0


__all__ = ["ANGLES", "COLUMNS", "Cuts", "crop", "fit_teams", "label_take", "revote", "run", "swap_cuts", "take_paths",
           "take_span", "track_take", "trim_span", "vote_field", "vote_teams"]


if __name__ == "__main__":
    raise SystemExit(main())
