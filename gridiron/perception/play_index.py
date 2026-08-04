"""Turn detected shots into a play index: the join between film time and play-by-play.

Everything downstream — snap detection, participation, personnel, formation — needs to know
which stretch of film is which play. This module builds that mapping in four steps.

1. Label each shot sideline or endzone.
2. Merge runs of same-view shots.
3. Pair the result: each play is one sideline shot and one endzone shot.
4. Join the pairs, in order, to the play-by-play rows that were snapped or kicked.

Why a sequence model for step 1
-------------------------------
The per-shot signal is weak — a two-sided threshold on `tilt` scores 84% held-out. The film's
*structure* is not weak: in hand-labelled ground truth, adjacent shots alternate view 37/37 with
no exceptions. Viterbi with a heavy switch prior lets a confident run of neighbours decide a
shot whose own tilt is ambiguous, which a per-shot classifier cannot do. Measured the same way,
it scores 92-96%.

Why step 2 exists
-----------------
Cut detection over-segments: it found 419 shots where 184 plays at two views each need 368. A
spurious cut splits one take into two pieces of the *same* view, so it shows up as a same-view
neighbour — which strict alternation says cannot happen. Merging those runs collapsed 419 to
exactly 368 and took alternation to 367/367. The count agreement is not something this code
arranges; it falls out, which is the reason to believe the step is right rather than convenient.

What is verified and what is not
--------------------------------
Verified on the demo game: the play count matches PBP exactly (184), the first play begins at
the first frame of film, and the last play contains the overtime game-winning field goal at the
media time it is independently known to occupy. Spot-checked: kick plays land on kick formations.
Not verified: that no interior play is off by one. A single insertion in the middle would keep
the count right and still misattach every later play, and nothing here would catch it. Treat the
index as strong evidence, not proof, until interior plays are checked against something external.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridiron.perception.shots import Shot

VIEWS = ("SIDELINE", "ENDZONE")

# Play types that put a snap or a kick on film. Timeouts, the two-minute warning and dead-ball
# penalties produce no clip, so they are not part of the ordered join.
FILMABLE_PLAY_TYPES = ("pass", "run", "punt", "field_goal", "extra_point", "kickoff")

# Probability that two adjacent shots share a view. Ground truth says this is near zero; a small
# non-zero value keeps a genuine repeat from being impossible rather than merely unlikely.
P_SAME_VIEW = 0.10

# KDE bandwidth over tilt, in degrees.
TILT_BANDWIDTH = 5.0


@dataclass(frozen=True)
class Play:
    """One play on film: the same action from two camera angles."""

    index: int                # position in the film's play order, 0-based
    sideline: Shot
    endzone: Shot

    @property
    def start(self) -> float:
        return min(self.sideline.start, self.endzone.start)

    @property
    def end(self) -> float:
        return max(self.sideline.end, self.endzone.end)


def kde_logpdf(x: float, points, bandwidth: float = TILT_BANDWIDTH) -> float:
    """Log density of a Gaussian KDE at `x`.

    Nonparametric on purpose: the sideline class is bimodal, reading near 0 when the yard lines
    dominate a frame and near 85 when upright players and the vertical LOS graphics do. A KDE
    represents that without anyone having to name the modes or fit a mixture.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return float("-inf")
    d = (x - points) / bandwidth
    density = np.exp(-0.5 * d * d).sum() / (points.size * bandwidth * np.sqrt(2 * np.pi))
    return float(np.log(density + 1e-300))


def label_views(tilts, emissions, p_same: float = P_SAME_VIEW) -> list[str]:
    """Most likely view for each shot, by Viterbi over the tilt sequence.

    `emissions` maps each view to the labelled tilt values that define it. A shot with no tilt
    (too few edges to measure) contributes no evidence and is carried by its neighbours.
    """
    if not len(tilts):
        return []
    log_same, log_switch = np.log(p_same), np.log(1 - p_same)

    def emit(tilt, view):
        if tilt is None or tilt != tilt:      # NaN != NaN
            return 0.0
        return kde_logpdf(tilt, emissions[view])

    n = len(tilts)
    delta = np.full((n, 2), -np.inf)
    back = np.zeros((n, 2), dtype=int)
    for j, view in enumerate(VIEWS):
        delta[0, j] = np.log(0.5) + emit(tilts[0], view)
    for i in range(1, n):
        for j, view in enumerate(VIEWS):
            scores = [delta[i - 1, k] + (log_same if k == j else log_switch) for k in range(2)]
            back[i, j] = int(np.argmax(scores))
            delta[i, j] = max(scores) + emit(tilts[i], view)

    path = [int(np.argmax(delta[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return [VIEWS[k] for k in reversed(path)]


def merge_same_view(shots: list[Shot], views: list[str]) -> tuple[list[Shot], list[str]]:
    """Collapse runs of same-view shots into one shot each.

    Strict alternation means two adjacent shots cannot really share a view, so a same-view
    neighbour is a spurious cut inside a single take. Merging repairs the over-segmentation
    that cut detection alone cannot avoid.
    """
    merged: list[Shot] = []
    kept: list[str] = []
    for shot, view in zip(shots, views):
        if merged and kept[-1] == view:
            previous = merged[-1]
            merged[-1] = Shot(start=previous.start,
                              duration=shot.end - previous.start,
                              tilt=previous.tilt,
                              samples=previous.samples + shot.samples)
        else:
            merged.append(shot)
            kept.append(view)
    return merged, kept


def pair_plays(shots: list[Shot], views: list[str]) -> list[Play]:
    """Pair consecutive shots into plays, one view each.

    Pairs start at the first shot: the demo game's film leads with the endzone angle, and
    assuming sideline-first drops the opening play and shifts every pairing after it.
    """
    plays = []
    for index, i in enumerate(range(0, len(shots) - 1, 2)):
        first, second = shots[i], shots[i + 1]
        if views[i] == views[i + 1]:
            raise ValueError(f"shots {i} and {i+1} share view {views[i]!r}; "
                             "run merge_same_view before pairing")
        if views[i] == "SIDELINE":
            plays.append(Play(index, sideline=first, endzone=second))
        else:
            plays.append(Play(index, sideline=second, endzone=first))
    return plays


def filmable(pbp):
    """The play-by-play rows that put something on film, in order."""
    return pbp[pbp["play_type"].isin(FILMABLE_PLAY_TYPES)].reset_index(drop=True)


def join(plays: list[Play], pbp, fps: float):
    """Join film plays to play-by-play rows positionally, and report any length mismatch.

    The join is by order alone. A mismatch in length means the film and the play list disagree
    about how many plays happened, and the result should not be trusted anywhere — an extra or
    missing clip shifts everything after it.
    """
    import pandas as pd

    rows = filmable(pbp)
    if len(plays) != len(rows):
        raise ValueError(
            f"{len(plays)} plays on film but {len(rows)} filmable rows in play-by-play. "
            "The ordered join is only valid when these agree; check shot detection and "
            "view labelling before using the index.")

    return pd.DataFrame([{
        "game_key": rows.iloc[p.index]["game_key"],
        "play_uid": rows.iloc[p.index]["play_uid"],
        "play_index": rows.iloc[p.index]["play_index"],
        "film_play_index": p.index,
        "play_type": rows.iloc[p.index]["play_type"],
        "desc": rows.iloc[p.index]["desc"],
        "start_s": round(p.start, 3),
        "end_s": round(p.end, 3),
        "sideline_start_s": round(p.sideline.start, 3),
        "sideline_end_s": round(p.sideline.end, 3),
        "endzone_start_s": round(p.endzone.start, 3),
        "endzone_end_s": round(p.endzone.end, 3),
        "sideline_start_frame": int(round(p.sideline.start * fps)),
        "endzone_start_frame": int(round(p.endzone.start * fps)),
    } for p in plays])


def emissions_from(labels) -> dict[str, list[float]]:
    """Build the per-view tilt sets a Viterbi pass needs, from ground-truth label records."""
    return {v: [r["tilt"] for r in labels
                if r["view"] == v and r.get("tilt") is not None and r["tilt"] == r["tilt"]]
            for v in VIEWS}


def build(shots: list[Shot], labels, pbp, fps: float):
    """Shots plus ground-truth view labels -> the play index. Raises if counts disagree."""
    views = label_views([s.tilt for s in shots], emissions_from(labels))
    merged, merged_views = merge_same_view(shots, views)
    return join(pair_plays(merged, merged_views), pbp, fps), merged, merged_views


def _labels_for(shots: list[Shot], path) -> list[dict]:
    """Attach time-keyed ground-truth labels to the shots they fall inside.

    Labels are keyed by media time rather than shot index precisely so they survive changes to
    the detector; the cost is having to locate them again here.
    """
    import json

    out = []
    for label in json.load(open(path))["labels"]:
        if not label.get("confident", True):
            continue
        midpoint = (label["start_s"] + label["end_s"]) / 2
        for shot in shots:
            if shot.start <= midpoint < shot.end:
                out.append({**label, "tilt": shot.tilt})
                break
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: build and write the play index for a game."""
    import argparse

    import pandas as pd

    from gridiron.data.layout import GamePaths

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    paths = GamePaths(args.data_root, args.season, args.week, args.away, args.home)
    table = pd.read_parquet(paths.shots_path)
    shots = [Shot(start=r.start_s, duration=r.duration_s,
                  tilt=None if pd.isna(r.tilt) else float(r.tilt), samples=int(r.samples))
             for r in table.itertuples()]
    labels = _labels_for(shots, paths.groundtruth_dir / f"{paths.game_key}.shot_views.json")
    pbp = pd.read_parquet(paths.pbp_path)

    index, merged, views = build(shots, labels, pbp, args.fps)
    index.to_parquet(paths.play_index_path, index=False)

    switches = sum(views[i] != views[i + 1] for i in range(len(views) - 1))
    print(f"{len(shots)} shots -> {len(merged)} after merging same-view runs "
          f"(alternation {switches}/{len(views)-1})")
    print(f"{len(index)} plays -> {paths.play_index_path}")
    print(index["play_type"].value_counts().to_dict())
    return 0


__all__ = ["Play", "VIEWS", "FILMABLE_PLAY_TYPES", "P_SAME_VIEW", "kde_logpdf", "label_views",
           "merge_same_view", "pair_plays", "filmable", "join", "emissions_from", "build"]


if __name__ == "__main__":
    raise SystemExit(main())
