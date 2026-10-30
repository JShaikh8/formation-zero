"""Turn detected shots into a play index: the join between film time and play-by-play.

Everything downstream — snap detection, participation, personnel, formation — needs to know
which stretch of film is which play. The first method built that mapping in four steps (the one in
use now is at the end of this note).

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

What it got wrong
-----------------
The count matched exactly on the demo game (184), and the count was wrong: the film holds 185
plays (a penalty-nullified snap and a pre-snap false start are on film, the kneel-down that ends
regulation is not). Two plays merged early in the game made up the difference, so about 55 plays
in between were attached to their neighbours' film. That is the insertion-and-deletion failure
this section used to warn about.

The method in use: `build_from_dense`
-------------------------------------
Since 2026-10-30 the pipeline builds the index by take alignment on dense camera readings (see
`formation_zero.perception.align`), counting every row that puts a play on film and letting the
film decide the optional ones. Verified the same day on both games: all 46 kicks (26 and 20) land
on film showing that kick, three frames per take, checked by eye. `build` and the steps above stay
for reference; nothing in the pipeline calls them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from formation_zero.perception.shots import Shot

VIEWS = ("SIDELINE", "ENDZONE")

# Play types that put a snap or a kick on film: shared with the puller so both count the same.
from formation_zero.data.playtypes import FILMABLE_PLAY_TYPES, maybe_on_film, on_film

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

    Nonparametric on purpose: the endzone class is bimodal, reading near 0 when its yard lines
    (which run across the frame) dominate and near 85 when snow hides them and the goalposts and
    upright players do. A KDE represents that without anyone having to name the modes or fit a
    mixture.
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

    Pairs start at the first shot, whichever view it is. The demo game's film leads with the
    sideline angle, as All-22 does; anchoring on a fixed view instead would drop the opening play
    whenever the film starts on the other one, and shift every pairing after it.
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
    """The counted plays (each has a play_uid), in order."""
    return pbp[pbp["play_type"].isin(FILMABLE_PLAY_TYPES)].reset_index(drop=True)


def on_film_rows(pbp):
    """Every row whose play is on film, in order: the counted plays plus nullified snaps."""
    return pbp[[on_film(t, d) for t, d in zip(pbp["play_type"], pbp["desc"])]].reset_index(drop=True)


def film_rows(pbp):
    """Every row whose play is or may be on film, in order, with `film_optional` set on the maybes.

    The maybes are pre-snap fouls (see `maybe_on_film`); the aligner decides from the film itself.
    """
    keep = [on_film(t, d) or maybe_on_film(t, d) for t, d in zip(pbp["play_type"], pbp["desc"])]
    rows = pbp[keep].reset_index(drop=True)
    rows["film_optional"] = [maybe_on_film(t, d) for t, d in zip(rows["play_type"], rows["desc"])]
    return rows


def join(plays: list[Play], pbp, fps: float):
    """Join film plays to play-by-play rows positionally, and report any length mismatch.

    The join is by order alone. A mismatch in length means the film and the play list disagree
    about how many plays happened, and the result should not be trusted anywhere — an extra or
    missing clip shifts everything after it.
    """
    import pandas as pd

    rows = on_film_rows(pbp)
    if len(plays) != len(rows):
        raise ValueError(
            f"{len(plays)} plays on film but {len(rows)} on-film rows in play-by-play. "
            "The ordered join is only valid when these agree; check shot detection and "
            "view labelling before using the index.")

    return pd.DataFrame([{
        "game_key": rows.iloc[p.index]["game_key"],
        "play_uid": rows.iloc[p.index]["play_uid"],
        "play_index": rows.iloc[p.index]["play_index"],
        "film_play_index": p.index,
        "play_type": rows.iloc[p.index]["play_type"],
        "nullified": rows.iloc[p.index]["play_type"] == "no_play",
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


def build_from_dense(dense, shots, pbp, labels):
    """Dense camera readings + takes + view labels + play-by-play -> (play index, report).

    The method that replaced `build` (see `formation_zero.perception.align` for why):

    1. A camera-angle model for this game from its own view labels (colour layout plus line
       direction, `views.dense_features`); it reads every 0.4 s of film.
    2. Cut candidates: every detected cut, plus luma spikes and angle-reading flips as possible
       missed cuts.
    3. The rows that put a play on film (`film_rows`): counted plays, nullified snaps, and optional
       rows (pre-snap fouls, kneel-downs) that some films keep and some drop.
    4. A take alignment for a choice of optional rows, scored by its angle evidence plus how
       predictable the official play types are from the takes it assigns (held out; see
       `perception.playtype`). Starting from every optional row on, one row is flipped at a time
       while the score improves.

    The index keeps its old contract, one row per counted play on film with both takes, so clips
    and per-play stages read it unchanged. No-play rows on film (nullified snaps, kept pre-snap fouls),
    optional rows left off, and plays with a missing take are in the report.
    """
    import pandas as pd

    from formation_zero.perception.align import (
        JUNK,
        align,
        angle_change_candidates,
        missed_cut_candidates,
    )
    from formation_zero.perception.playtype import heldout_loglik, type_class
    from formation_zero.perception.views import ViewModel, dense_features, labelled_samples

    frames, diffs, fps = np.asarray(dense["frames"]), np.asarray(dense["diffs"]), float(dense["fps"])
    features = dense_features(dense["thumbs"], dense["orient"])
    idx, y = labelled_samples(frames, fps, labels)
    if len(set(y.tolist())) < 2:
        raise ValueError("view labels for both camera angles are needed to read this game's film "
                         "(fz-sheets views, one keystroke per take)")
    p = ViewModel.train(features[idx], y).proba(features)
    cuts = sorted({int(np.searchsorted(frames, f)) for f in shots["start_frame"].iloc[1:]})
    missed = sorted(set(missed_cut_candidates(diffs, set(cuts))) | set(angle_change_candidates(p, diffs, set(cuts))))
    rows = film_rows(pbp)
    optional = [int(i) for i in np.where(rows["film_optional"].to_numpy())[0]]

    def evaluate(on: tuple[bool, ...]) -> dict:
        kept = rows.drop(index=[i for i, keep in zip(optional, on) if not keep]).reset_index(drop=True)
        r = align(p, cuts, missed, len(kept))
        plays = []
        for pl in r.plays(len(kept)):
            segs = [sg for sg in (pl["sideline"], pl["endzone"]) if sg is not None]
            if segs:
                plays.append((type_class(kept["play_type"].iloc[pl["index"]]),
                              np.concatenate([np.arange(sg.start, sg.end, 2) for sg in segs])))
        t = heldout_loglik(features, plays)
        return {"on": on, "kept": kept, "alignment": r, "angle_score": r.score, "type_score": t,
                "total": r.score + t}

    tried = {}
    current = evaluate(tuple(True for _ in optional))
    tried[current["on"]] = current
    improved = True
    while improved:
        improved = False
        for k in range(len(optional)):
            on = tuple(not v if j == k else v for j, v in enumerate(current["on"]))
            if on not in tried:
                tried[on] = evaluate(on)
            if tried[on]["total"] > current["total"] + 1e-9:
                current, improved = tried[on], True
    kept, r = current["kept"], current["alignment"]

    def seconds(m: int) -> float:
        return round(float(frames[m] if m < len(frames) else frames[-1] + (frames[-1] - frames[-2])) / fps, 3)

    out, nullified, partial = [], [], []
    for pl in r.plays(len(kept)):
        row = kept.iloc[pl["index"]]
        s, e = pl["sideline"], pl["endzone"]
        spans = {"sideline_start_s": seconds(s.start) if s else None, "sideline_end_s": seconds(s.end) if s else None,
                 "endzone_start_s": seconds(e.start) if e else None, "endzone_end_s": seconds(e.end) if e else None}
        if row["play_type"] == "no_play":
            nullified.append({"film_play_index": pl["index"], "desc": str(row["desc"]), **spans})
            continue
        if not (s and e):
            partial.append({"play_uid": row["play_uid"], "film_play_index": pl["index"], **spans})
            continue
        out.append({"game_key": row["game_key"], "play_uid": row["play_uid"], "play_index": row["play_index"],
                    "film_play_index": pl["index"], "play_type": row["play_type"], "desc": row["desc"],
                    "start_s": min(spans["sideline_start_s"], spans["endzone_start_s"]),
                    "end_s": max(spans["sideline_end_s"], spans["endzone_end_s"]), **spans,
                    "sideline_start_frame": int(frames[s.start]), "endzone_start_frame": int(frames[e.start])})
    report = {
        "method": "take alignment on dense camera readings (perception.align), optional rows by play-type predictability",
        "view_labels": len(labels), "takes": len(shots), "cut_candidates": len(cuts), "missed_cut_candidates": len(missed),
        "rows_on_film": len(kept), "plays_in_index": len(out),
        "optional": [{"row": str(rows["desc"].iloc[i])[:120], "play_type": str(rows["play_type"].iloc[i]),
                      "qtr": int(rows["qtr"].iloc[i]), "time": str(rows["time"].iloc[i]), "on_film": bool(v)}
                     for i, v in zip(optional, current["on"])],
        "choices_scored": [{"on": list(c["on"]), "angle_score": round(c["angle_score"], 1),
                            "type_score": round(c["type_score"], 1)} for c in tried.values()],
        "merged_cuts": len(r.merged_cuts), "missed_cuts_used": len(r.missed_cuts),
        "skipped_slots": r.skipped_slots,
        "junk": [[seconds(sg.start), seconds(sg.end)] for sg in r.segments if sg.view == JUNK],
        "takes_agreeing_with_reading": round(float(np.mean([(p[sg.start:sg.end].mean() >= 0.5) == (sg.view == "SIDELINE")
                                                            for sg in r.segments if sg.view != JUNK])), 3),
        "no_plays_on_film": nullified, "plays_missing_a_take": partial,
    }
    return pd.DataFrame(out), report


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

    from formation_zero.data.layout import GamePaths

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


__all__ = [
    "FILMABLE_PLAY_TYPES",
    "P_SAME_VIEW",
    "VIEWS",
    "Play",
    "build",
    "build_from_dense",
    "emissions_from",
    "film_rows",
    "filmable",
    "join",
    "kde_logpdf",
    "label_views",
    "merge_same_view",
    "on_film_rows",
    "pair_plays",
]


if __name__ == "__main__":
    raise SystemExit(main())
