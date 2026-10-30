"""Align the film's camera takes to its expected structure: one sideline and one end-zone take per play.

Why alignment
-------------
Cut detection and the camera-angle classifier are each good and neither is exact. On the clear
game the film holds 286 takes where 141 plays need 282, and the count hides two kinds of defect
that cancel: stray fragments (a false cut inside one take) and missed cuts (one take holding both
angles). Merging the weakest same-angle boundaries to the target count got the count exactly right
and the angles 60% right, flipping at a missed cut and flipping back at a fragment. Repairing one
defect at a time oscillated, because the recording opens on a minute of pregame from one camera
where alternation is wrong by nature.

So the problem is set as sequence alignment, the way gene sequences are aligned: the expected
sequence is `S E S E ...`, two slots per filmable play, and the observed sequence is the film
sampled every 0.4 s with a camera-angle reading at each sample. A dynamic programme chooses where
the take boundaries are and which slot each take fills, scoring every choice:

- each sample's angle reading against the slot's angle (down-weighted, because readings inside one
  take are strongly correlated);
- each detected cut kept (cheap) or declared false (expensive), so fragments merge only when the
  angle readings demand it;
- each weaker luma jump inside a take used as a missed cut (expensive) or not (nearly free);
- a take length prior, footage outside any play (cheap before the first play and after the last,
  expensive in between), and a slot with no take (expensive).

The play count comes from the official play list, so the alignment cannot drift by a play without
paying for a missing slot and an unexplained stretch of film somewhere. That is the property the
ordered join needs, and it is the reason to prefer this to local repair.

What it does not verify: that a play's two takes are the right official play. A missing take and an
extra one in the middle can still cancel. The review sheet (each play's frames next to its official
description) is the check for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIDELINE, ENDZONE, JUNK = "SIDELINE", "ENDZONE", "JUNK"

# Evidence weight per sample. Readings 0.4 s apart inside one take are nearly the same picture, so
# a 15-second take counts as a handful of independent observations, not 37.
EVIDENCE_WEIGHT = 0.15
PROBA_CLIP = 0.03            # a single reading can never be certain

P_CUT_REAL = 0.97            # a detected cut is a real cut...
P_MISSED_CUT = 0.03          # ...a weaker jump inside a take is rarely one
MISSED_RATIO = 2.0           # a jump counts as a candidate at this multiple of its neighbours' median...
MISSED_MIN = 4.0             # ...and at least this absolute luma change
NEIGHBOURS = 4               # samples each side that define "neighbours"

JUNK_EDGE = 2.0              # log cost of footage before the first play or after the last
JUNK_MID = 12.0              # log cost of unexplained footage between plays
JUNK_RATE = 0.01             # and per sample of it, so unexplained stretches stay short
LOG_P_SKIP = -12.0           # log probability of a slot with no take
LOG_P_SKIP_OPTIONAL = -0.7   # ...when the play may or may not be on film (a pre-snap foul)

TAKE_MEDIAN_S = 13.0         # take length prior: log-normal
TAKE_SIGMA = 0.6
TAKE_MIN_S = 4.0             # shorter is not a take: a split that leaves less is a fake
TAKE_MAX_S = 75.0            # longer is not a take (long kick returns run about 40 s)


@dataclass(frozen=True)
class Segment:
    """A stretch of film between two chosen boundaries, assigned to a slot or to junk."""

    start: int                # sample index, inclusive
    end: int                  # sample index, exclusive
    view: str                 # SIDELINE, ENDZONE or JUNK
    slot: int | None          # 1-based slot in the expected sequence; None for junk


@dataclass
class Alignment:
    segments: list[Segment]
    skipped_slots: list[int]
    score: float
    merged_cuts: list[int] = field(default_factory=list)     # sample indices of cuts declared false
    missed_cuts: list[int] = field(default_factory=list)     # sample indices of jumps used as cuts

    def plays(self, n_plays: int, first: str = SIDELINE) -> list[dict]:
        """One row per play: its two takes as sample ranges, None where a slot was skipped."""
        by_slot = {s.slot: s for s in self.segments if s.slot is not None}
        second = ENDZONE if first == SIDELINE else SIDELINE
        out = []
        for p in range(n_plays):
            a, b = by_slot.get(2 * p + 1), by_slot.get(2 * p + 2)
            out.append({"index": p, first.lower(): a, second.lower(): b})
        return out


def missed_cut_candidates(diffs: np.ndarray, exclude: set[int], *, ratio: float = MISSED_RATIO,
                          minimum: float = MISSED_MIN, neighbours: int = NEIGHBOURS) -> list[int]:
    """Samples whose luma change spikes against their neighbours but were not detected as cuts.

    A missed cut is a real cut that fell under the detector's threshold, so it still looks like a
    spike, just a smaller one. Every such spike is offered to the aligner at a high price.
    """
    out = []
    n = len(diffs)
    near = {e + d for e in exclude for d in (-1, 0, 1)}
    for m in range(1, n):
        if m in near:
            continue
        lo, hi = max(0, m - neighbours), min(n, m + neighbours + 1)
        around = np.concatenate([diffs[lo:m], diffs[m + 1:hi]])
        if not len(around):
            continue
        if diffs[m] >= minimum and diffs[m] >= ratio * max(float(np.median(around)), 1e-6) \
                and diffs[m] == diffs[max(0, m - 1):m + 2].max():
            out.append(m)
    return out


def angle_change_candidates(p_sideline: np.ndarray, diffs: np.ndarray, exclude: set[int], *,
                            smooth: int = 5, snap: int = 3) -> list[int]:
    """Samples where the angle reading flips and stays flipped, placed on the nearest luma jump.

    A cut made while the camera is moving can be a small spike against already-large neighbours
    and escape both detectors. The angle readings still change across it, so wherever the
    smoothed reading crosses 0.5 a cut is offered at the biggest luma change within `snap`
    samples, at the same price as any other missed cut.
    """
    p = np.asarray(p_sideline, dtype=float)
    if len(p) < smooth:
        return []
    half = smooth // 2
    padded = np.pad(p, half, mode="edge")
    med = np.array([np.median(padded[m:m + smooth]) for m in range(len(p))])
    near = {e + d for e in exclude for d in range(-snap, snap + 1)}
    out = []
    for m in np.where((med[1:] >= 0.5) != (med[:-1] >= 0.5))[0] + 1:
        lo, hi = max(1, m - snap), min(len(p), m + snap + 1)
        c = lo + int(np.argmax(diffs[lo:hi]))
        if c not in near and (not out or out[-1] != c):
            out.append(c)
    return out


def _duration_logprior(seconds: float) -> float:
    if seconds < TAKE_MIN_S or seconds > TAKE_MAX_S:
        return -np.inf
    z = (np.log(seconds) - np.log(TAKE_MEDIAN_S)) / TAKE_SIGMA
    return float(-0.5 * z * z)


def align(p_sideline: np.ndarray, cuts: list[int], missed: list[int], n_plays: int, *,
          step_s: float = 0.4, first: str = SIDELINE, optional=None) -> Alignment:
    """Best alignment of sampled film to `n_plays` plays of (first, other) takes.

    `p_sideline[m]` is the classifier's reading at sample m; `cuts` are sample indices where a
    detected take starts (excluding sample 0); `missed` are weaker jumps offered as missed cuts.
    `optional[i]` marks play i as possibly absent from the film: its slots skip nearly free.
    """
    p = np.clip(np.asarray(p_sideline, dtype=float), PROBA_CLIP, 1 - PROBA_CLIP)
    m_total = len(p)
    log_s = EVIDENCE_WEIGHT * np.log(p)
    log_e = EVIDENCE_WEIGHT * np.log(1 - p)
    # Footage outside any play is still from one of the two cameras: it explains its readings as
    # well as the better angle does, and pays for existing through JUNK_EDGE / JUNK_MID instead.
    log_j = np.maximum(log_s, log_e) - JUNK_RATE
    cum = {SIDELINE: np.concatenate([[0.0], np.cumsum(log_s)]),
           ENDZONE: np.concatenate([[0.0], np.cumsum(log_e)]),
           JUNK: np.concatenate([[0.0], np.cumsum(log_j)])}

    # Boundaries: film start, every candidate, film end. Each interior one has a prior of being real.
    real_p = {c: P_CUT_REAL for c in cuts if 0 < c < m_total}
    for c in missed:
        if 0 < c < m_total and c not in real_p:
            real_p[c] = P_MISSED_CUT
    bounds = [0] + sorted(real_p) + [m_total]
    k_b = len(bounds)
    log_real = np.array([0.0] + [np.log(real_p[b]) for b in bounds[1:-1]] + [0.0])
    log_false = np.array([0.0] + [np.log(1 - real_p[b]) for b in bounds[1:-1]] + [0.0])
    cum_false = np.concatenate([[0.0], np.cumsum(log_false)])     # cum_false[j] = sum log_false[<j]

    n_slots = 2 * n_plays
    other = ENDZONE if first == SIDELINE else SIDELINE
    slot_view = [None] + [first if s % 2 == 1 else other for s in range(1, n_slots + 1)]
    is_first = np.array([slot_view[s] == first for s in range(1, n_slots + 1)])
    junk_cost = np.full(n_slots + 1, -JUNK_MID)
    junk_cost[0] = junk_cost[n_slots] = -JUNK_EDGE

    neg = -np.inf
    score = np.full((k_b, n_slots + 1), neg)
    back_i = np.full((k_b, n_slots + 1), -1, dtype=np.int32)
    back_kind = np.zeros((k_b, n_slots + 1), dtype=np.int8)     # 1 take, 2 junk, 3 skip
    score[0, 0] = 0.0
    max_samples = int(np.ceil(TAKE_MAX_S / step_s))
    bounds_arr = np.array(bounds)

    opt = np.zeros(n_plays, dtype=bool) if optional is None else np.asarray(optional, dtype=bool)
    skip_cost = np.where(np.repeat(opt, 2), LOG_P_SKIP_OPTIONAL, LOG_P_SKIP)   # per slot, 1-based below

    def skip_slots(j):
        for k in range(1, n_slots + 1):
            cand = score[j, k - 1] + skip_cost[k - 1]
            if cand > score[j, k]:
                score[j, k] = cand
                back_i[j, k] = j
                back_kind[j, k] = 3

    skip_slots(0)
    for j in range(1, k_b):
        bj = bounds[j]
        i_lo = int(np.searchsorted(bounds_arr, bj - max_samples, side="left"))
        for i in range(max(0, i_lo), j):
            row = score[i]
            if not np.isfinite(row).any():
                continue
            bi = bounds[i]
            between = cum_false[j] - cum_false[i + 1]                 # cuts skipped over are false
            edge = log_real[j] + between
            # a take filling the next slot
            dur = _duration_logprior((bj - bi) * step_s)
            if np.isfinite(dur):
                ev_first = cum[first][bj] - cum[first][bi]
                ev_other = cum[other][bj] - cum[other][bi]
                gain = np.where(is_first, ev_first, ev_other) + dur + edge
                cand = row[:-1] + gain
                better = cand > score[j, 1:]
                score[j, 1:][better] = cand[better]
                back_i[j, 1:][better] = i
                back_kind[j, 1:][better] = 1
            # footage that belongs to no play, slot count unchanged; junk may be any length
            cand = row + (cum[JUNK][bj] - cum[JUNK][bi]) + junk_cost + edge
            better = cand > score[j]
            score[j][better] = cand[better]
            back_i[j][better] = i
            back_kind[j][better] = 2
        # very long junk (pregame) can exceed the take window: allow it from the film start
        if i_lo > 0:
            for i in (0,):
                row = score[i]
                cand = row + (cum[JUNK][bj] - cum[JUNK][bounds[i]]) + junk_cost \
                    + log_real[j] + (cum_false[j] - cum_false[i + 1])
                better = cand > score[j]
                score[j][better] = cand[better]
                back_i[j][better] = i
                back_kind[j][better] = 2
        skip_slots(j)

    total = score[k_b - 1, n_slots]
    if not np.isfinite(total):
        raise ValueError(f"no alignment of {m_total} samples to {n_plays} plays fits the constraints")

    segments: list[Segment] = []
    skipped: list[int] = []
    used: set[int] = set()
    j, k = k_b - 1, n_slots
    while j > 0 or k > 0:
        kind, i = int(back_kind[j, k]), int(back_i[j, k])
        if kind == 3:
            skipped.append(k)
            k -= 1
            continue
        if kind == 1:
            segments.append(Segment(bounds[i], bounds[j], slot_view[k], k))
            k -= 1
        elif kind == 2:
            segments.append(Segment(bounds[i], bounds[j], JUNK, None))
        else:
            raise RuntimeError(f"broken backtrack at boundary {j}, slot {k}")
        used.add(bounds[j])
        j = i
    segments.reverse()
    chosen = used - {m_total}
    return Alignment(segments=segments, skipped_slots=sorted(skipped), score=float(total),
                     merged_cuts=sorted(c for c in cuts if 0 < c < m_total and c not in chosen),
                     missed_cuts=sorted(c for c in chosen if real_p.get(c) == P_MISSED_CUT))


__all__ = ["ENDZONE", "JUNK", "SIDELINE", "Alignment", "Segment", "align", "missed_cut_candidates"]
