"""Camera angle per take: sideline or end zone, from a tiny learned model on frame layout.

Why learned: the hand-tuned edge-orientation feature from the summer read 88/47 on the snow
game and 2 to 6 / 15 to 33 on a clear one, and two hand-built replacements failed the same
afternoon. The difference between the two NFL+ cameras is obvious to a person in a thumbnail
(far-sideline wall at the top vs the crowd band; rotated vs upright yard numbers), so it is
obvious to a linear model on layout features. Labels come from contact sheets, one keystroke each.

Features: from a 32x18 RGB thumbnail, the mean colour of every row (18x3) and every column
(32x3), plus the row-to-row brightness change (17). 167 numbers, no image library.
Model: L2-regularised logistic regression by gradient descent, saved as a small .npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VIEWS = ("SIDELINE", "ENDZONE")   # class 1 = SIDELINE
THUMB = (32, 18)


def thumbnail(image: np.ndarray, size=THUMB) -> np.ndarray:
    h, w = image.shape[:2]
    ys = (np.arange(size[1]) * h / size[1]).astype(int).clip(0, h - 1)
    xs = (np.arange(size[0]) * w / size[0]).astype(int).clip(0, w - 1)
    return image[ys][:, xs].astype(np.float32) / 255.0


def layout_features(image_or_thumb: np.ndarray) -> np.ndarray:
    t = image_or_thumb if image_or_thumb.shape[:2] == (THUMB[1], THUMB[0]) and image_or_thumb.dtype != np.uint8 \
        else thumbnail(image_or_thumb)
    rows = t.mean(axis=1).ravel()                        # 18 x 3
    cols = t.mean(axis=0).ravel()                        # 32 x 3
    bright = t.mean(axis=(1, 2))
    return np.concatenate([rows, cols, np.diff(bright)]).astype(np.float32)


class ViewModel:
    def __init__(self, w: np.ndarray, b: float, mu: np.ndarray, sd: np.ndarray):
        self.w, self.b, self.mu, self.sd = w, float(b), mu, sd

    @classmethod
    def train(cls, X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, steps: int = 3000, lr: float = 0.1) -> "ViewModel":
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        w = np.zeros(Z.shape[1]); b = 0.0
        n = len(y)
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = p - y
            w -= lr * (Z.T @ g / n + l2 * w / n)
            b -= lr * g.mean()
        return cls(w, b, mu, sd)

    def proba(self, X: np.ndarray) -> np.ndarray:
        """P(SIDELINE) per row."""
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))

    def save(self, path: str | Path) -> None:
        np.savez(path, w=self.w, b=self.b, mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path: str | Path) -> "ViewModel":
        d = np.load(path)
        return cls(d["w"], float(d["b"]), d["mu"], d["sd"])


def emissions_from_proba(p_sideline: np.ndarray, floor: float = 0.02) -> list[dict[str, float]]:
    """Per-take log-likelihoods for the Viterbi labeller in play_index."""
    p = np.clip(p_sideline, floor, 1 - floor)
    return [{"SIDELINE": float(np.log(a)), "ENDZONE": float(np.log(1 - a))} for a in p]


def label_views_viterbi(emissions: list[dict[str, float]], p_same: float = 0.10) -> list[str]:
    """Most likely view per take under a strong alternation prior, from per-take log-likelihoods."""
    if not emissions:
        return []
    log_same, log_switch = np.log(p_same), np.log(1 - p_same)
    n = len(emissions)
    delta = np.full((n, 2), -np.inf); back = np.zeros((n, 2), dtype=int)
    for j, v in enumerate(VIEWS):
        delta[0, j] = np.log(0.5) + emissions[0][v]
    for i in range(1, n):
        for j, v in enumerate(VIEWS):
            scores = [delta[i - 1, k] + (log_same if k == j else log_switch) for k in range(2)]
            back[i, j] = int(np.argmax(scores)); delta[i, j] = max(scores) + emissions[i][v]
    path = [int(np.argmax(delta[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return [VIEWS[k] for k in reversed(path)]


def take_thumbnails(proxy_path, shots, first_frame: int = 0) -> np.ndarray:
    """One 32x18 thumbnail per take (mid frame), from the proxy."""
    from formation_zero.ingest.sources import FileSource

    out = []
    with FileSource(proxy_path) as src:
        for r in shots.itertuples():
            mid = int((r.start_frame + r.end_frame) // 2) - first_frame
            fr = next(src.iter_frames(max(0, mid), max(0, mid) + 1))
            out.append(thumbnail(fr.image))
    return np.array(out)


__all__ = ["VIEWS", "thumbnail", "layout_features", "ViewModel", "emissions_from_proba", "label_views_viterbi", "take_thumbnails", "pair_by_count", "per_frame_features", "split_missed_cuts"]


def pair_by_count(shots, p_sideline, n_plays: int, *, p_same: float = 0.10):
    """Label, then merge only the weakest same-view boundaries until there are two takes per play.

    Returns (merged_shots, views, report). Merging is count-guided because the official play list
    says exactly how many plays are on film; where the classifier says two neighbours share a view,
    the boundary with the smallest luma jump is the likeliest false cut.
    """
    from formation_zero.perception.shots import Shot

    views = label_views_viterbi(emissions_from_proba(np.asarray(p_sideline)), p_same=p_same)
    need = len(shots) - 2 * n_plays
    report = {"takes": len(shots), "target": 2 * n_plays, "merges_needed": need}
    if need < 0:
        report["status"] = "too_few_takes"
        return list(shots), list(views), report
    cands = [(shots[i + 1].cut_diff if shots[i + 1].cut_diff is not None else 0.0, i)
             for i in range(len(shots) - 1) if views[i] == views[i + 1]]
    cands.sort()
    chosen = {i for _, i in cands[:need]}
    merged, mv = [], []
    i = 0
    while i < len(shots):
        sh = shots[i]; vw = views[i]
        while i in chosen:
            nxt = shots[i + 1]
            sh = Shot(start=sh.start, duration=nxt.end - sh.start, tilt=sh.tilt, samples=sh.samples + nxt.samples, cut_diff=sh.cut_diff)
            i += 1
        merged.append(sh); mv.append(vw); i += 1
    report.update({"same_view_boundaries": len(cands), "merged": len(chosen),
                   "status": "ok" if len(merged) == 2 * n_plays else "count_mismatch"})
    if len(merged) == 2 * n_plays:
        # The film strictly alternates angles. With the count exact, the only free choice is which
        # angle comes first; the classifier decides that by total agreement, and its per-take
        # mistakes stop mattering for the pairing.
        p = np.asarray(p_sideline)
        keep = [i for i in range(len(shots)) if i not in {c + 1 for c in chosen}]
        pk = np.clip(p[keep], 0.02, 0.98)
        starts = {"SIDELINE": 0.0, "ENDZONE": 0.0}
        for first in starts:
            for k, pr in enumerate(pk):
                is_s = (k % 2 == 0) == (first == "SIDELINE")
                starts[first] += float(np.log(pr if is_s else 1 - pr))
        first = max(starts, key=starts.get)
        mv = [VIEWS[(k + (0 if first == "SIDELINE" else 1)) % 2] for k in range(len(merged))]
        report["classifier_agreement"] = round(float(np.mean([(pr >= 0.5) == (vw == "SIDELINE") for pr, vw in zip(pk, mv)])), 3)
        report["first_view"] = first
    return merged, mv, report


def per_frame_features(proxy_path, shots, first_frame: int = 0, fractions=(0.15, 0.5, 0.85)) -> np.ndarray:
    """Layout features at several points inside every take: (takes, len(fractions), features)."""
    from formation_zero.ingest.sources import FileSource

    out = []
    with FileSource(proxy_path) as src:
        for sh in shots:
            fs = []
            for q in fractions:
                f = int(round(sh.start + q * sh.duration) * src.fps) - first_frame
                fr = next(src.iter_frames(max(0, f), max(0, f) + 1))
                fs.append(layout_features(fr.image))
            out.append(np.array(fs))
    return np.array(out)


def split_missed_cuts(proxy_path, shots, model, first_frame: int = 0, *, margin: float = 0.25,
                      min_jump: float = 6.0):
    """A take whose first and last frames the classifier reads as different angles contains a missed
    cut. Split it at the strongest luma jump inside it. Returns (new_shots, n_split)."""
    from formation_zero.perception.shots import Shot, sample

    feats = per_frame_features(proxy_path, shots, first_frame, fractions=(0.15, 0.85))
    out, n_split = [], 0
    fps = None
    for sh, f in zip(shots, feats):
        p0, p1 = model.proba(f[0])[0], model.proba(f[1])[0]
        if (p0 - 0.5) * (p1 - 0.5) < 0 and abs(p0 - 0.5) > margin and abs(p1 - 0.5) > margin and sh.duration > 4.0:
            rows = list(sample(proxy_path, sh.start - first_frame / 25.0, sh.end - first_frame / 25.0))
            if len(rows) > 4:
                inner = rows[2:-1]
                t, d = max(inner, key=lambda r: r[1])[:2]
                if d >= min_jump:
                    t_src = t + first_frame / 25.0
                    out.append(Shot(start=sh.start, duration=t_src - sh.start, tilt=sh.tilt, samples=sh.samples // 2, cut_diff=sh.cut_diff))
                    out.append(Shot(start=t_src, duration=sh.end - t_src, tilt=sh.tilt, samples=sh.samples // 2, cut_diff=float(d)))
                    n_split += 1
                    continue
        out.append(sh)
    return out, n_split


def repair_sequence(proxy_path, shots, model, first_frame: int = 0, *, window: int = 11, max_iter: int = 6):
    """Fix the take sequence where strict alternation and the classifier disagree in a long stretch.

    An inverted stretch starts where a cut was missed (one take holds both angles: split it at its
    strongest internal jump) or where a stray fragment was inserted (merge it into its neighbour).
    Change points come from a running mean of per-take agreement with strict alternation.
    """
    from formation_zero.perception.shots import Shot, sample

    shots = list(shots)
    log = []
    for it in range(max_iter):
        feats = per_frame_features(proxy_path, shots, first_frame)
        p = np.array([model.proba(f).mean() for f in feats])
        best = None
        for first in (0, 1):
            agree = np.array([(pr >= 0.5) == (((k + first) % 2) == 0) for k, pr in enumerate(p)], dtype=float)
            if best is None or agree.mean() > best[1].mean():
                best = (first, agree)
        first, agree = best
        if len(agree) < window:
            break
        run = np.convolve(agree, np.ones(window) / window, mode="same")
        flips = [i for i in range(1, len(run)) if (run[i - 1] >= 0.5) != (run[i] >= 0.5)]
        log.append({"iter": it, "agreement": round(float(agree.mean()), 3), "flips": len(flips)})
        if not flips:
            break
        i = flips[0]
        # Is the take at the flip a two-angle take? Its ends would disagree.
        f2 = per_frame_features(proxy_path, [shots[i]], first_frame, fractions=(0.12, 0.88))[0]
        p0, p1 = model.proba(f2[0])[0], model.proba(f2[1])[0]
        sh = shots[i]
        if (p0 - 0.5) * (p1 - 0.5) < 0 and sh.duration > 4.0:
            rows = list(sample(proxy_path, sh.start - first_frame / 25.0, sh.end - first_frame / 25.0))
            inner = rows[2:-1] if len(rows) > 4 else rows
            t, d = max(inner, key=lambda r: r[1])[:2]
            t_src = t + first_frame / 25.0
            shots[i:i + 1] = [Shot(sh.start, t_src - sh.start, sh.tilt, sh.samples // 2, sh.cut_diff),
                              Shot(t_src, sh.end - t_src, sh.tilt, sh.samples // 2, float(d))]
            log[-1]["action"] = f"split take {i} at {t_src:.1f}s (jump {d:.1f})"
        else:
            # merge the weaker of the two boundaries around the flip
            j = i if (i + 1 >= len(shots) or (shots[i].cut_diff or 0) <= (shots[i + 1].cut_diff or 0)) else i + 1
            j = max(1, j)
            a, b = shots[j - 1], shots[j]
            shots[j - 1:j + 1] = [Shot(a.start, b.end - a.start, a.tilt, a.samples + b.samples, a.cut_diff)]
            log[-1]["action"] = f"merged boundary before take {j} (jump {b.cut_diff})"
    return shots, log
