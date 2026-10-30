"""Split detected players into their two teams by uniform colour (architecture 2.3, the zero-model fallback).

SigLIP embeddings are the planned primary path; this one needs no model. Fit once per game on
player crops sampled across the whole game, then predict: the two ids stay fixed for that game.

    teams = ColourTeams().fit(crops)          # RGB uint8 crops of detected players
    teams.predict(crops)                      # 0/1 per crop, -1 if a crop is unusable

Features (`crop_features`): the box is resampled to a 32x16 grid and converted to Lab. Its middle
columns are cut into a shirt band and a pants band, and each band becomes a soft histogram over a
fixed palette of 85 colours, square-rooted. Shirt and pants stay apart because some teams swap them
(Rams white over blue, Bears navy over white). Pixels that look like the crop's own side columns
are down-weighted: those columns are mostly field, whatever colour the field is that day, so this
works on snow, where masking green does not.

Clustering: 2-means, then the split is refined as a two-class LDA (pooled covariance, reassign,
repeat), which learns which colours separate the teams and stops lighting changes across the game
from pulling the split their way. Team 0 is the side with the lighter shirts. `margin` gives the
signed LDA score, scaled so a typical crop sits at 1.

Measured (2026-10-30) on SAM 3 teacher player boxes (score > 0.5), fit on every crop of about 250
frames spread across each game: the snow game (2025 wk20 LA at CHI, 1080p, 4,521 crops) and a clear one
(2026 wk1 SF at LA, 576p, 5,584 crops). Counted by eye, leaving out boxes holding two opposing
players and blurs nobody could call:
- random crops, hand-labelled: snow first half 64 of 65, second half 72 of 73; clear 90 of 90;
- 48 random crops per predicted team: snow first half 79 of 82 (three Rams crouched or down on the
  white field, called Bears), snow second half 81 of 82, clear 84 of 85.
On the way: mean colour with green masked scored 75-81% on the snow game (92-95% clear); per-band
median Lab about 90%; the band histograms with plain 2-means 94% and 99% by half (100% clear),
but on 48 crops per game where versions disagreed, 2-means alone got 12-22 of 30 (snow) and 7-20
of 25 (clear), and the LDA step 27 of 30 and 25 of 25.

What still goes wrong: two players of opposite teams in one box; motion blur from camera pans;
referees SAM 3 called players, and sideline people. Blurs, referees and staff land on whichever
side looks greyer (the Bears in the snow game, the 49ers in the clear one), often with confident
margins. True errors mostly sit under UNSURE (6 of the 9 seen), which also catches about a quarter
of the mixed pairs and blurs, and 1-3% of good crops (4-5% of all crops fall under it).

Crops under 24 px are mostly fragments (a helmet, a pair of legs, a number); on them the lighter
side was right only about half the time, so they are -1. Fit on a few hundred crops agrees with
fitting on all of them on 98% of crops (1,000: 99%); fit on one half, the other half agrees on
98-99%. About 0.65 ms a crop on one CPU thread.
"""

from __future__ import annotations

import numpy as np

GRID = (32, 16)                          # rows, columns every crop is resampled to
COLUMNS = (0.2, 0.8)                     # the middle of the box, where the body is
BANDS = ((0.15, 0.45), (0.5, 0.8))       # shirt, pants: fractions of the box height
SIDE = 2                                 # grid columns at each edge taken as the crop's background
BACKGROUND_SIGMA = 12.0                  # Lab distance at which a pixel stops looking like background
SOFT = 10.0                              # Lab width of each palette colour's soft vote
MIN_HEIGHT = 24                          # px; shorter crops are unusable (-1): see the docstring
MIN_WIDTH = 6
RIDGE = 0.1                              # LDA covariance shrinkage, as a fraction of its mean variance
UNSURE = 0.25                            # |margin| below this is usually a mixed pair or a blur


def palette(levels=(12, 30, 50, 70, 90), chromas=(18, 42), hues: int = 8) -> np.ndarray:
    """Fixed Lab colours: a grey at each lightness, plus `hues` directions at each chroma."""
    angles = np.deg2rad(np.arange(hues) * 360 / hues)
    rows = [[L, 0.0, 0.0] for L in levels]
    rows += [[L, c * np.cos(t), c * np.sin(t)] for c in chromas for L in levels for t in angles]
    return np.array(rows, np.float32)


PALETTE = palette()


def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIE Lab, D65 white."""
    c = rgb.astype(np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def resample(crop: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Nearest-pixel resample to a fixed grid (up or down)."""
    h, w = crop.shape[:2]
    ys = ((np.arange(rows) + 0.5) * h / rows).astype(int)
    xs = ((np.arange(cols) + 0.5) * w / cols).astype(int)
    return crop[ys][:, xs]


def crop_features(crop: np.ndarray) -> np.ndarray | None:
    """Shirt and pants colour histograms of one RGB crop, background down-weighted; None if unusable."""
    if crop is None or crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] < MIN_HEIGHT or crop.shape[1] < MIN_WIDTH:
        return None
    g = lab(resample(crop, *GRID))
    rows, cols = GRID
    background = np.median(np.concatenate([g[:, :SIDE], g[:, -SIDE:]]).reshape(-1, 3), axis=0)
    x0, x1 = int(cols * COLUMNS[0]), int(cols * COLUMNS[1])
    out = []
    for top, bottom in BANDS:
        px = g[int(rows * top):int(rows * bottom), x0:x1].reshape(-1, 3)
        weight = 1 - np.exp(-((px - background) ** 2).sum(1) / (2 * BACKGROUND_SIGMA**2))
        if weight.sum() < 1e-3:  # the whole band is background colour: count it all
            weight = np.ones(len(px), np.float32)
        votes = np.exp(-((px[:, None] - PALETTE[None]) ** 2).sum(-1) / (2 * SOFT**2))
        votes /= votes.sum(1, keepdims=True) + 1e-12
        hist = weight @ votes
        out.append(np.sqrt(hist / hist.sum()))
    return np.concatenate(out).astype(np.float32)


def kmeans2(x: np.ndarray, seed: int = 0, restarts: int = 5, iters: int = 50) -> np.ndarray:
    """Labels 0/1 from the best of `restarts` runs of 2-means."""
    rng = np.random.default_rng(seed)
    xx = (x**2).sum(1)
    best, best_cost = None, np.inf
    for _ in range(restarts):
        centres = x[rng.choice(len(x), 2, replace=False)]
        for _ in range(iters):
            d = xx[:, None] - 2 * x @ centres.T + (centres**2).sum(1)
            labels = d.argmin(1)
            new = np.stack([x[labels == k].mean(0) if (labels == k).any() else centres[k] for k in (0, 1)])
            if np.allclose(new, centres):
                break
            centres = new
        cost = d.min(1).sum()
        if cost < best_cost:
            best, best_cost = labels, cost
    return best


def lda_split(x: np.ndarray, labels: np.ndarray, ridge: float = RIDGE, iters: int = 20):
    """Refine a 0/1 split: fit a pooled-covariance LDA, reassign by it, repeat.

    Returns (w, b) with team 1 where x @ w + b > 0, or (None, None) if `labels` has a near-empty side.
    """
    w = b = None
    if min((labels == 0).sum(), (labels == 1).sum()) < 2:
        return w, b
    for _ in range(iters):
        m0, m1 = x[labels == 0].mean(0), x[labels == 1].mean(0)
        r = x - np.where(labels[:, None] == 0, m0, m1)
        cov = r.T @ r / len(x)
        cov += (ridge * np.trace(cov) / len(cov) + 1e-12) * np.eye(len(cov))
        w_new = np.linalg.solve(cov, m1 - m0)
        b_new = -w_new @ (m0 + m1) / 2
        new = (x @ w_new + b_new > 0).astype(int)
        if min(new.sum(), len(new) - new.sum()) < 2:
            break  # this one would empty a side: keep the last
        w, b = w_new, b_new
        if (new == labels).all():
            break
        labels = new
    return w, b


class ColourTeams:
    """Two teams by uniform colour. Fit once per game; ids are fixed after that (0 = lighter shirts)."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.w = self.b = None

    def fit(self, crops: list[np.ndarray]) -> ColourTeams:
        feats = [f for f in map(crop_features, crops) if f is not None]
        if len(feats) < 4:
            raise ValueError(f"need at least 4 usable crops to split teams, got {len(feats)}")
        x = np.stack(feats).astype(np.float64)
        self.mean = x.mean(0)
        x -= self.mean
        w, b = lda_split(x, kmeans2(x, self.seed))
        if w is None:
            raise ValueError("crops did not split into two groups")
        score = x @ w + b
        # team 0 wears the lighter shirt: compare the mean shirt lightness the palette votes imply
        shirt_light = (x[:, :len(PALETTE)] + self.mean[:len(PALETTE)]) ** 2 @ PALETTE[:, 0]
        sign = -1.0 if shirt_light[score > 0].mean() > shirt_light[score <= 0].mean() else 1.0
        scale = max(float(np.median(np.abs(score))), 1e-12)  # margins in units of the typical crop's
        self.w, self.b = sign * w / scale, sign * b / scale
        return self

    def margin(self, crops: list[np.ndarray]) -> np.ndarray:
        """Signed score per crop: > 0 is team 1, about 1 for a typical crop, under UNSURE doubtful; NaN if unusable."""
        out = np.full(len(crops), np.nan)
        for i, crop in enumerate(crops):
            f = crop_features(crop)
            if f is not None:
                out[i] = (f - self.mean) @ self.w + self.b
        return out

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        m = self.margin(crops)
        return np.where(np.isnan(m), -1, m > 0).astype(int)
