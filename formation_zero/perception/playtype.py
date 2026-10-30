"""Play type read from film: scrimmage, kickoff, punt, or place kick (field goal, extra point).

Weak on its own: from 32x18 colour layouts and edge-orientation histograms it misses most place
kicks and calls some scrimmage plays punts. It does not need to be strong. The take aligner cannot
tell apart alignments that differ only in which optional plays the film kept (a false start on and
a kneel-down off, or the reverse), because both explain every take equally well. Between those, the
right one is the alignment under which play types are most predictable from the film, measured on
held-out plays: a stretch shifted by one play scrambles which takes are kicks, and the score drops.
On the Rams-Bears film that margin was 75 nats, for the choice a person confirmed by eye.
"""

from __future__ import annotations

import numpy as np

TYPE_CLASSES = ("scrimmage", "kickoff", "punt", "place_kick")
_CLASS = {"kickoff": 1, "punt": 2, "field_goal": 3, "extra_point": 3}


def type_class(play_type) -> int:
    """Index into TYPE_CLASSES for a play-by-play play_type; anything not a kick is scrimmage."""
    return _CLASS.get(play_type, 0)


class SoftmaxModel:
    """Multinomial logistic regression with balanced class weights, by gradient descent."""

    def __init__(self, W: np.ndarray, b: np.ndarray, mu: np.ndarray, sd: np.ndarray):
        self.W, self.b, self.mu, self.sd = W, b, mu, sd

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, k: int = len(TYPE_CLASSES), *, l2: float = 1.0,
            steps: int = 800, lr: float = 0.2) -> SoftmaxModel:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        W, b = np.zeros((Z.shape[1], k)), np.zeros(k)
        Y = np.eye(k)[y]
        n = len(y)
        # Kicks are a tenth of the plays; without balancing the model just says "scrimmage".
        w = (n / (k * np.maximum(np.bincount(y, minlength=k), 1)))[y][:, None]
        for _ in range(steps):
            G = (np.exp(cls._logsoftmax(Z @ W + b)) - Y) * w
            W -= lr * (Z.T @ G / n + l2 * W / n)
            b -= lr * G.mean(0)
        return cls(W, b, mu, sd)

    @staticmethod
    def _logsoftmax(L: np.ndarray) -> np.ndarray:
        L = L - L.max(axis=-1, keepdims=True)
        return L - np.log(np.exp(L).sum(axis=-1, keepdims=True))

    def logp(self, X: np.ndarray) -> np.ndarray:
        return self._logsoftmax(((X - self.mu) / self.sd) @ self.W + self.b)


def heldout_loglik(features: np.ndarray, plays: list[tuple[int, np.ndarray]], folds: int = 5) -> float:
    """Sum over plays of held-out log P(official type | the play's samples).

    `plays` holds (type class, sample indices of the play's takes). Each fold trains on the others'
    samples; a play's reading is the mean per-sample log-probability, renormalised.
    """
    total = 0.0
    for f in range(folds):
        train = [q for i, q in enumerate(plays) if i % folds != f]
        test = [q for i, q in enumerate(plays) if i % folds == f]
        if not train or not test:
            continue
        model = SoftmaxModel.fit(np.concatenate([features[q[1]] for q in train]),
                                 np.concatenate([np.full(len(q[1]), q[0]) for q in train]))
        for t, idx in test:
            m = model.logp(features[idx]).mean(0)
            total += float(m[t] - np.log(np.exp(m).sum()))
    return total


__all__ = ["TYPE_CLASSES", "SoftmaxModel", "heldout_loglik", "type_class"]
