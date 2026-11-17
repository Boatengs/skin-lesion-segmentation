"""Per image overlap metrics and lesion level bootstrap intervals."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def dice(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Dice per image for boolean arrays shaped (n, h, w). Two empty masks score 1."""
    p, t = pred.reshape(len(pred), -1), truth.reshape(len(truth), -1)
    inter = (p & t).sum(1)
    denom = p.sum(1) + t.sum(1)
    return np.where(denom == 0, 1.0, 2 * inter / np.maximum(denom, 1))


def iou(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    p, t = pred.reshape(len(pred), -1), truth.reshape(len(truth), -1)
    inter = (p & t).sum(1)
    union = (p | t).sum(1)
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def cluster_bootstrap(stat: Callable[[np.ndarray], float], groups: np.ndarray, n_boot: int = 1000,
                      seed: int = 0) -> tuple[float, float]:
    """95% percentile interval resampling whole lesions. `stat` takes row indices."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = [stat(np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])) for _ in range(n_boot)]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)
