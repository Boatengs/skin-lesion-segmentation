"""Two methods with no learned weights, to show what the U-Net adds.

  prior   one fixed mask: the share of training images where each pixel is lesion,
          cut at the threshold that maximises mean Dice on validation. Most
          HAM10000 photos are centred on the lesion, so this is a real baseline.
  otsu    classic image processing: Otsu threshold on a grey version of the image
          (lesions are darker than skin), keep the connected region nearest the
          centre and fill its holes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .metrics import dice


def prior_mask(train_masks: np.ndarray, val_masks: np.ndarray) -> tuple[np.ndarray, float]:
    freq = np.asarray(train_masks).mean(0)
    best = max(np.linspace(0.05, 0.95, 19), key=lambda t: dice(np.broadcast_to(freq > t, val_masks.shape), val_masks).mean())
    return freq > best, float(best)


def _otsu(v: np.ndarray) -> float:
    hist, edges = np.histogram(v, bins=256, range=(0, 256))
    w = np.cumsum(hist)
    m = np.cumsum(hist * edges[:-1])
    total, mt = w[-1], m[-1]
    w1 = total - w
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w / total - m) ** 2 * total / (w * w1)
    return float(edges[np.nanargmax(np.nan_to_num(between, nan=-1))])


def otsu_mask(img: np.ndarray) -> np.ndarray:
    grey = img.astype(float) @ np.array([0.299, 0.587, 0.114])
    grey = ndimage.gaussian_filter(grey, 1.5)
    dark = grey <= _otsu(grey)
    lab, n = ndimage.label(dark)
    if n == 0:
        return dark
    h, w = dark.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    best, score = 0, np.inf
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < 0.005 * h * w:
            continue
        # prefer large regions near the centre; dark borders and corners lose
        d = np.hypot(ys.mean() - cy, xs.mean() - cx) / np.hypot(cy, cx) - 0.5 * len(ys) / (h * w)
        touches = ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
        d += 0.5 * touches
        if d < score:
            best, score = k, d
    return ndimage.binary_fill_holes(lab == best) if best else dark


def otsu_masks(imgs: np.ndarray) -> np.ndarray:
    return np.stack([otsu_mask(np.asarray(i)) for i in imgs])
