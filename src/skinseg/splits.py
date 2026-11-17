"""Train, validation and test splits.

`image_split` reproduces version 1 exactly: the 10,015 image and mask pairs in
sorted file order, split 80/10/10 with scikit-learn's train_test_split and
random_state 42 (not stratified). HAM10000 often holds several photos of one
lesion, so that split puts photos of the same lesion on both sides.

`lesion_split` keeps every photo of a lesion in one partition, stratified by
diagnosis, so the test set holds only lesions the model has never seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def image_split(meta: pd.DataFrame, seed: int = 42) -> Split:
    idx = np.arange(len(meta))
    train, temp = train_test_split(idx, test_size=0.2, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return Split(np.sort(train), np.sort(val), np.sort(test))


def lesion_split(meta: pd.DataFrame, seed: int = 42) -> Split:
    idx = np.arange(len(meta))
    y = meta["dx"].to_numpy()
    groups = meta["lesion_id"].to_numpy()
    rest, test = next(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed).split(idx, y, groups))
    tr, va = next(StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=seed + 1).split(rest, y[rest], groups[rest]))
    return Split(np.sort(rest[tr]), np.sort(rest[va]), np.sort(test))


def seen_lesion(meta: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """For each test row, whether its lesion also has a photo in training."""
    seen = set(meta["lesion_id"].iloc[train])
    return meta["lesion_id"].iloc[test].isin(seen).to_numpy()


SPLITTERS = {"image": image_split, "lesion": lesion_split}
