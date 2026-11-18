import numpy as np
import pandas as pd
import pytest

from skinseg import CLASSES, SIZE


def synthetic(n_lesions: int = 140, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, imgs, masks = [], [], []
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for k in range(n_lesions):
        dx = CLASSES[k % len(CLASSES)]
        cy, cx = SIZE / 2 + rng.normal(0, 8, 2)
        ry, rx = rng.uniform(15, 40, 2)
        for _ in range(int(rng.integers(1, 3))):
            m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            img = np.full((SIZE, SIZE, 3), [200, 160, 140], float) + rng.normal(0, 8, (SIZE, SIZE, 3))
            img[m] -= 90
            rows.append({"lesion_id": f"L{k:04d}", "image_id": f"ISIC_{len(rows):07d}", "dx": dx})
            imgs.append(img.clip(0, 255).astype(np.uint8))
            masks.append(m)
    return pd.DataFrame(rows), np.stack(imgs), np.stack(masks)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    meta, imgs, masks = synthetic()
    meta.to_csv(d / "metadata.csv", index=False)
    np.save(d / f"images{SIZE}.npy", imgs)
    np.save(d / f"masks{SIZE}.npy", masks)
    return d
