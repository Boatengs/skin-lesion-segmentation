import numpy as np
import pandas as pd
import torch
from conftest import synthetic

from skinseg.baselines import otsu_masks, prior_mask
from skinseg.metrics import cluster_bootstrap, dice, iou
from skinseg.model import UNet, n_params
from skinseg.splits import image_split, lesion_split, seen_lesion
from skinseg.train import augment, loss_fn


def test_v1_layout_has_v1_parameter_count():
    assert n_params(UNet(64)) == 31_037_633


def test_unet_output_shape():
    assert UNet(4)(torch.zeros(2, 3, 64, 64)).shape == (2, 1, 64, 64)


def test_dice_and_iou():
    a = np.zeros((1, 4, 4), bool)
    b = a.copy()
    a[0, :2] = True
    b[0, :1] = True
    assert dice(a, b)[0] == 2 * 4 / (8 + 4)
    assert iou(a, b)[0] == 4 / 8
    assert dice(np.zeros((1, 2, 2), bool), np.zeros((1, 2, 2), bool))[0] == 1


def test_image_split_matches_version_1_shape():
    meta = pd.DataFrame({"image_id": [f"i{k}" for k in range(10015)], "lesion_id": 0, "dx": "nv"})
    s = image_split(meta)
    assert (len(s.train), len(s.val), len(s.test)) == (8012, 1001, 1002)


def test_lesion_split_keeps_lesions_apart():
    meta, _, _ = synthetic()
    s = lesion_split(meta)
    assert not seen_lesion(meta, s.train, s.test).any()
    assert not seen_lesion(meta, s.train, s.val).any()
    assert len(s.train) + len(s.val) + len(s.test) == len(meta)


def test_augment_moves_image_and_mask_together():
    x = torch.rand(8, 3, 16, 16)
    y = (x[:, :1] > 0.5).float()
    _, ya = augment(x, y, torch.Generator().manual_seed(0))
    # brightness jitter changes values, but the mask must follow every flip and rotation
    for i in range(8):
        same = [torch.equal(ya[i], t) for t in [y[i], y[i].flip(-1), y[i].flip(-2), y[i].flip(-1).flip(-2)]] + \
               [torch.equal(ya[i], y[i].rot90(k, (-2, -1)).flip(f)) for k in (1, 2, 3) for f in (-1, -2)] + \
               [torch.equal(ya[i], y[i].flip(f).rot90(k, (-2, -1))) for k in (1, 2, 3) for f in (-1, -2)] + \
               [torch.equal(ya[i], y[i].flip(-1).flip(-2).rot90(k, (-2, -1))) for k in (1, 2, 3)] + \
               [torch.equal(ya[i], y[i].rot90(k, (-2, -1))) for k in (1, 2, 3)]
        assert any(same)


def test_loss_is_lower_for_better_prediction():
    y = torch.zeros(1, 1, 8, 8)
    y[..., :4] = 1
    good, bad = (y * 20 - 10), (-y * 20 + 10)
    assert loss_fn(good, y) < loss_fn(bad, y)


def test_baselines_find_dark_centred_lesions():
    _, imgs, masks = synthetic(30)
    assert dice(otsu_masks(imgs), masks).mean() > 0.8
    prior, _ = prior_mask(masks[:25], masks[25:])
    assert prior.any()


def test_bootstrap_interval_contains_mean():
    v = np.random.default_rng(0).random(200)
    lo, hi = cluster_bootstrap(lambda r: v[r].mean(), np.arange(200) // 2, n_boot=200)
    assert lo < v.mean() < hi
