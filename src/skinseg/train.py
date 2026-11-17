"""Train one U-Net on one split and save the best checkpoint and test predictions.

    python -m skinseg.train --split lesion --seed 42 --out runs/lesion-s42

Recipe kept from version 1: same U-Net layout, BCE plus Dice loss (50/50), Adam
(lr 1e-3, weight decay 1e-5), ReduceLROnPlateau (patience 3, factor 0.5), batch
16, 20 epochs, flips, 90 degree rotations and brightness/contrast jitter, and the
checkpoint with the lowest validation loss. Changed for CPU runners: 32 base
channels instead of 64 (7.8 million parameters instead of 31.0 million) and
128 by 128 inputs instead of 256 by 256.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import SIZE
from .data import build_cache
from .metrics import dice
from .model import UNet, n_params
from .splits import SPLITTERS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BASE, EPOCHS, BATCH = 32, 20, 16


def to_tensor(imgs: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float() / 255.0


def augment(x: torch.Tensor, y: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Per sample: horizontal and vertical flips (p 0.5 each), a random multiple of
    90 degrees (p 0.5), brightness and contrast jitter of up to 20% (p 0.3)."""
    out_x, out_y = [], []
    for xi, yi in zip(x, y, strict=True):
        r = torch.rand(5, generator=g)
        if r[0] < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        if r[1] < 0.5:
            xi, yi = xi.flip(-2), yi.flip(-2)
        if r[2] < 0.5:
            k = int(torch.randint(1, 4, (1,), generator=g))
            xi, yi = xi.rot90(k, (-2, -1)), yi.rot90(k, (-2, -1))
        if r[3] < 0.3:
            b, c = 0.8 + 0.4 * torch.rand(2, generator=g)
            xi = ((xi * b - xi.mean()) * c + xi.mean()).clamp(0, 1)
        out_x.append(xi)
        out_y.append(yi)
    return torch.stack(out_x), torch.stack(out_y)


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    d = 1 - (2 * inter + 1e-6) / (p.sum((2, 3)) + target.sum((2, 3)) + 1e-6)
    return 0.5 * bce + 0.5 * d.mean()


@torch.no_grad()
def predict(model: nn.Module, imgs: np.ndarray, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        x = (to_tensor(imgs[i:i + batch]) - MEAN) / STD
        out.append(torch.sigmoid(model(x))[:, 0].numpy())
    return np.concatenate(out)


def run(split: str, seed: int, data: Path, out: Path, epochs: int = EPOCHS, base: int = BASE, limit: int | None = None) -> dict:
    torch.set_num_threads(os.cpu_count() or 1)
    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data / "metadata.csv")
    imgs, masks = build_cache(meta, data)
    s = SPLITTERS[split](meta, seed)
    train, val, test = s.train, s.val, s.test
    if limit:
        train, val, test = train[:limit], val[: limit // 4], test[: limit // 4]
    xtr, ytr = np.asarray(imgs[train]), np.asarray(masks[train])
    xva, yva = np.asarray(imgs[val]), np.asarray(masks[val])

    model = UNet(base)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    g = torch.Generator().manual_seed(seed)
    history, best = [], float("inf")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order = torch.randperm(len(train), generator=g).numpy()
        tot = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            x, y = augment(to_tensor(xtr[b]), torch.from_numpy(ytr[b]).float()[:, None], g)
            opt.zero_grad()
            loss = loss_fn(model((x - MEAN) / STD), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        pv = predict(model, xva)
        with torch.no_grad():
            vloss = float(loss_fn(torch.logit(torch.from_numpy(pv).clamp(1e-6, 1 - 1e-6))[:, None],
                                  torch.from_numpy(yva).float()[:, None]))
        vdice = float(dice(pv > 0.5, yva).mean())
        sched.step(vloss)
        saved = vloss < best
        if saved:
            best = vloss
            torch.save(model.state_dict(), out / "model.pth")
        history.append({"epoch": epoch + 1, "train_loss": tot / len(order), "val_loss": vloss, "val_dice": vdice,
                        "lr": opt.param_groups[0]["lr"], "saved": saved, "seconds": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)

    model.load_state_dict(torch.load(out / "model.pth"))
    probs = predict(model, np.asarray(imgs[test]))
    np.savez_compressed(out / "test_probs.npz", probs=(probs * 255).round().astype(np.uint8), index=test)
    np.savez(out / "split.npz", train=train, val=val, test=test)
    info = {"split": split, "seed": seed, "base_channels": base, "params": n_params(model), "size": SIZE,
            "epochs": epochs, "best_val_loss": best, "test_dice_at_0.5": float(dice(probs > 0.5, np.asarray(masks[test])).mean()),
            "history": history}
    (out / "run.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITTERS), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    run(a.split, a.seed, a.data, a.out, a.epochs)


if __name__ == "__main__":
    main()
