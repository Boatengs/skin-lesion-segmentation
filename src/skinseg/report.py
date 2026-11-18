"""Score every run and write results/summary.json, results/RESULTS.md and figures.

    python -m skinseg.report --runs runs --data data --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES
from .baselines import otsu_masks, prior_mask
from .data import build_cache
from .metrics import cluster_bootstrap, dice, iou
from .splits import SPLITTERS, seen_lesion

FAIL = 0.5  # an image counts as a failure when its Dice is below this
V1 = {"test_dice": 0.9115, "best_val_dice": 0.9191, "params": 31037633, "size": 256}


def score(pred: np.ndarray, truth: np.ndarray, groups: np.ndarray) -> dict:
    d, j = dice(pred, truth), iou(pred, truth)
    return {"n": len(d), "dice": float(d.mean()), "dice_median": float(np.median(d)), "iou": float(j.mean()),
            "fail_rate": float((d < FAIL).mean()),
            "dice_ci95": cluster_bootstrap(lambda r: d[r].mean(), groups),
            "fail_ci95": cluster_bootstrap(lambda r: (d[r] < FAIL).mean(), groups)}


def load_runs(runs: Path) -> list[dict]:
    out = []
    for d in sorted(runs.glob("*-s*")):
        if not (d / "run.json").exists():
            continue
        info = json.loads((d / "run.json").read_text())
        z = np.load(d / "test_probs.npz")
        info["probs"], info["index"] = z["probs"], z["index"]
        info["dir"] = d
        out.append(info)
    return out


def analyse(runs: list[dict], meta: pd.DataFrame, imgs: np.ndarray, masks: np.ndarray) -> dict:
    groups = meta["lesion_id"].to_numpy()
    dx = meta["dx"].to_numpy()
    res = {"v1": V1, "runs": [], "by_diagnosis": {}, "baselines": {}, "leak": {}}
    per_image = []
    for r in runs:
        idx = r["index"]
        pred, truth = r["probs"] >= 128, np.asarray(masks[idx])
        s = score(pred, truth, groups[idx])
        split = SPLITTERS[r["split"]](meta, r["seed"])
        seen = seen_lesion(meta, split.train, idx)
        s.update({"split": r["split"], "seed": r["seed"], "params": r["params"], "best_val_loss": r["best_val_loss"],
                  "best_epoch": max(h["epoch"] for h in r["history"] if h["saved"]),
                  "test_images_with_seen_lesion": int(seen.sum())})
        res["runs"].append(s)
        d = dice(pred, truth)
        per_image.append(pd.DataFrame({"split": r["split"], "seed": r["seed"], "image_id": meta.image_id.to_numpy()[idx],
                                       "lesion_id": groups[idx], "dx": dx[idx], "seen_lesion": seen, "dice": d,
                                       "iou": iou(pred, truth), "lesion_share": truth.mean((1, 2))}))
    pim = pd.concat(per_image, ignore_index=True)

    for split in SPLITTERS:
        x = pim[pim.split == split]
        res["by_diagnosis"][split] = {c: {"images": int((x.dx == c).sum() / max(x.seed.nunique(), 1)),
                                          "dice": float(x.loc[x.dx == c, "dice"].mean()),
                                          "fail_rate": float((x.loc[x.dx == c, "dice"] < FAIL).mean())}
                                      for c in CLASSES if (x.dx == c).any()}

    # baselines on each split's seed 42 test set
    for split in SPLITTERS:
        s = SPLITTERS[split](meta, 42)
        truth = np.asarray(masks[s.test])
        prior, thr = prior_mask(np.asarray(masks[s.train]), np.asarray(masks[s.val]))
        res["baselines"][split] = {
            "prior": score(np.broadcast_to(prior, truth.shape), truth, groups[s.test]) | {"threshold": thr},
            "otsu": score(otsu_masks(np.asarray(imgs[s.test])), truth, groups[s.test]),
        }

    # how alike are two photos of the same lesion? (image split, seed 42)
    s = SPLITTERS["image"](meta, 42)
    train_by_lesion = pd.Series(s.train).groupby(groups[s.train]).apply(list).to_dict()
    copy = []
    for i in s.test:
        if groups[i] in train_by_lesion:
            t = np.asarray(masks[i])[None]
            copy.append(max(float(dice(np.asarray(masks[j])[None], t)[0]) for j in train_by_lesion[groups[i]]))
    x = pim[(pim.split == "image") & (pim.seed == 42)]
    res["leak"] = {
        "test_images": len(s.test), "with_seen_lesion": len(copy),
        "copy_training_mask_dice_mean": float(np.mean(copy)), "copy_training_mask_dice_median": float(np.median(copy)),
        "model_dice_seen": float(x[x.seen_lesion].dice.mean()), "model_dice_unseen": float(x[~x.seen_lesion].dice.mean()),
        "model_dice_seen_nv": float(x[x.seen_lesion & (x.dx == "nv")].dice.mean()),
        "model_dice_unseen_nv": float(x[~x.seen_lesion & (x.dx == "nv")].dice.mean()),
        "n_seen_nv": int((x.seen_lesion & (x.dx == "nv")).sum()), "n_unseen_nv": int((~x.seen_lesion & (x.dx == "nv")).sum()),
    }
    by = pd.DataFrame(res["runs"]).pivot(index="seed", columns="split", values="dice")
    res["image_minus_lesion_dice_by_seed"] = {int(k): float(v) for k, v in (by["image"] - by["lesion"]).items()} \
        if {"image", "lesion"} <= set(by.columns) else {}
    return res, pim


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def markdown(res: dict) -> str:
    runs = pd.DataFrame(res["runs"]).sort_values(["split", "seed"])
    L = ["# Results", "",
         f"U-Net, {runs.params.iloc[0]:,} parameters, 128 by 128 inputs, 20 epochs on CPU. Version 1 used the same layout with "
         f"{V1['params']:,} parameters at 256 by 256 on a GPU and reported test Dice {V1['test_dice']}.", "",
         "Dice is the overlap between the predicted and the true lesion outline: 2 x shared pixels / (predicted + true "
         f"pixels), from 0 (no overlap) to 1 (identical). A failure is an image with Dice below {FAIL}. Intervals are 95% and "
         "resample whole lesions.", "",
         "## Every run", "",
         "| Split | Seed | Test images | Seen lesion in training | Mean Dice (95% interval) | Median Dice | Mean IoU | Failures (95% interval) | Best epoch |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in runs.iterrows():
        L.append(f"| {r.split} | {r.seed} | {r.n} | {r.test_images_with_seen_lesion} | {r.dice:.3f} ({r.dice_ci95[0]:.3f} to "
                 f"{r.dice_ci95[1]:.3f}) | {r.dice_median:.3f} | {r.iou:.3f} | {pct(r.fail_rate)} ({pct(r.fail_ci95[0])} to "
                 f"{pct(r.fail_ci95[1])}) | {r.best_epoch} |")
    if res["image_minus_lesion_dice_by_seed"]:
        diffs = res["image_minus_lesion_dice_by_seed"]
        L += ["", "Image split minus lesion split, mean Dice, by seed: " +
              ", ".join(f"seed {k}: {v:+.3f}" for k, v in diffs.items()) + "."]
    lk = res["leak"]
    L += ["", "## How alike are two photos of one lesion", "",
          f"In the version 1 split (image split, seed 42), {lk['with_seen_lesion']} of {lk['test_images']} test images show a "
          "lesion that also has a photo in training. Copying that training photo's mask, with no model at all, scores mean "
          f"Dice {lk['copy_training_mask_dice_mean']:.3f} (median {lk['copy_training_mask_dice_median']:.3f}).", "",
          "| Test images (image split, seed 42) | Images | Model mean Dice |", "|---|---|---|",
          f"| Lesion seen in training | {lk['with_seen_lesion']} | {lk['model_dice_seen']:.3f} |",
          f"| Lesion not seen | {lk['test_images'] - lk['with_seen_lesion']} | {lk['model_dice_unseen']:.3f} |",
          f"| Nevi only, seen | {lk['n_seen_nv']} | {lk['model_dice_seen_nv']:.3f} |",
          f"| Nevi only, not seen | {lk['n_unseen_nv']} | {lk['model_dice_unseen_nv']:.3f} |", "",
          "The seen and unseen groups differ in diagnosis mix, so the nevus only rows are the fairer comparison.", "",
          "## Methods with no learning (seed 42 test sets)", "",
          "| Split | Method | Mean Dice (95% interval) | Failures |", "|---|---|---|---|"]
    for split, b in res["baselines"].items():
        for name, s in b.items():
            L.append(f"| {split} | {name} | {s['dice']:.3f} ({s['dice_ci95'][0]:.3f} to {s['dice_ci95'][1]:.3f}) | {pct(s['fail_rate'])} |")
    L += ["", "prior: one fixed mask from how often each pixel is lesion in training. otsu: grey level threshold, region "
          "nearest the centre, holes filled. See `baselines.py`.", "",
          "## By diagnosis (all seeds pooled)", "", "| Diagnosis | Split | Test images per seed | Mean Dice | Failures |",
          "|---|---|---|---|---|"]
    for c in CLASSES:
        for split, d in res["by_diagnosis"].items():
            if c in d:
                L.append(f"| {c} | {split} | {d[c]['images']} | {d[c]['dice']:.3f} | {pct(d[c]['fail_rate'])} |")
    return "\n".join(L) + "\n"


def figures(runs: list[dict], pim: pd.DataFrame, meta: pd.DataFrame, imgs: np.ndarray, masks: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = next((x for x in runs if x["split"] == "lesion" and x["seed"] == 42), runs[0])
    x = pim[(pim.split == r["split"]) & (pim.seed == r["seed"])].reset_index(drop=True)
    order = x.dice.argsort().to_numpy()
    picks = [order[-1], order[len(order) // 2], order[len(order) // 10], order[0]]
    labels = ["best", "median", "10th percentile", "worst"]
    fig, ax = plt.subplots(len(picks), 3, figsize=(7.5, 10))
    for row, (k, lab) in enumerate(zip(picks, labels, strict=True)):
        i = r["index"][k]
        img, truth, pred = np.asarray(imgs[i]), np.asarray(masks[i]), r["probs"][k] >= 128
        ax[row, 0].imshow(img)
        ax[row, 0].set_title(f"{lab}: {meta.dx[i]}", fontsize=9)
        ax[row, 1].imshow(truth, cmap="gray")
        ax[row, 1].set_title("true outline", fontsize=9)
        ax[row, 2].imshow(pred, cmap="gray")
        ax[row, 2].set_title(f"predicted, Dice {x.dice[k]:.2f}", fontsize=9)
        for a in ax[row]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=110)

    fig, a = plt.subplots(figsize=(7, 3.5))
    for split, color in (("image", "#94a3b8"), ("lesion", "#2563eb")):
        a.hist(pim[pim.split == split].dice, bins=np.linspace(0, 1, 41), alpha=0.6, color=color, label=f"{split} split",
               density=True)
    a.axvline(FAIL, color="#dc2626", ls="--", lw=1)
    a.set_xlabel("Dice per test image (all seeds)")
    a.set_yticks([])
    a.legend()
    fig.tight_layout()
    fig.savefig(out / "dice_distribution.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(a.data / "metadata.csv")
    imgs, masks = build_cache(meta, a.data)
    runs = load_runs(a.runs)
    res, pim = analyse(runs, meta, imgs, masks)
    (a.out / "summary.json").write_text(json.dumps(res, indent=1))
    (a.out / "RESULTS.md").write_text(markdown(res))
    pim.to_csv(a.out / "per_image.csv", index=False, float_format="%.4f")
    for r in runs:
        (a.out / "runs").mkdir(exist_ok=True)
        hist = {k: v for k, v in json.loads((r["dir"] / "run.json").read_text()).items()}
        (a.out / "runs" / f"{r['split']}-s{r['seed']}.json").write_text(json.dumps(hist, indent=1))
    figures(runs, pim, meta, imgs, masks, a.out)
    print((a.out / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
