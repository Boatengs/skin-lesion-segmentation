"""Download HAM10000 images, metadata and lesion masks from Harvard Dataverse and
cache them at SIZE by SIZE.

Source: Tschandl, Rosendahl and Kittler, "The HAM10000 dataset", Scientific Data
5, 180161 (2018), Harvard Dataverse doi:10.7910/DVN/DBW86T, CC BY-NC 4.0. The
lesion masks are Tschandl et al.'s HAM10000_segmentations_lesion_tschandl set,
published in the same Dataverse record.
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES, SIZE

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/DBW86T"
HEADERS = {"User-Agent": "skinseg/2.0 (+https://github.com/Boatengs/skin-lesion-segmentation)"}


def _open(url: str, attempts: int = 4):
    for i in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=180)
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(10 * (i + 1))
    raise RuntimeError("unreachable")


def _files() -> dict[str, int]:
    with _open(f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}") as r:
        files = json.load(r)["data"]["latestVersion"]["files"]
    return {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files}


def _wanted(names) -> list[str]:
    seg = [n for n in names if "segmentation" in n.lower() and n.lower().endswith(".zip")]
    if not seg:
        raise RuntimeError(f"no segmentation archive in the Dataverse record; files: {sorted(names)}")
    return ["HAM10000_metadata.tab", "HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip", seg[0]]


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ids = _files()
    print("dataverse files:", sorted(ids), flush=True)
    for name in _wanted(ids):
        target = root / name.replace(".tab", ".csv")
        if target.exists():
            continue
        print(f"downloading {name}", flush=True)
        part = target.with_suffix(target.suffix + ".part")
        with _open(f"{DATAVERSE}/api/access/datafile/{ids[name]}?format=original") as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        part.rename(target)
    for sub, pattern, ext in (("images", "HAM10000_images_part_*.zip", ".jpg"), ("masks", "*segmentation*.zip", ".png")):
        out = root / sub
        if out.exists():
            continue
        out.mkdir()
        for z in sorted(root.glob(pattern)):
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if m.lower().endswith(ext) and not Path(m).name.startswith("."):
                        (out / Path(m).name).write_bytes(zf.read(m))
    return root


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    missing = {"lesion_id", "image_id", "dx"} - set(meta.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if set(meta["dx"]) - set(CLASSES):
        raise ValueError("unexpected diagnosis codes")
    if meta["image_id"].duplicated().any():
        raise ValueError("duplicate image ids")
    # version 1 listed the masks in sorted file order before splitting; keep that order
    return meta.sort_values("image_id").reset_index(drop=True)


def mask_name(image_id: str) -> str:
    return f"{image_id}_segmentation.png"


def build_cache(meta: pd.DataFrame, root: Path, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """uint8 images (n, size, size, 3) and bool masks (n, size, size)."""
    from PIL import Image

    img_path, mask_path = root / f"images{size}.npy", root / f"masks{size}.npy"
    if img_path.exists() and mask_path.exists():
        return np.load(img_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")
    imgs = np.zeros((len(meta), size, size, 3), np.uint8)
    masks = np.zeros((len(meta), size, size), bool)
    for i, iid in enumerate(meta["image_id"]):
        with Image.open(root / "images" / f"{iid}.jpg") as im:
            imgs[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
        with Image.open(root / "masks" / mask_name(iid)) as m:
            # area averaging, then majority: a pixel is lesion if most of it was lesion
            masks[i] = np.asarray(m.convert("L").resize((size, size), Image.BOX)) > 127
        if i % 2000 == 0:
            print(f"cached {i}/{len(meta)}", flush=True)
    np.save(img_path, imgs)
    np.save(mask_path, masks)
    return imgs, masks


def prepare(root: Path) -> None:
    download(root)
    meta = load_metadata(root / "HAM10000_metadata.csv")
    have = {p.name for p in (root / "masks").iterdir()}
    lacking = [i for i in meta["image_id"] if mask_name(i) not in have]
    if lacking:
        raise RuntimeError(f"{len(lacking)} images have no mask, e.g. {lacking[:3]}")
    meta.to_csv(root / "metadata.csv", index=False)
    build_cache(meta, root)
    print(f"{len(meta)} images, {meta.lesion_id.nunique()} lesions", flush=True)


if __name__ == "__main__":
    import sys

    prepare(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
