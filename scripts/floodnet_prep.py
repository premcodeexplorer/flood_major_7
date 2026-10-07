"""
Model 2 - Step 1: Verify FloodNet-Supervised v1.0 and cache it at training size.

Dataset: FloodNet-Supervised v1.0 (BinaLab, CDLA-Permissive 1.0),
Rahnemoonfar et al., IEEE Access 2021 - reference [4]. Use the OFFICIAL
train/val/test folders as given; never re-split, or results stop being
comparable with the paper's Table 4.

Images are ~4000x3000. Decoding that every epoch would starve a Colab GPU, so
each image is resized ONCE to --width x --height (default 1024x768, aspect kept)
and cached:
  * image: INTER_AREA (anti-aliased)        -> <out>/<split>/images/<id>.jpg
  * mask : INTER_NEAREST (class ids intact) -> <out>/<split>/masks/<id>.png
Training then takes random 512x512 crops of these; evaluation uses the whole
1024x768 frame. State that resolution in the paper.

Pairing is discovered, not hard-coded: masks are '<id>_lab.png', images
'<id>.jpg', and the split is the first path component starting with
train/val/test. Any unpaired file, unexpected class id, or image/mask size
mismatch is reported and the script exits non-zero.

Usage:
    python floodnet_prep.py --src /content/floodnet_raw --out /content/floodnet_1024
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["background", "building-flooded", "building-non-flooded", "road-flooded",
           "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def split_of(path: Path):
    for part in path.parts:
        low = part.lower()
        for s in SPLITS:
            if low.startswith(s):
                return s
    return None


def discover(src: Path):
    """-> {split: {id: {'img': path, 'mask': path}}}, plus a list of problems."""
    found = {s: {} for s in SPLITS}
    problems = []
    for root, _, files in os.walk(src):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() not in IMG_EXT or f.startswith("._"):
                continue
            s = split_of(p.relative_to(src))
            if s is None:
                problems.append(f"no split in path: {p}")
                continue
            if p.stem.endswith("_lab"):
                key, kind = p.stem[:-4], "mask"
            else:
                key, kind = p.stem, "img"
            entry = found[s].setdefault(key, {})
            if kind in entry:
                problems.append(f"duplicate {kind} for {s}/{key}: {entry[kind]} and {p}")
            entry[kind] = p
    for s in SPLITS:
        for key, e in found[s].items():
            if "img" not in e or "mask" not in e:
                problems.append(f"{s}/{key}: missing {'mask' if 'img' in e else 'image'}")
    return found, problems


def process(job):
    """Resize one pair. Returns (split, id, class pixel counts, problem or None, rotated)."""
    split, key, img_p, mask_p, out, W, H = job
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
    if img is None or mask is None:
        return split, key, None, f"unreadable: {img_p if img is None else mask_p}", False
    if mask.ndim == 3:   # some exports store the label in all 3 channels
        if not (np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 0], mask[..., 2])):
            return split, key, None, f"colour mask, not class ids: {mask_p}", False
        mask = mask[..., 0]
    if img.shape[:2] != mask.shape[:2]:
        return split, key, None, f"size mismatch {img.shape[:2]} vs {mask.shape[:2]}: {key}", False
    if mask.max() >= len(CLASSES):
        return split, key, None, f"class id {int(mask.max())} >= {len(CLASSES)}: {mask_p}", False
    rotated = img.shape[0] > img.shape[1]
    if rotated:   # keep every cached frame landscape, so no image is squashed
        img, mask = np.rot90(img).copy(), np.rot90(mask).copy()
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(Path(out) / split / "images" / f"{key}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(Path(out) / split / "masks" / f"{key}.png"), mask)
    counts = np.bincount(mask.ravel(), minlength=len(CLASSES))
    return split, key, counts, None, rotated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped FloodNet-Supervised v1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    # The official Dropbox zip also ships ColorMasks-FloodNetv1.0/ (RGB previews
    # of the same masks, no split in their path). Use only the supervised folder.
    sup = [d for d in src.iterdir() if d.is_dir() and d.name.lower().startswith("floodnet-supervised")]
    if len(sup) == 1:
        print(f"[discover] using {sup[0]} (ignoring {[d.name for d in src.iterdir() if d != sup[0]]})")
        src = sup[0]
    found, problems = discover(src)
    for s in SPLITS:
        print(f"[discover] {s:5s} {len(found[s])} pairs")
    if problems:
        print(f"[discover] {len(problems)} problem(s):")
        for p in problems[:20]:
            print("   ", p)
        sys.exit("fix the download before training (see problems above)")
    if not all(found[s] for s in SPLITS):
        sys.exit(f"a split is empty - is --src the unzipped FloodNet-Supervised_v1.0 folder?")

    for s in SPLITS:
        for sub in ("images", "masks"):
            (out / s / sub).mkdir(parents=True, exist_ok=True)

    jobs = [(s, k, e["img"], e["mask"], str(out), args.width, args.height)
            for s in SPLITS for k, e in sorted(found[s].items())]
    pix = {s: np.zeros(len(CLASSES), np.int64) for s in SPLITS}
    present = {s: np.zeros(len(CLASSES), np.int64) for s in SPLITS}
    rotated, bad = 0, []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (s, k, counts, prob, rot) in enumerate(ex.map(process, jobs, chunksize=4), 1):
            if prob:
                bad.append(prob)
                continue
            pix[s] += counts
            present[s] += counts > 0
            rotated += rot
            if i % 100 == 0 or i == len(jobs):
                print(f"  resized {i}/{len(jobs)}", end="\r")
    print()
    if bad:
        print(f"[resize] {len(bad)} problem(s):")
        for p in bad[:20]:
            print("   ", p)
        sys.exit("fix the download before training (see problems above)")

    stats = {"source": str(src), "size": [args.width, args.height],
             "portrait_images_rotated": rotated, "classes": CLASSES, "splits": {}}
    for s in SPLITS:
        tot = pix[s].sum()
        stats["splits"][s] = {
            "images": len(found[s]),
            "pixel_fraction": {c: round(float(pix[s][i] / tot), 5) for i, c in enumerate(CLASSES)},
            "images_containing": {c: int(present[s][i]) for i, c in enumerate(CLASSES)},
        }
    (out / "floodnet_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{'class':22s}" + "".join(f"{s:>16s}" for s in SPLITS))
    for i, c in enumerate(CLASSES):
        print(f"{c:22s}" + "".join(
            f"{100 * stats['splits'][s]['pixel_fraction'][c]:8.2f}% {stats['splits'][s]['images_containing'][c]:5d}im"
            for s in SPLITS))
    print(f"\n[done] {out}  (portrait images rotated: {rotated})")


if __name__ == "__main__":
    main()
