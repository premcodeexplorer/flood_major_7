"""
Phase 1 - Step 1: Convert VisDrone2019-DET annotations to YOLO format.

VisDrone annotation line:
    bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion

VisDrone categories:
    0 ignored-regions  1 pedestrian  2 people   3 bicycle  4 car   5 van
    6 truck            7 tricycle    8 awning-tricycle     9 bus  10 motor
    11 others

Project class scheme (kept IDENTICAL for the Phase-2 flood fine-tune so the
YOLO detection head is never reinitialized):
    0 person   <- pedestrian, people
    1 vehicle  <- bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
    2 animal   <- (none in VisDrone; appears only in the flood dataset)

No images are copied or moved. Labels are written to <split>/labels/ next to
<split>/images/, which is exactly where Ultralytics looks for them.

Usage:
    python visdrone_to_yolo.py --root /content/datasets/VisDrone
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

VISDRONE_TO_OURS = {
    1: 0, 2: 0,                                  # person
    3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1,  # vehicle
}
CLASS_NAMES = ["person", "vehicle", "animal"]
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}


def convert_split(split_dir: Path, min_side_px: float = 2.0) -> dict:
    img_dir, ann_dir = split_dir / "images", split_dir / "annotations"
    lbl_dir = split_dir / "labels"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"expected images/ and annotations/ inside {split_dir}")
    lbl_dir.mkdir(exist_ok=True)

    stats = {
        "images": 0, "images_with_no_objects": 0, "boxes_kept": 0,
        "skipped_ignored_region": 0, "skipped_other_class": 0,
        "skipped_degenerate": 0, "per_class": Counter(),
        # COCO definition of a "small" object: area < 32*32 px
        "small_boxes_coco": 0, "medium_boxes_coco": 0, "large_boxes_coco": 0,
    }

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        ann_path = ann_dir / f"{img_path.stem}.txt"
        stats["images"] += 1
        with Image.open(img_path) as im:
            W, H = im.size

        lines = []
        if ann_path.exists():
            for raw in ann_path.read_text().splitlines():
                raw = raw.strip().rstrip(",")
                if not raw:
                    continue
                p = raw.split(",")
                if len(p) < 6:
                    continue
                x, y, w, h = (float(v) for v in p[:4])
                score, cat = int(float(p[4])), int(float(p[5]))

                if score == 0 or cat == 0:          # ignored region
                    stats["skipped_ignored_region"] += 1
                    continue
                if cat not in VISDRONE_TO_OURS:     # category 11 "others"
                    stats["skipped_other_class"] += 1
                    continue

                # clamp to image bounds, then drop slivers
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(W), x + w), min(float(H), y + h)
                bw, bh = x2 - x1, y2 - y1
                if bw < min_side_px or bh < min_side_px:
                    stats["skipped_degenerate"] += 1
                    continue

                cls = VISDRONE_TO_OURS[cat]
                lines.append(
                    f"{cls} {(x1 + bw / 2) / W:.6f} {(y1 + bh / 2) / H:.6f} "
                    f"{bw / W:.6f} {bh / H:.6f}"
                )
                stats["boxes_kept"] += 1
                stats["per_class"][CLASS_NAMES[cls]] += 1
                area = bw * bh
                if area < 32 ** 2:
                    stats["small_boxes_coco"] += 1
                elif area < 96 ** 2:
                    stats["medium_boxes_coco"] += 1
                else:
                    stats["large_boxes_coco"] += 1

        (lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        if not lines:
            stats["images_with_no_objects"] += 1

    stats["per_class"] = dict(stats["per_class"])
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing VisDrone2019-DET-train / -val")
    ap.add_argument("--yaml-out", default=None,
                    help="path for the generated data.yaml (default <root>/visdrone.yaml)")
    ap.add_argument("--stats-out", default=None,
                    help="path for the conversion stats JSON")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    all_stats = {}
    for split, folder in SPLITS.items():
        print(f"[convert] {split}: {folder}")
        all_stats[split] = convert_split(root / folder)
        s = all_stats[split]
        print(f"  images={s['images']}  boxes={s['boxes_kept']}  "
              f"per_class={s['per_class']}  "
              f"small(COCO)={s['small_boxes_coco']} "
              f"({100 * s['small_boxes_coco'] / max(1, s['boxes_kept']):.1f}%)  "
              f"empty_images={s['images_with_no_objects']}")

    yaml_path = Path(args.yaml_out) if args.yaml_out else root / "visdrone.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "# Auto-generated by visdrone_to_yolo.py - Phase 1 base detector\n"
        f"path: {root.as_posix()}\n"
        f"train: {SPLITS['train']}/images\n"
        f"val: {SPLITS['val']}/images\n"
        "nc: 3\n"
        "names:\n"
        "  0: person\n"
        "  1: vehicle\n"
        "  2: animal   # unused in VisDrone; reserved for the Phase-2 flood fine-tune\n"
    )
    print(f"[convert] wrote {yaml_path}")

    stats_path = Path(args.stats_out) if args.stats_out else root / "visdrone_conversion_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"[convert] wrote {stats_path}")


if __name__ == "__main__":
    main()
