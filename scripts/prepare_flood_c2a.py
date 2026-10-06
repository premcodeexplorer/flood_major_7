"""
Phase 2 - Step 1: Convert the Roboflow "Flood C2A" COCO export to YOLO format.

Dataset: Flood C2A v2 "Just Original" (Roboflow: wsar/flood-c2a-8vj8c, CC BY 4.0),
the flood subset of C2A (Nihal et al., ICPR 2024 - reference [3]).

What the export contains, and what this script does about it:
  * train: 2,079 aerial flood backgrounds with pasted-in people, but Roboflow
    stored 10 augmented copies of each (20,790 files). Ultralytics augments on
    the fly anyway, so only --copies-per-source of them are kept (default 2).
  * valid: 541 images whose backgrounds ALL also appear in train (different
    pasted people). Fine for picking the best epoch, too optimistic to report.
  * test: 109 REAL flood/landslide photos, disjoint from train. This is the
    split the Phase-1 vs Phase-2 comparison is reported on.
  * every category ("human", "0", "1") is a person -> class 0. Roboflow's
    parent category carries no boxes; "1" is 34 stray boxes in valid.
  * no vehicle labels at all. Fine-tuning on that would teach the model that
    cars are background. With --vehicle-weights, the Phase-1 model's confident
    vehicle detections are added to train/val as pseudo-labels. Never to test:
    scoring a model against its own predictions is circular.

Class scheme is identical to Phase 1 (0 person, 1 vehicle, 2 animal), so the
detection head of visdrone_base.pt is reused, not reinitialized.

Usage (Colab):
    python prepare_flood_c2a.py --src /content/datasets/flood_c2a_coco \\
                                --out /content/datasets/flood_c2a \\
                                --vehicle-weights /content/drive/MyDrive/FloodRescue/models/visdrone_base.pt
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = ["person", "vehicle", "animal"]
SPLITS = {"train": "train", "valid": "val", "test": "test"}   # export dir -> ours
PSEUDO_SPLITS = {"train", "val"}


def source_key(file_name: str) -> str:
    """Roboflow names augmented copies '<source>.rf.<hash>.jpg'."""
    return file_name.split(".rf.")[0]


def iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def yolo_line(cls: int, x1, y1, x2, y2, W, H) -> str:
    return (f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")


def size_bucket(stats, area):
    # COCO definition, same as Phase 1, so the two datasets are comparable
    if area < 32 ** 2:
        stats["small_boxes_coco"] += 1
    elif area < 96 ** 2:
        stats["medium_boxes_coco"] += 1
    else:
        stats["large_boxes_coco"] += 1


def convert_split(src: Path, dst: Path, copies_per_source: int, seed: int,
                  min_side_px: float = 2.0):
    """Returns (stats, {image path: [person xyxy boxes]}) for pseudo-labelling."""
    coco = json.loads((src / "_annotations.coco.json").read_text())
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    images = sorted(coco["images"], key=lambda im: im["file_name"])
    by_source = defaultdict(list)
    for im in images:
        by_source[source_key(im["file_name"])].append(im)
    if copies_per_source > 0:
        rng = random.Random(seed)
        keep = []
        for key in sorted(by_source):
            group = by_source[key]
            keep += group if len(group) <= copies_per_source else rng.sample(group, copies_per_source)
        images = sorted(keep, key=lambda im: im["file_name"])

    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    for sub in ("images", "labels"):
        if (dst / sub).exists():
            shutil.rmtree(dst / sub)
        (dst / sub).mkdir(parents=True)

    stats = {
        "images_in_export": len(coco["images"]), "sources": len(by_source),
        "images": 0, "images_with_no_objects": 0, "boxes_kept": 0,
        "skipped_degenerate": 0, "source_categories": Counter(),
        "per_class": Counter(),
        "small_boxes_coco": 0, "medium_boxes_coco": 0, "large_boxes_coco": 0,
    }
    persons = {}
    for im in images:
        W, H = float(im["width"]), float(im["height"])
        lines, boxes = [], []
        for a in anns[im["id"]]:
            x, y, w, h = a["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 - x1 < min_side_px or y2 - y1 < min_side_px:
                stats["skipped_degenerate"] += 1
                continue
            stats["source_categories"][cat_names[a["category_id"]]] += 1
            lines.append(yolo_line(0, x1, y1, x2, y2, W, H))
            boxes.append((x1, y1, x2, y2))
            stats["boxes_kept"] += 1
            stats["per_class"]["person"] += 1
            size_bucket(stats, (x2 - x1) * (y2 - y1))

        img_out = dst / "images" / im["file_name"]
        shutil.copy2(src / im["file_name"], img_out)
        (dst / "labels" / f"{img_out.stem}.txt").write_text("\n".join(lines))
        persons[img_out] = boxes
        stats["images"] += 1
        if not lines:
            stats["images_with_no_objects"] += 1

    stats["source_categories"] = dict(stats["source_categories"])
    stats["per_class"] = dict(stats["per_class"])
    return stats, persons


def add_vehicle_pseudo_labels(persons: dict, weights: str, conf: float, imgsz: int,
                              max_person_iou: float = 0.5, chunk: int = 32) -> dict:
    """Append the base model's confident vehicle boxes to existing label files."""
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    paths = sorted(persons)
    added = dropped_overlap = imgs_with_vehicle = 0
    for i in range(0, len(paths), chunk):
        batch = paths[i:i + chunk]
        results = model.predict([str(p) for p in batch], imgsz=imgsz, conf=conf,
                                classes=[1], device=device, verbose=False)
        for p, r in zip(batch, results):
            H, W = r.orig_shape
            new = []
            for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
                # a "vehicle" sitting on a labelled person is the person, misread
                if any(iou((x1, y1, x2, y2), b) >= max_person_iou for b in persons[p]):
                    dropped_overlap += 1
                    continue
                new.append(yolo_line(1, x1, y1, x2, y2, W, H))
            if new:
                lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
                old = lbl.read_text()
                lbl.write_text("\n".join(([old] if old else []) + new))
                added += len(new)
                imgs_with_vehicle += 1
        print(f"  pseudo-labelled {min(i + chunk, len(paths))}/{len(paths)}", end="\r")
    print()
    return {"vehicle_boxes_added": added, "images_with_vehicle": imgs_with_vehicle,
            "dropped_overlapping_person": dropped_overlap,
            "conf": conf, "imgsz": imgsz, "weights": weights}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Roboflow COCO export (has train/ valid/ test/)")
    ap.add_argument("--out", required=True, help="YOLO dataset output dir")
    ap.add_argument("--copies-per-source", type=int, default=2,
                    help="augmented copies kept per train source image (0 = all 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vehicle-weights", default=None,
                    help="Phase-1 weights; adds vehicle pseudo-labels to train/val")
    ap.add_argument("--vehicle-conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960, help="inference size for pseudo-labelling")
    ap.add_argument("--stats-out", default=None,
                    help="path for the conversion stats JSON (default <out>/flood_c2a_stats.json)")
    args = ap.parse_args()

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    all_stats = {"copies_per_source": args.copies_per_source, "seed": args.seed}
    for theirs, ours in SPLITS.items():
        print(f"[convert] {theirs} -> {ours}")
        # only train carries augmented copies; valid/test are kept whole
        cps = args.copies_per_source if ours == "train" else 0
        s, persons = convert_split(src / theirs, out / ours, cps, args.seed)
        print(f"  images={s['images']} (of {s['images_in_export']}, {s['sources']} sources)  "
              f"boxes={s['boxes_kept']}  from={s['source_categories']}  "
              f"small(COCO)={s['small_boxes_coco']} "
              f"({100 * s['small_boxes_coco'] / max(1, s['boxes_kept']):.1f}%)")
        if args.vehicle_weights and ours in PSEUDO_SPLITS:
            print(f"[pseudo] vehicles in {ours} from {args.vehicle_weights} (conf>={args.vehicle_conf})")
            s["pseudo_vehicle"] = add_vehicle_pseudo_labels(
                persons, args.vehicle_weights, args.vehicle_conf, args.imgsz)
            s["per_class"]["vehicle"] = s["pseudo_vehicle"]["vehicle_boxes_added"]
            print(f"  {s['pseudo_vehicle']}")
        all_stats[ours] = s

    yaml_path = out / "flood_c2a.yaml"
    yaml_path.write_text(
        "# Auto-generated by prepare_flood_c2a.py - Phase 2 flood fine-tune\n"
        f"path: {out.as_posix()}\n"
        "train: train/images\n"
        "val: val/images     # shares backgrounds with train: epoch selection only\n"
        "test: test/images   # real flood photos, person labels only: report on this\n"
        "nc: 3\n"
        "names:\n"
        "  0: person\n"
        "  1: vehicle   # pseudo-labels from the Phase-1 model (train/val only)\n"
        "  2: animal    # no instances in C2A; kept so the Phase-1 head is reused\n"
    )
    print(f"[convert] wrote {yaml_path}")

    stats_path = Path(args.stats_out) if args.stats_out else out / "flood_c2a_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"[convert] wrote {stats_path}")


if __name__ == "__main__":
    main()
