"""
Phase 3 comparison: plain YOLO vs YOLO + SAHI, on the 109 real flood test photos.

Same detector (models/flood_rehearsal.pt), same images, same confidence
floor, same scorer - only tiling changes. Scored with pycocotools (COCO
protocol), which also gives the small-object AP the project plan asks for
(COCO "small" = under 32x32 px). Every Roboflow category in the export is a
person, so all ground truth is merged into one "person" class.

pycocotools interpolates precision slightly differently from Ultralytics,
so the plain-YOLO row here will not match train_flood.py's 0.399 to the
third decimal. Compare rows within this table, not across tools.

SAHI is built for high-resolution footage where people are a few pixels
tall. The flood test photos are only 640 px, so the same comparison should
also be run on VisDrone val (1-2K px, many tiny people) via --yolo-labels.

Usage:
    python scripts/eval_sahi.py --images data/flood_c2a/test \\
        --coco data/flood_c2a/test/_annotations.coco.json --out outputs/rescue/sahi_eval.json
    python scripts/eval_sahi.py ... --slices 256 320 416 --max-images 20
    python scripts/eval_sahi.py --images /content/datasets/VisDrone/VisDrone2019-DET-val/images \
        --yolo-labels /content/datasets/VisDrone/VisDrone2019-DET-val/labels --slices 512 640 \
        --out outputs/rescue/sahi_eval_visdrone.json
"""

import argparse
import contextlib
import copy
import io
import json
import time
from pathlib import Path

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rescue.detect import PERSON, Detector

ROOT = Path(__file__).resolve().parent.parent
STAT_NAMES = ["AP50-95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR@10", "AR@100", "AR@300", "AR_small", "AR_medium", "AR_large"]


def person_only(gt_json):
    d = copy.deepcopy(gt_json)
    d["categories"] = [{"id": 1, "name": "person", "supercategory": "none"}]
    for a in d["annotations"]:
        a["category_id"] = 1
    return d


def yolo_to_coco(images_dir, labels_dir, cls=0):
    """COCO ground truth from YOLO txt labels, keeping one class (person = 0)."""
    from PIL import Image
    images, anns = [], []
    files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for iid, p in enumerate(files):
        w, h = Image.open(p).size
        images.append(dict(id=iid, file_name=p.name, width=w, height=h))
        lp = Path(labels_dir) / f"{p.stem}.txt"
        for line in (lp.read_text().splitlines() if lp.exists() else []):
            v = line.split()
            if len(v) == 5 and int(v[0]) == cls:
                cx, cy, bw, bh = (float(x) for x in v[1:])
                anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=1, iscrowd=0,
                                 bbox=[(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                                 area=bw * w * bh * h))
    return dict(images=images, annotations=anns,
                categories=[{"id": 1, "name": "person", "supercategory": "none"}])


def evaluate(gt, dets, img_ids):
    if not dets:
        return {k: 0.0 for k in STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = img_ids
        E.params.maxDets = [10, 100, 300]          # crowded scenes: up to ~300 people
        E.evaluate()
        E.accumulate()
        E.summarize()
    return {k: round(float(v), 4) for k, v in zip(STAT_NAMES, E.stats)}


def run(detector, gt, images_dir, img_ids):
    dets, secs, n_boxes = [], 0.0, 0
    for iid in img_ids:
        info = gt.imgs[iid]
        frame = cv2.imread(str(images_dir / info["file_name"]))
        if frame is None:
            raise SystemExit(f"missing image {info['file_name']}")
        t = time.perf_counter()
        d = detector(frame)
        secs += time.perf_counter() - t
        keep = d.cls == PERSON
        for (x1, y1, x2, y2), s in zip(d.xyxy[keep], d.conf[keep]):
            dets.append(dict(image_id=iid, category_id=1, score=float(s),
                             bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]))
        n_boxes += int(keep.sum())
    return dets, secs / len(img_ids), n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data/flood_c2a/test"))
    ap.add_argument("--coco", default=str(ROOT / "data/flood_c2a/test/_annotations.coco.json"))
    ap.add_argument("--yolo-labels", default=None, help="YOLO txt label dir instead of --coco (person = 0)")
    ap.add_argument("--weights", default=str(ROOT / "models/flood_rehearsal.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.01, help="low floor, as for any mAP evaluation")
    ap.add_argument("--slices", type=int, nargs="+", default=[320, 416])
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs/rescue/sahi_eval.json"))
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = (yolo_to_coco(args.images, args.yolo_labels) if args.yolo_labels
                      else person_only(json.loads(Path(args.coco).read_text())))
        gt.createIndex()
    img_ids = sorted(gt.getImgIds())[: args.max_images or None]
    n_gt = sum(len(gt.getAnnIds(imgIds=[i])) for i in img_ids)
    print(f"[data] {len(img_ids)} images, {n_gt} person boxes")

    configs = [("YOLO", dict(sahi=False))] + [
        (f"YOLO + SAHI ({s}px tiles)", dict(sahi=True, slice_size=s)) for s in args.slices]
    rows = []
    for name, kw in configs:
        det = Detector(args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device,
                       overlap=args.overlap, **kw)
        dets, sec, nb = run(det, gt, Path(args.images), img_ids)
        m = evaluate(gt, dets, img_ids)
        rows.append(dict(method=name, sec_per_image=round(sec, 3), boxes=nb, **m))
        print(f"{name:28s} AP50 {m['AP50']:.3f}  AP50-95 {m['AP50-95']:.3f}  "
              f"AP_small {m['AP_small']:.3f}  AR@300 {m['AR@300']:.3f}  {sec:.2f}s/img")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = dict(weights=args.weights, imgsz=args.imgsz, conf=args.conf, overlap=args.overlap,
               images=len(img_ids), gt_boxes=n_gt, scorer="pycocotools (COCO protocol), person only",
               results=rows)
    out.write_text(json.dumps(res, indent=2))

    keys = ["AP50", "AP50-95", "AP75", "AP_small", "AP_medium", "AR@300", "sec_per_image"]
    md = ["| Method | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for r in rows:
        md.append(f"| {r['method']} | " + " | ".join(f"{r[k]:.3f}" if k != "sec_per_image"
                                                    else f"{r[k]:.2f}" for k in keys) + " |")
    md.append(f"\n{len(img_ids)} images from {Path(args.images).name} ({n_gt} people), COCO protocol via pycocotools, "
              f"conf >= {args.conf}, imgsz {args.imgsz}. AP_small = people under 32x32 px. "
              f"sec_per_image depends on the GPU.")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
