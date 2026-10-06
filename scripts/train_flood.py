"""
Phase 2 - Step 2: Fine-tune the Phase-1 VisDrone model on Flood C2A, then
compare it against the Phase-1 model on the same test data.

Fine-tune, not retrain:
  * starts from models/visdrone_base.pt with the same nc=3 head
  * lr0 10x lower than Phase 1 (0.001 vs 0.01), short warmup, no bias-lr spike
  * same imgsz as Phase 1 for the GPU, so both models are scored at one resolution

Comparison written to phase2_flood_metrics.json:
  * flood test (109 real photos, person only) - "does fine-tuning help?"
  * VisDrone val (optional, --visdrone-data) - "did it forget vehicles?"

Usage (Colab):
    python train_flood.py --data /content/datasets/flood_c2a/flood_c2a.yaml \\
                          --weights /content/drive/MyDrive/FloodRescue/models/visdrone_base.pt \\
                          --visdrone-data /content/datasets/VisDrone/visdrone_val.yaml \\
                          --project /content/drive/MyDrive/FloodRescue/runs
    python train_flood.py ... --resume      # after a disconnect
    python train_flood.py ... --val-only    # comparison only, no training
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from train_visdrone import pick_preset


def summarize(metrics) -> dict:
    box = metrics.box
    # As in Phase 1: box.maps fills classes with no instances with the overall
    # mAP. Only ap_class_index classes were actually measured. On the flood
    # test split that is person only.
    present = {int(c) for c in box.ap_class_index}
    return {
        "mAP50-95": round(float(box.map), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP75": round(float(box.map75), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_mAP50-95": {
            metrics.names[c]: (round(float(box.maps[c]), 4) if c in present else None)
            for c in range(len(box.maps))
        },
        "per_class_mAP50": {
            metrics.names[c]: (round(float(box.all_ap[i, 0]), 4))
            for i, c in enumerate(box.ap_class_index)
        },
        "classes_with_no_instances": [
            metrics.names[c] for c in range(len(box.maps)) if c not in present
        ],
    }


def evaluate(weights, data, split, imgsz, batch, project, name):
    m = YOLO(str(weights)).val(
        data=data, split=split, imgsz=imgsz, batch=batch, device=0,
        project=project, name=name, exist_ok=True, plots=True,
    )
    return summarize(m)


def delta(after: dict, before: dict) -> dict:
    keys = ["mAP50-95", "mAP50", "mAP75", "precision", "recall"]
    d = {k: round(after[k] - before[k], 4) for k in keys}
    d["per_class_mAP50-95"] = {
        c: (round(v - before["per_class_mAP50-95"][c], 4)
            if v is not None and before["per_class_mAP50-95"].get(c) is not None else None)
        for c, v in after["per_class_mAP50-95"].items()
    }
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to flood_c2a.yaml")
    ap.add_argument("--weights", required=True, help="Phase-1 visdrone_base.pt")
    ap.add_argument("--visdrone-data", default=None,
                    help="VisDrone yaml whose val split is the forgetting check")
    ap.add_argument("--project", default="runs/phase2")
    ap.add_argument("--name", default="flood_finetune")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze the first N layers (10 = whole backbone); 0 = train all")
    ap.add_argument("--imgsz", type=int, default=0, help="0 = auto from GPU, as Phase 1")
    ap.add_argument("--batch", type=int, default=0, help="0 = auto from GPU")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val-only", action="store_true",
                    help="skip training; just rerun the comparison")
    ap.add_argument("--finetuned", default=None,
                    help="with --val-only: fine-tuned weights (default <run>/weights/best.pt)")
    ap.add_argument("--export-to", default="models",
                    help="dir to copy best.pt into as flood_finetuned.pt")
    args = ap.parse_args()

    auto_imgsz, auto_batch = pick_preset()
    imgsz = args.imgsz or auto_imgsz
    batch = args.batch or auto_batch

    run_dir = Path(args.project) / args.name
    last = run_dir / "weights" / "last.pt"

    if args.val_only:
        best = Path(args.finetuned) if args.finetuned else run_dir / "weights" / "best.pt"
        if not best.exists():
            raise SystemExit(f"--val-only given but {best} not found")
        print(f"[val-only] skipping training, comparing {best}")
    else:
        if args.resume:
            if not last.exists():
                raise SystemExit(f"--resume given but {last} not found")
            print(f"[resume] {last}")
            results = YOLO(str(last)).train(resume=True)
        else:
            results = YOLO(args.weights).train(
                data=args.data,
                epochs=args.epochs,
                imgsz=imgsz,
                batch=batch,
                device=0,
                workers=args.workers,
                project=args.project,
                name=args.name,
                exist_ok=True,
                # --- optimisation: gentle, so VisDrone knowledge survives ---
                optimizer="SGD",
                lr0=args.lr0,
                lrf=0.1,
                cos_lr=True,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=1.0,
                warmup_bias_lr=0.0,   # default 0.1 would be 100x lr0 here
                freeze=args.freeze or None,
                patience=args.patience,
                # --- augmentation: as Phase 1 ---
                mosaic=1.0,
                close_mosaic=5,
                scale=0.5,
                translate=0.1,
                fliplr=0.5,
                flipud=0.0,           # Roboflow already baked flips/rotations in
                degrees=0.0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                erasing=0.0,
                # --- runtime ---
                amp=True,
                cache=False,
                save_period=5,
                plots=True,
                seed=0,
                val=True,
            )
        best = Path(getattr(results, "save_dir", None) or run_dir) / "weights" / "best.pt"
        print(f"[train] done -> {best}")

    # ---- comparison: Phase-1 base vs Phase-2 fine-tuned, same data, same imgsz ----
    report = {
        "base_weights": args.weights, "finetuned_weights": str(best),
        "imgsz": imgsz, "batch": batch,
        "flood_test": {}, "visdrone_val": {},
    }
    print("[compare] flood test split (real photos, person only)")
    report["flood_test"]["base"] = evaluate(
        args.weights, args.data, "test", imgsz, batch, args.project, f"{args.name}_test_base")
    report["flood_test"]["finetuned"] = evaluate(
        best, args.data, "test", imgsz, batch, args.project, f"{args.name}_test_finetuned")
    report["flood_test"]["delta"] = delta(report["flood_test"]["finetuned"],
                                          report["flood_test"]["base"])

    if args.visdrone_data:
        print("[compare] VisDrone val (forgetting check)")
        report["visdrone_val"]["base"] = evaluate(
            args.weights, args.visdrone_data, "val", imgsz, batch, args.project,
            f"{args.name}_visdrone_base")
        report["visdrone_val"]["finetuned"] = evaluate(
            best, args.visdrone_data, "val", imgsz, batch, args.project,
            f"{args.name}_visdrone_finetuned")
        report["visdrone_val"]["delta"] = delta(report["visdrone_val"]["finetuned"],
                                                report["visdrone_val"]["base"])

    print(json.dumps(report, indent=2))
    out = run_dir / "phase2_flood_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[compare] wrote {out}")

    t = report["flood_test"]
    print(f"\nflood test  person mAP@50: base {t['base']['mAP50']:.3f} -> "
          f"fine-tuned {t['finetuned']['mAP50']:.3f} ({t['delta']['mAP50']:+.3f})")
    if report["visdrone_val"]:
        v = report["visdrone_val"]
        print(f"VisDrone val       mAP@50: base {v['base']['mAP50']:.3f} -> "
              f"fine-tuned {v['finetuned']['mAP50']:.3f} ({v['delta']['mAP50']:+.3f})")

    if args.export_to:
        dest_dir = Path(args.export_to)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "flood_finetuned.pt"
        if not (dest.exists() and dest.samefile(best)):
            shutil.copy2(best, dest)
        shutil.copy2(out, dest_dir / "phase2_flood_metrics.json")
        print(f"[export] {dest}  <-- feed this to Phase 3")


if __name__ == "__main__":
    main()
