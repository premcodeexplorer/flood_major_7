"""
Phase 1 - Step 2: Train the Model-1 BASE detector on VisDrone (YOLOv8).

Tuned for one-shot success on Colab:
  * imgsz / batch auto-selected from the GPU actually allocated
  * mosaic kept on (big win for small objects), disabled for the last epochs
  * save_period checkpointing + --resume, so a Colab disconnect costs nothing
  * writes metrics JSON + copies best.pt to models/ for Phase 2

Usage (Colab):
    python train_visdrone.py --data /content/datasets/VisDrone/visdrone.yaml \
                             --project /content/drive/MyDrive/FloodRescue/runs
    python train_visdrone.py ... --resume      # after a disconnect
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# GPU-name substring -> (imgsz, batch).  Fallback used for anything unlisted.
GPU_PRESETS = [
    ("a100", (1280, 16)),
    ("h100", (1280, 24)),
    ("l4",   (1024, 10)),
    ("v100", (1024,  8)),
    ("t4",   (1024,  8)),
    ("4090", (1280, 12)),
]
FALLBACK = (1024, 8)


def pick_preset():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU. In Colab: Runtime > Change runtime type > GPU.")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    low = name.lower()
    for key, preset in GPU_PRESETS:
        if key in low:
            print(f"[gpu] {name} ({vram:.0f} GB) -> imgsz={preset[0]} batch={preset[1]}")
            return preset
    print(f"[gpu] {name} ({vram:.0f} GB) unlisted -> imgsz={FALLBACK[0]} batch={FALLBACK[1]}")
    return FALLBACK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to visdrone.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--project", default="runs/phase1",
                    help="put this on Google Drive so checkpoints survive a disconnect")
    ap.add_argument("--name", default="visdrone_base")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=0, help="0 = auto from GPU")
    ap.add_argument("--batch", type=int, default=0, help="0 = auto from GPU")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export-to", default="models",
                    help="dir to copy best.pt into as visdrone_base.pt")
    args = ap.parse_args()

    auto_imgsz, auto_batch = pick_preset()
    imgsz = args.imgsz or auto_imgsz
    batch = args.batch or auto_batch

    run_dir = Path(args.project) / args.name
    last = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last.exists():
            raise SystemExit(f"--resume given but {last} not found")
        print(f"[resume] {last}")
        model = YOLO(str(last))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=imgsz,
            batch=batch,
            device=0,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            # --- optimisation ---
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            cos_lr=True,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            patience=15,
            # --- augmentation: biased toward small, dense aerial targets ---
            mosaic=1.0,
            close_mosaic=10,      # last 10 epochs clean, so val-time stats match
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,           # drone imagery is gravity-oriented
            degrees=0.0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            erasing=0.0,
            # --- runtime ---
            amp=True,
            cache=False,          # VisDrone at 1024+ will not fit in Colab RAM
            save_period=5,        # checkpoint every 5 epochs
            plots=True,
            seed=0,
            val=True,
        )

    save_dir = Path(results.save_dir if hasattr(results, "save_dir") else run_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"[train] done -> {best}")

    # ---- final validation at the trained resolution -------------------------
    print("[val] running final validation on the VisDrone val split")
    metrics = YOLO(str(best)).val(
        data=args.data, imgsz=imgsz, batch=batch, device=0,
        project=args.project, name=f"{args.name}_val", exist_ok=True, plots=True,
    )
    box = metrics.box
    summary = {
        "weights": str(best),
        "imgsz": imgsz,
        "batch": batch,
        "mAP50-95": round(float(box.map), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP75": round(float(box.map75), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_mAP50-95": {
            metrics.names[c]: round(float(box.maps[c]), 4) for c in range(len(box.maps))
        },
    }
    print(json.dumps(summary, indent=2))

    out = save_dir / "phase1_visdrone_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[val] wrote {out}")

    if args.export_to:
        dest_dir = Path(args.export_to)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest_dir / "visdrone_base.pt")
        shutil.copy2(out, dest_dir / "phase1_visdrone_metrics.json")
        print(f"[export] {dest_dir / 'visdrone_base.pt'}  <-- feed this to Phase 2")


if __name__ == "__main__":
    main()
