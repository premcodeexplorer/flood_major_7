"""
Model 2 - Step 3: Build the comparison table from every results/<run>.json.

Writes, next to the results folder:
  comparison.md   - paper-ready markdown table (published rows marked as context)
  comparison.csv  - same numbers, for Excel / LaTeX
  per_class.png   - per-class IoU bars, hard classes highlighted
  ablation.md     - each change vs plain U-Net (scSE only, loss only, both)

Published rows are FloodNet Table 4 (Rahnemoonfar et al., IEEE Access 2021):
9-class mIoU, 713x713 crops, PSPNet with ResNet-101, their own training setup.
They are CONTEXT ONLY - different resolution, backbone and schedule.

Usage:
    python floodnet_compare.py --results /content/drive/MyDrive/FloodSeg/results
"""

import argparse
import csv
import json
from pathlib import Path

CLASSES = ["building-flooded", "building-non-flooded", "road-flooded", "road-non-flooded",
           "water", "tree", "vehicle", "pool", "grass"]
HARD = ["building-flooded", "road-flooded", "vehicle", "pool"]
SHORT = {"building-flooded": "Bld-F", "building-non-flooded": "Bld-NF", "road-flooded": "Road-F",
         "road-non-flooded": "Road-NF", "water": "Water", "tree": "Tree", "vehicle": "Vehicle",
         "pool": "Pool", "grass": "Grass"}

# FloodNet paper, Table 4 "Per-class results on FloodNet testing set" (verified from arXiv:2012.02951)
PUBLISHED = {
    "ENet [FloodNet]":       [6.94, 47.35, 12.49, 48.43, 48.95, 68.36, 32.26, 42.49, 76.23, 42.61],
    "DeepLabV3+ [FloodNet]": [32.7, 72.8, 52.00, 70.2, 75.2, 77.00, 42.5, 47.1, 84.3, 61.53],
    "PSPNet-R101 [FloodNet]": [68.93, 89.75, 82.16, 91.18, 92.00, 89.55, 46.15, 64.19, 93.29, 79.69],
}
NAMES = {
    "unet_r50": "U-Net (R50)", "deeplabv3p_r50": "DeepLabV3+ (R50)", "pspnet_r50": "PSPNet (R50)",
    "unet_r50_scse_hybrid": "**Ours**: U-Net (R50) + scSE + Dice/Focal",
    "unet_r50_scse": "U-Net (R50) + scSE (CE) [ablation]",
    "unet_r50_hybrid": "U-Net (R50) + Dice/Focal [ablation]",
}
ORDER = ["unet_r50", "deeplabv3p_r50", "pspnet_r50", "unet_r50_scse", "unet_r50_hybrid",
         "unet_r50_scse_hybrid"]
LADDER = [("unet_r50", "U-Net R50 (CE)"), ("unet_r50_scse", "+ scSE only (CE)"),
          ("unet_r50_hybrid", "+ Dice/Focal only"), ("unet_r50_scse_hybrid", "+ scSE + Dice/Focal (ours)")]


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    rdir = Path(args.results)
    res = {}
    for p in sorted(rdir.glob("*.json")):
        r = json.loads(p.read_text())
        res[r["run"]] = r
    if not res:
        raise SystemExit(f"no results/*.json in {rdir}")
    runs = [r for r in ORDER if r in res] + sorted(r for r in res if r not in ORDER)
    gpus = sorted({res[r]["gpu"] for r in runs})

    head = ["Model", "mIoU (9)"] + [SHORT[c] for c in CLASSES] + ["Hard-4", "Params (M)", "FPS"]
    rows = []
    for name, v in PUBLISHED.items():
        rows.append([f"{name} *", fmt(v[-1])] + [fmt(x) for x in v[:-1]]
                    + [fmt(sum(v[CLASSES.index(c)] for c in HARD) / 4), "-", "-"])
    for r in runs:
        t = res[r]["test"]
        rows.append([NAMES.get(r.split("_seed")[0], r) + (f" (seed {res[r]['seed']})" if res[r]["seed"] else ""),
                     fmt(t["mIoU9"])] + [fmt(t["per_class_IoU"][c]) for c in CLASSES]
                    + [fmt(res[r]["hard_class_mIoU"]), fmt(res[r]["params_M"]),
                       "-" if res[r]["fps"] is None else f"{res[r]['fps']:.1f}"])

    f0 = res[runs[0]]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    md += ["", "\\* Published, FloodNet Table 4: 713x713 crops, PSPNet uses ResNet-101, different "
           "training setup - context only, not a like-for-like comparison.",
           f"All other rows: trained by us on the official FloodNet split "
           f"({f0['splits']['train']}/{f0['splits']['val']}/{f0['splits']['test']}), frames "
           f"{f0['frame'][0]}x{f0['frame'][1]}, {f0['hparams']['crop']}px crops, "
           f"{f0['hparams']['epochs']} epochs, identical schedule; test split. "
           f"mIoU (9) excludes background, as in the paper. "
           f"Hard-4 = mean IoU of Bld-F, Road-F, Vehicle, Pool. FPS: batch 1, "
           f"{f0['frame'][0]}x{f0['frame'][1]}, fp16 on {', '.join(gpus)}."]
    if len(gpus) > 1:
        md.append("**Warning: runs used different GPUs - FPS values are not comparable. "
                  "Re-measure FPS for all models on one GPU before publishing.**")
    (rdir.parent / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    with open(rdir.parent / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([head] + rows)
    print("\n".join(md))

    lad = [(lbl, res[r]) for r, lbl in LADDER if r in res]
    if lad:
        base = lad[0][1]["test"]
        am = ["| Step | mIoU (9) | Δ | Hard-4 | Δ | Vehicle | Pool |", "|---|---|---|---|---|---|---|"]
        for lbl, r in lad:
            t = r["test"]
            am.append(f"| {lbl} | {t['mIoU9']:.2f} | {t['mIoU9'] - base['mIoU9']:+.2f} | "
                      f"{r['hard_class_mIoU']:.2f} | {r['hard_class_mIoU'] - lad[0][1]['hard_class_mIoU']:+.2f} | "
                      f"{fmt(t['per_class_IoU']['vehicle'])} | {fmt(t['per_class_IoU']['pool'])} |")
        (rdir.parent / "ablation.md").write_text("\n".join(am), encoding="utf-8")
        print("\n" + "\n".join(am))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(14, 5))
        w = 0.8 / len(runs)
        for i, r in enumerate(runs):
            vals = [res[r]["test"]["per_class_IoU"][c] or 0 for c in CLASSES]
            ax.bar(np.arange(len(CLASSES)) + i * w - 0.4 + w / 2, vals, w,
                   label=NAMES.get(r.split("_seed")[0], r).replace("**", ""))
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([SHORT[c] + (" *" if c in HARD else "") for c in CLASSES])
        ax.set_ylabel("test IoU (%)"); ax.set_ylim(0, 100)
        ax.set_title("FloodNet per-class IoU (* = hard class)")
        ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(rdir.parent / "per_class.png", dpi=150)
        print(f"\n[plot] {rdir.parent / 'per_class.png'}")
    except ImportError:
        print("[plot] matplotlib not installed - skipped")


if __name__ == "__main__":
    main()
