# Phase 1 — Model 1 Base Training on VisDrone

## What you run

Open **`notebooks/Phase1_VisDrone_Colab.ipynb`** in Google Colab and run every cell
top to bottom. That notebook is self-contained — it writes both scripts into the
Colab VM itself, so nothing needs uploading except the notebook.

`Runtime > Change runtime type > GPU` before you start. The notebook adapts
`imgsz`/`batch` to whichever GPU you get.

| GPU | imgsz | batch | ~time/epoch | 60 epochs |
|---|---|---|---|---|
| A100 | 1280 | 16 | ~1 min | ~1 h |
| L4 | 1024 | 10 | ~2.5 min | ~2.5 h |
| T4 | 1024 | 8 | ~4 min | ~4 h |

## Source of truth

| File | Role |
|---|---|
| `scripts/visdrone_to_yolo.py` | VisDrone DET annotations → YOLO labels + `visdrone.yaml` |
| `scripts/train_visdrone.py` | YOLOv8 base training + final validation + metrics export |
| `scripts/_gen_phase1_notebook.py` | Rebuilds the Colab notebook from those two scripts |
| `scripts/_test_converter.py` | Converter smoke test (synthetic fixture) — run after any edit |

The notebook is **generated**. Edit the `.py` files, then:

```
python scripts/_test_converter.py
python scripts/_gen_phase1_notebook.py
```

## Class scheme — important

| id | name | VisDrone source |
|---|---|---|
| 0 | person | pedestrian (1), people (2) |
| 1 | vehicle | bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10) |
| 2 | animal | *none* — reserved for Phase 2 |

`nc=3` even though VisDrone has no animals. This is deliberate and differs from
the original plan's `nc=2`: changing the class count between Stage 1 and Stage 2
forces Ultralytics to reinitialize the detection head, discarding the
classification weights learned here. Keeping `nc=3` makes Phase 2 a true
fine-tune.

Dropped during conversion: category `0` (ignored regions), category `11`
(others), and any box with `score == 0`. Boxes overflowing the frame are clamped;
boxes under 2 px on either side after clamping are dropped.

## Expected results (`yolov8s`, imgsz 1024–1280)

| metric | expect |
|---|---|
| mAP@50 | 0.42 – 0.52 |
| mAP@50-95 | 0.24 – 0.31 |
| person mAP@50 | 0.35 – 0.45 |
| vehicle mAP@50 | 0.50 – 0.62 |
| animal | 0.0 (correct — no instances) |

### Context for the report

| reference point | mAP@50 | task |
|---|---|---|
| YOLOv8s baseline, published | 0.25 - 0.30 | VisDrone, **10 fine-grained classes** |
| Specialized 2024 methods | 0.40 - 0.50 | VisDrone, **10 fine-grained classes** |
| **Ours (Phase 1)** | **0.42 - 0.52** | VisDrone, **2 merged classes** |

Ours looks higher mainly because merging 8 vehicle types into one `vehicle`
class removes the hardest sub-problem (van vs. truck vs. bus). That is a
legitimate simplification for rescue, but it makes the tasks non-comparable:
**do not put our number next to published VisDrone results in the report.**
The only valid comparison is against our own Phase-2 fine-tuned model, measured
on the same flood test split.

### Why the number is capped low (for the viva)

1. **Tiny objects** - a pedestrian at drone altitude is 20-40 px tall; see the
   COCO small-object percentage in the conversion stats.
2. **Extreme density** - 300+ heavily-occluding objects in some frames, plus
   night, motion blur, and 10-100 m altitude variation.
3. **mAP@50-95 is structurally unfair to small boxes** - a 3 px error shifts IoU
   by ~0.15 on a 20 px box but is noise on a 200 px box. A metric property, not
   a model failure.

Note for the presentation: mAP weights precision and recall equally, but in
rescue a **missed victim costs a life while a false positive costs a wasted
glance**. mAP therefore understates the system's usefulness at the low
confidence threshold we deliberately adopt in Phase 8.

### What raises the number that actually matters

| Phase | Attacks |
|---|---|
| 2 - flood fine-tune | Domain gap: dry city streets -> water, rooftops, debris |
| 3 - SAHI | Small objects head-on; slicing makes a 25 px victim look like 150 px |
| 4 - tracking | Temporal recall: a victim missed in frame 37 is still tracked from 36 |

Phase 1 is the floor, not the ceiling.

## Deliverables → where they go

| Artifact | Local path | Used by |
|---|---|---|
| Base weights | `models/visdrone_base.pt` | **Phase 2** (fine-tune starting point) |
| Baseline metrics | `models/phase1_visdrone_metrics.json` | **Phase 9** ("fine-tuning helps" comparison) |
| Conversion stats | `outputs/phase1/visdrone_conversion_stats.json` | **Phase 3** (COCO small-object % justifies SAHI) |
| Curves / confusion matrix | `outputs/phase1/runs/` | Report figures |

Download all four from Drive at the end of the notebook (cell 11) and commit the
two JSONs — the `.pt` is gitignored.

## If Colab disconnects

Nothing is lost. Checkpoints write to Drive every 5 epochs. Re-run cells 1–7
(download and convert both skip if already present), then run cell **8b**
(`--resume`) instead of cell 8.
