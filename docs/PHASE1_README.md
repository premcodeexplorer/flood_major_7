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

Published VisDrone numbers look lower than these because they are over 10
fine-grained classes; merging 8 vehicle types into one class legitimately raises
mAP. Do not compare our number against theirs in the report — compare only
against our own Phase-2 fine-tuned model.

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
