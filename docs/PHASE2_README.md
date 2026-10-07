# Phase 2 — Fine-tune Model 1 on Flood C2A

## What you run

1. Upload `data/flood_c2a_coco.zip` (the Roboflow COCO export zipped, `train/
   valid/ test/` at the top level, ~1.1 GB) to **`MyDrive/FloodRescue/data/`**.
   Rebuild it from `data/flood_c2a/` if it is missing; `*.zip` is gitignored.
2. Open **`notebooks/Phase2_FloodC2A_Colab.ipynb`** in Google Colab and run every
   cell top to bottom once. It also needs `models/visdrone_base.pt` on Drive
   (Phase 1 put it there).

| GPU | imgsz | ~time/epoch | epochs | ~total incl. comparison |
|---|---|---|---|---|
| A100 | 1280 | ~1 min | 30 | ~45 min |
| T4 | 960 | ~3 min | 30 | ~1.8 h |

Estimates scaled from the measured Phase-1 T4 throughput (6,471 images in
~4.2 min/epoch → 4,158 images here). `patience=8` usually stops it early.

## Source of truth

| File | Role |
|---|---|
| `scripts/prepare_flood_c2a.py` | Roboflow COCO export → YOLO labels, train subsampling, vehicle pseudo-labels, `flood_c2a.yaml` |
| `scripts/train_flood.py` | Fine-tune from `visdrone_base.pt`, then base vs fine-tuned comparison |
| `scripts/_gen_phase2_notebook.py` | Rebuilds the Colab notebook from the scripts |
| `scripts/_test_flood_prep.py` | Converter smoke test (synthetic fixture) — run after any edit |

The notebook is **generated**. Edit the `.py` files, then:

```
python scripts/_test_flood_prep.py
python scripts/_gen_phase2_notebook.py
```

## Dataset choice

**Flood C2A v2 "Just Original"**: Roboflow `wsar/flood-c2a-8vj8c`, CC BY 4.0.
This is the flood subset of C2A (Nihal et al., ICPR 2024, reference [3]). Local
copy: `data/flood_c2a/` (gitignored).

### Rejected: `dados_desastre_2` (Roboflow, 703 images)

It was the only candidate with person, vehicle and animal labels. Inspection showed:

1. **Not aerial.** Mostly ground-level news and phone photos. Median box ~117 px
   versus mostly under 32 px in VisDrone, so fine-tuning on it would pull the model
   away from the drone view.
2. **Animals not in floods.** Most animal boxes are close-up goat and horse portraits
   on farmland, many colour-inverted.
3. **Train/test leakage.** Augmented copies were split *after* augmentation:
   46 test–train and 81 valid–train near-duplicate pairs (the test split has 57 images).

### What the C2A export contains

Measured from the export with `prepare_flood_c2a.py`:

| split | in export | kept | boxes | small (<32²) | notes |
|---|---|---|---|---|---|
| train | 20,790 | **4,158** | 126,300 | 76.9 % | 2,079 scenes × 10 Roboflow copies → 2 kept |
| val | 541 | 541 | 18,018 | 86.9 % | every background also in train |
| test | 109 | 109 | 1,329 | 67.1 % | **real** flood/landslide photos, disjoint from train |

Dropped: boxes under 2 px on either side (train 1,530, val 514, test 1). These are
1–1.5 px slivers, the same rule as Phase 1.

### The catches, and what was done

| catch | handling |
|---|---|
| Training people are **composited** onto real aerial backgrounds | Report gains as synthetic→real transfer, measured on the real test photos |
| Roboflow baked **10 augmented copies** of each training image (flips, 90° rotations, blur) | Keep 2 per source (`--copies-per-source`); Ultralytics augments on the fly |
| **val shares backgrounds** with train | Use val only to select the best epoch. **Report on test** |
| **No vehicle labels.** Fine-tuning would teach "cars are background" | Phase-1 model's vehicle detections at conf ≥ 0.5 are added to train/val as pseudo-labels, minus any overlapping a labelled person (IoU ≥ 0.5). Not to test, because that would be circular |
| **No animal labels** | `animal` stays empty, `nc=3` kept so the Phase-1 head is reused. Stated as a limitation |
| Labels `human` / `0` / `1` | All → `person` (`1` is 34 stray boxes in val) |

## Training (fine-tune, not retrain)

| | Phase 1 | Phase 2 |
|---|---|---|
| start weights | `yolov8s.pt` (COCO) | `visdrone_base.pt` |
| lr0 → final | 0.01 → 1e-4 | **0.001 → 1e-4** |
| warmup | 3 epochs, bias lr 0.1 | 1 epoch, bias lr 0 (0.1 would be 100× lr0) |
| close_mosaic | 10 | 5 |
| imgsz | auto per GPU | same preset, so both models are scored at one resolution |

`--freeze 10` (frozen backbone) is available as an ablation but is off by default.

## The comparison it produces

`models/phase2_flood_metrics.json`, for both `base` and `finetuned`, plus `delta`:

| block | data | question |
|---|---|---|
| `flood_test` | 109 real photos, person only | Does fine-tuning help? This is comparison #1 for Phase 9 |
| `visdrone_val` | VisDrone val, person + vehicle | Did it forget? Watch `vehicle` |

On `flood_test`, `vehicle` and `animal` are `null` because they have no
instances. As in Phase 1, Ultralytics would otherwise fill them with the overall mAP.

### Reading the result

- A **flood-test mAP increase** is the headline result.
- A **small VisDrone drop** (a few points) is the normal cost of specialising.
  A large `vehicle` drop means the pseudo-labels were too sparse: re-run cell 6
  with `--vehicle-conf 0.35`.
- The test set is small (109 images, 1,329 people), and it includes landslides as
  well as floods. Say so when presenting the numbers.

## Result of v1 (C2A only) - fine-tuning made it worse

Measured 2026-10-06, T4, imgsz 960 (`models/phase2_flood_metrics_v1_c2a_only.json`):

| | Phase 1 | v1 (C2A only) | change |
|---|---|---|---|
| flood test, person mAP50 | 0.324 | 0.222 | -0.102 |
| flood test, mAP50-95 | 0.136 | 0.096 | -0.040 |
| VisDrone val, mAP50 | 0.699 | 0.372 | -0.328 |
| VisDrone val, person mAP50 | 0.588 | 0.186 | -0.402 |
| VisDrone val, vehicle mAP50 | 0.810 | 0.557 | -0.253 |

Per-size recall on the flood test split (conf 0.25, IoU 0.5) was lower at every size:
<16 px 0.23 -> 0.12, 16-32 px 0.33 -> 0.31, 32-96 px 0.30 -> 0.26, >96 px 0.46 -> 0.00.
False positives rose from 217 to 322. v1 *did* win on ~6 muddy disaster scenes where
Phase 1 found nobody (`outputs/phase2/figures/base_vs_finetuned_test_v1.jpg`).

**Diagnosis.** C2A's people are pasted cut-outs, often oversized relative to the scene,
lying or sitting, with hard edges. On 150 C2A training images Phase 1 confidently
recognised only 181 of 4,427 labelled people (4 %)
(`outputs/phase2/figures/c2a_pasted_people_vs_phase1.jpg`, green = label).
Training on them alone overwrote the model's notion of "person": this is catastrophic
forgetting across a synthetic-to-real gap. Unlabelled real people in C2A backgrounds
(red boxes in the same figure) contribute a little but are not the main cause.

**Fix: rehearsal (Phase 2b).** `notebooks/Phase2b_Rehearsal_Colab.ipynb` fine-tunes
from Phase 1 on **VisDrone train (6,471) + C2A (2,079, 1 copy per scene)** mixed,
about 3 real images per synthetic one, for 10 epochs at lr0 0.001. Validation uses both
val sets. It exports `models/flood_rehearsal.pt` + `flood_rehearsal_metrics.json`
and prints Phase 1 / v1 / rehearsal side by side. For the report, "naive fine-tuning on
synthetic data hurts, and rehearsal is the remedy" is the finding, whichever way
the rehearsal numbers land.

## Result of Phase 2b (rehearsal) - the final Model 1

Measured 2026-10-07, T4, imgsz 960 (`models/flood_rehearsal_metrics.json`). 10 epochs
completed; the saved weights are epoch 7's (best mixed-val mAP50-95, 0.398).

| | Phase 1 | v1 (C2A only) | **rehearsal** | rehearsal vs Phase 1 |
|---|---|---|---|---|
| flood test, person mAP50 | 0.324 | 0.222 | **0.399** | +0.075 (+23 %) |
| flood test, mAP50-95 | 0.136 | 0.096 | **0.176** | +0.040 |
| flood test, recall | 0.300 | 0.258 | **0.383** | +0.083 |
| flood test, precision | 0.581 | 0.409 | 0.560 | -0.022 |
| VisDrone val, mAP50 | 0.699 | 0.372 | **0.698** | -0.001 |
| VisDrone val, person mAP50 | 0.588 | 0.186 | 0.586 | -0.002 |
| VisDrone val, vehicle mAP50 | 0.810 | 0.557 | 0.810 | -0.001 |

Rehearsal fixed the forgetting: VisDrone accuracy is unchanged, and real-flood accuracy
rose 23 %, mostly through recall. **`models/flood_rehearsal.pt` is the detector used by
the pipeline.**

Caveats: 109 test photos (one scene appears twice), person labels only, so flood vehicles
are not measured. The VisDrone val number also took part in epoch selection, so it is
very slightly optimistic. Figures: `outputs/phase2b/figures/` (training curves,
Phase 1 / v1 / rehearsal side by side).

## Deliverables → where they go

| Artifact | Local path | Used by |
|---|---|---|
| Fine-tuned weights | `models/flood_finetuned.pt` | **Phase 3** (SAHI runs on this) |
| Comparison | `models/phase2_flood_metrics.json` | **Phase 9** (comparison #1) |
| Dataset stats | `outputs/phase2/flood_c2a_stats.json` | Report: dataset section, pseudo-label counts |
| Side-by-side figure | `outputs/phase2/base_vs_finetuned_test.jpg` | Report figure |

## If Colab disconnects

Checkpoints write to Drive every 5 epochs. Re-run cells 1–8, then **9b**
(`--resume`) instead of cell 9. Once `models/flood_finetuned.pt` exists, cell 9
refuses to run, so use **9c** to regenerate the comparison.

## Citation

> R. A. Nihal, B. Yen, K. Itoyama, and K. Nakadai, "UAV-Enhanced Combination to
> Application: benchmarking of a human detection dataset for disaster scenarios,"
> ICPR 2024. Flood subset via Roboflow Universe (`wsar/flood-c2a-8vj8c`), CC BY 4.0.
