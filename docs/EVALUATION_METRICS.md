# How This Project Is Evaluated

Reference for Phase 9 and the viva. Covers what each metric means, why
detection cannot use plain accuracy, and which number matters most for rescue.

---

## 1. Why not "accuracy"?

Accuracy is `correct / total`. It is well-defined for **classification** (one
image, one label, fixed denominator). Object detection breaks it four ways:

1. **No fixed number of predictions.** The model chooses how many boxes to emit
   - 3 in one image, 300 in another. There is no denominator.
2. **No true negatives.** Accuracy is `(TP+TN)/(TP+TN+FP+FN)`. "Correctly placed
   no box here" applies to every possible position and scale in the image -
   effectively infinite. TN is undefined, so accuracy is undefined.
3. **Correctness is not binary.** A box can be 80% right. IoU expresses that;
   accuracy cannot.
4. **Background dominance (the decisive one).** If accuracy were forced -
   per-pixel or per-anchor - a model predicting **nothing anywhere** would score
   ~99.9%, because a drone frame is almost entirely empty ground and water. It
   would be useless and look near-perfect.

Point 4 is the clean one-line answer if asked in the viva.

## 2. What is used instead

Neither of these involves true negatives:

| metric | formula | plain English |
|---|---|---|
| **Precision** | `TP / (TP + FP)` | of what I flagged, how much was real |
| **Recall** | `TP / (TP + FN)` | of what was really there, how much did I find |

A prediction counts as **TP** when its **IoU** (intersection over union) with a
ground-truth box clears a threshold, normally 0.50.

Both move as the confidence threshold moves: lower it and recall rises while
precision falls. **mAP** removes that arbitrary choice - it sweeps every
threshold, takes the area under the precision-recall curve per class (= AP),
and averages over classes.

- **mAP@50** - IoU >= 0.50 only. "Did I find it, roughly in the right place?"
- **mAP@50-95** - averaged over IoU 0.50, 0.55 ... 0.95. Also demands tight
  boxes. Always much lower. The headline COCO metric.
- **AP_small / AP_medium / AP_large** - COCO size bands, split at area 32**2 and
  96**2 px.

**mAP is not a percentage of victims found.** It is an area under a curve,
averaged over classes and IoU levels.

## 3. The number that matters most here: recall

In rescue the costs are asymmetric: **a missed victim can cost a life; a false
positive costs a wasted glance.** mAP weights precision and recall equally, so it
understates the system's real usefulness at the low confidence threshold this
project deliberately adopts.

So report **recall at the chosen operating threshold** prominently, next to mAP.
Recall against the 38,758 ground-truth instances in the VisDrone val split is
directly interpretable as "we found this fraction of the objects that were
there" - the intuition "accuracy" was reaching for.

Also report **miss rate = 1 - recall**. For a rescue audience that is the most
honest single figure.

## 4. Per component

| Component | Phase | Metrics | Notes |
|---|---|---|---|
| Model 1 - detector | 1-3 | mAP@50, mAP@50-95, precision, recall, F1 | Ultralytics reports these each epoch |
| SAHI gain | 3 | **AP_small, before vs. after** | The headline experiment - isolates the tiny-victim case SAHI exists to fix |
| Tracking | 4 | MOTA, IDF1, ID switches | Standard MOT metrics; IDF1 measures identity consistency |
| Model 2 - U-Net | 6 | **IoU / mIoU**, Dice | Segmentation, not detection - mAP does not apply |
| Full system | 9 | Ablations: risk score +/- memory, +/- severity | Shows each module earns its place |

## 5. Comparisons that are valid - and one that is not

**Valid:**
- Phase 1 (VisDrone-only) vs. Phase 2 (flood fine-tuned), on the same flood test
  split. This is the "fine-tuning helps" result.
- Plain YOLO vs. YOLO+SAHI, on AP_small. The "small-object handling works"
  result.
- Risk score with vs. without memory / severity. The ablations.

**Not valid:**
- Our VisDrone mAP vs. published VisDrone papers. They evaluate 10 fine-grained
  classes; we merged 8 vehicle types into one. Different task, higher number,
  no conclusion can be drawn. Do not put them side by side.

## 6. Figures to pull for the report

From `runs/visdrone_base/`:

| file | shows |
|---|---|
| `results.png` | loss curves + mAP over epochs |
| `confusion_matrix_normalized.png` | per-class TP/FP/FN - closest thing to the accuracy intuition |
| `PR_curve.png` | the precision-recall tradeoff mAP integrates over |
| `val_batch0_pred.jpg` | qualitative detections |
