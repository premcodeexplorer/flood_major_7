"""Generates notebooks/Phase1_VisDrone_Colab.ipynb from the Phase-1 scripts.

The notebook embeds visdrone_to_yolo.py and train_visdrone.py via %%writefile,
so the .py files in scripts/ stay the single source of truth. Re-run this after
editing either script:   python scripts/_gen_phase1_notebook.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
conv = (ROOT / "scripts" / "visdrone_to_yolo.py").read_text()
train = (ROOT / "scripts" / "train_visdrone.py").read_text()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s}


cells = [
md("""# Phase 1 - Model 1 Base Training on VisDrone (YOLOv8)

**Flood Victim Detection & Dynamic Rescue Priority Mapping**

Run every cell top to bottom. Expected total time: **~1.5 h on A100**, **~3.5 h on T4**.
Cell 8 asks you to set `EPOCHS` for your GPU - that is the only knob you touch.

Before you start: `Runtime > Change runtime type > GPU` (pick the best GPU you have access to).

Class scheme is `0 person, 1 vehicle, 2 animal`. `animal` is empty in VisDrone but
reserved **on purpose** so the Phase-2 flood fine-tune reuses this exact detection
head instead of reinitializing it."""),

md("## 1. Check the GPU"),
code("!nvidia-smi"),

md("""## 2. Mount Google Drive

All checkpoints go to Drive. If Colab disconnects mid-training you lose nothing -
re-run and use the resume cell (8b)."""),
code("""from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT = '/content/drive/MyDrive/FloodRescue'
RUNS    = f'{PROJECT}/runs'
MODELS  = f'{PROJECT}/models'
os.makedirs(RUNS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
os.makedirs(f'{PROJECT}/outputs/phase1', exist_ok=True)
print(PROJECT)"""),

md("## 3. Install dependencies"),
code("""!pip -q install "ultralytics>=8.3.0"
import ultralytics, torch
ultralytics.checks()
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"""),

md("""## 4. Download VisDrone2019-DET

Pulled from the Ultralytics asset mirror (fast, no Google-Drive quota walls).
train is ~1.4 GB, val ~0.08 GB. Kept on local Colab disk, **not** Drive - it is
regenerable, and Drive I/O would throttle every training epoch."""),
code("""import os
DATA = '/content/datasets/VisDrone'
os.makedirs(DATA, exist_ok=True)

BASE = 'https://github.com/ultralytics/assets/releases/download/v0.0.0'
for zipname in ['VisDrone2019-DET-train.zip', 'VisDrone2019-DET-val.zip']:
    if not os.path.isdir(f'{DATA}/{zipname[:-4]}'):
        !curl -L -o /content/{zipname} {BASE}/{zipname}
        !unzip -q /content/{zipname} -d {DATA}
        !rm /content/{zipname}

!ls {DATA}
!echo "--- train images:" && ls {DATA}/VisDrone2019-DET-train/images | wc -l
!echo "--- val images:"   && ls {DATA}/VisDrone2019-DET-val/images   | wc -l"""),

md("## 5. Write the annotation converter"),
code("%%writefile /content/visdrone_to_yolo.py\n" + conv),

md("""## 6. Convert VisDrone -> YOLO format

No images are copied. Labels land in `<split>/labels/`, exactly where Ultralytics
looks for them. Also generates `visdrone.yaml` and a stats JSON - keep the stats,
the COCO *small-object* percentage is the number that justifies SAHI in Phase 3."""),
code("""!python /content/visdrone_to_yolo.py \\
    --root {DATA} \\
    --yaml-out {DATA}/visdrone.yaml \\
    --stats-out {PROJECT}/outputs/phase1/visdrone_conversion_stats.json

print()
!cat {DATA}/visdrone.yaml"""),

md("""### 6b. Sanity check - overlay a few converted labels

If the boxes do not sit on people/cars here, **stop** and fix the conversion
before burning GPU hours."""),
code("""import random, glob, cv2, matplotlib.pyplot as plt

NAMES  = ['person', 'vehicle', 'animal']
COLORS = [(0, 255, 0), (0, 160, 255), (255, 0, 255)]

imgs = random.sample(sorted(glob.glob(f'{DATA}/VisDrone2019-DET-train/images/*.jpg')), 3)
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, p in zip(axes, imgs):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    lbl = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    n = 0
    for line in open(lbl):
        c, xc, yc, w, h = line.split()
        c = int(c); xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
        x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
        x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], 2)
        n += 1
    ax.imshow(img); ax.set_title(f'{os.path.basename(p)} - {n} boxes'); ax.axis('off')
plt.tight_layout(); plt.show()"""),

md("## 7. Write the training script"),
code("%%writefile /content/train_visdrone.py\n" + train),

md("""## 8. Train

`imgsz` and `batch` are picked automatically from whichever GPU Colab handed you
(1280 on A100, 1024 on L4/V100, 960 on T4). A checkpoint is written to Drive
every 5 epochs, and `patience` stops early once mAP plateaus.

**Set `EPOCHS` below to match your GPU** - cell 1 told you which one you got:

| GPU | EPOCHS | PATIENCE | ~total |
|---|---|---|---|
| A100 | 60 | 15 | ~1.5 h |
| L4 | 50 | 12 | ~3.5 h |
| **T4** | **35** | **10** | **~3.5 h** |

On a T4, 60 epochs would run 8-10 h and will not survive one free Colab session.
35 epochs from COCO-pretrained weights is ample for a *base* model that Phase 2
fine-tunes anyway - over-optimizing Stage 1 is wasted GPU time.

Leave this tab open and the browser awake: a free Colab session is reclaimed
after roughly 90 minutes of no interaction."""),
code("""EPOCHS   = 35   # T4: 35  |  L4: 50  |  A100: 60
PATIENCE = 10   # T4: 10  |  L4: 12  |  A100: 15

!python /content/train_visdrone.py \\
    --data {DATA}/visdrone.yaml \\
    --model yolov8s.pt \\
    --project {RUNS} \\
    --name visdrone_base \\
    --epochs {EPOCHS} \\
    --patience {PATIENCE} \\
    --export-to {MODELS}"""),

md("""### 8b. If Colab disconnected - resume

Re-run cells 1-7 (download and convert are idempotent and skip if already done),
then run this **instead of** cell 8."""),
code("""!python /content/train_visdrone.py \\
    --data {DATA}/visdrone.yaml \\
    --project {RUNS} \\
    --name visdrone_base \\
    --resume \\
    --export-to {MODELS}"""),

md("## 9. Results - the numbers for your report"),
code("""import json
from IPython.display import Image as IPImage, display

m = json.load(open(f'{MODELS}/phase1_visdrone_metrics.json'))
print(json.dumps(m, indent=2))

for png in ['results.png', 'confusion_matrix_normalized.png', 'PR_curve.png',
            'val_batch0_pred.jpg']:
    p = f'{RUNS}/visdrone_base/{png}'
    if os.path.exists(p):
        print('\\n' + png)
        display(IPImage(p, width=900))"""),

md("""### Sanity targets for `yolov8s` on VisDrone (2 merged classes)

| metric | A100 @1280, 60ep | **T4 @960, 35ep** |
|---|---|---|
| mAP@50 | 0.46 - 0.52 | **0.40 - 0.48** |
| mAP@50-95 | 0.27 - 0.31 | **0.23 - 0.28** |
| person mAP@50 | 0.38 - 0.45 | 0.32 - 0.40 |
| vehicle mAP@50 | 0.54 - 0.62 | 0.48 - 0.57 |

The T4 column is 2-4 points lower - that is the cost of 960 px and 35 epochs,
and it is the right trade. Phase 2 fine-tuning and Phase 3 SAHI both move the
number far more than those points would.

`animal` will read **0.0** - correct, VisDrone has no animal instances. Landing
well below this range means something went wrong in conversion, not in training."""),

md("## 10. Inference spot-check on val images"),
code("""from ultralytics import YOLO
import glob, random, matplotlib.pyplot as plt, cv2

model = YOLO(f'{MODELS}/visdrone_base.pt')
picks = random.sample(sorted(glob.glob(f'{DATA}/VisDrone2019-DET-val/images/*.jpg')), 3)
res = model.predict(picks, imgsz=m['imgsz'], conf=0.25, verbose=False)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, r in zip(axes, res):
    ax.imshow(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
    ax.set_title(f'{len(r.boxes)} detections'); ax.axis('off')
plt.tight_layout(); plt.show()"""),

md("""## 11. Download for the local repo

Put `visdrone_base.pt` into `E:/Flood_major_7/models/` - Phase 2 fine-tunes from it."""),
code("""from google.colab import files
files.download(f'{MODELS}/visdrone_base.pt')
files.download(f'{MODELS}/phase1_visdrone_metrics.json')"""),

md("""---
### Phase 1 deliverables checklist

- [ ] `models/visdrone_base.pt` - base detector weights (input to Phase 2)
- [ ] `models/phase1_visdrone_metrics.json` - the VisDrone-only baseline mAP
      (**Phase 9 compares the fine-tuned model against this**)
- [ ] `outputs/phase1/visdrone_conversion_stats.json` - class counts + COCO small/medium/large split
- [ ] `runs/visdrone_base/` - results.png, PR_curve.png, confusion_matrix_normalized.png

Next: **Phase 2** - fine-tune this on the Roboflow flood dataset at a lower LR."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

out = ROOT / "notebooks" / "Phase1_VisDrone_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
