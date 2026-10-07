"""Generates notebooks/Phase2b_Rehearsal_Colab.ipynb - Phase 2 re-run with rehearsal.

Why it exists: v1 (C2A only, notebooks/Phase2_FloodC2A_Colab.ipynb) made the
model WORSE - flood test person mAP50 0.324 -> 0.222, VisDrone val 0.699 -> 0.372.
C2A's pasted-in people look unlike real ones (Phase 1 recognises 4 % of them),
so training on them alone overwrote the model's notion of "person"
(catastrophic forgetting). Rehearsal = keep training on VisDrone at the same
time, so real people stay in every batch.

Same scripts as Phase 1/2 (embedded via %%writefile). Re-run after editing them:
    python scripts/_gen_phase2b_notebook.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
S = ROOT / "scripts"
conv_vd = (S / "visdrone_to_yolo.py").read_text()
train_vd = (S / "train_visdrone.py").read_text()
prep = (S / "prepare_flood_c2a.py").read_text()
train = (S / "train_flood.py").read_text()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
md("""# Phase 2b - Flood fine-tune with rehearsal (VisDrone + C2A mixed)

**Why a re-run:** the first fine-tune (v1, C2A only) made the detector worse:

| | Phase 1 | v1 (C2A only) |
|---|---|---|
| flood test, person mAP50 | 0.324 | 0.222 |
| VisDrone val, mAP50 | 0.699 | 0.372 |
| VisDrone val, person mAP50 | 0.588 | 0.186 |

C2A's people are pasted-in cut-outs (often oversized, lying, hard-edged); the Phase-1
model recognises only ~4 % of them. Training on them alone replaced the model's idea of
a "person" - **catastrophic forgetting** across a **synthetic-to-real gap**.

**Fix - rehearsal:** fine-tune on **VisDrone train (6,471 real) + C2A (2,079, one copy per
scene)** together, so real people are in every batch while flood backgrounds are learned.
Validation uses both val sets, so the chosen epoch must stay good on both.

Same comparison as v1: Phase 1 vs this model on the **real flood test photos** and
**VisDrone val**, same resolution - and v1's numbers alongside.

`Runtime > Change runtime type > GPU`. ~1.5 h on a T4. Run every cell once, top to bottom.
**Coming back to a finished run?** Cells 1-9, then **10c**."""),

md("## 1. Check the GPU"),
code("!nvidia-smi"),

md("## 2. Mount Drive"),
code("""from google.colab import drive
drive.mount('/content/drive')
import os
PROJECT = '/content/drive/MyDrive/FloodRescue'
RUNS, MODELS, OUT = f'{PROJECT}/runs', f'{PROJECT}/models', f'{PROJECT}/outputs/phase2'
os.makedirs(OUT, exist_ok=True)
BASE = f'{MODELS}/visdrone_base.pt'
V1 = f'{MODELS}/flood_finetuned.pt'           # the C2A-only run, for the side-by-side
C2A_ZIP = f'{PROJECT}/data/flood_c2a_coco.zip'
for p in [BASE, C2A_ZIP]:
    assert os.path.exists(p), f'missing {p}'
print('ok')"""),

md("## 3. Install"),
code("""!pip -q install "ultralytics>=8.3.0"
import ultralytics, torch
ultralytics.checks()"""),

md("## 4. Flood C2A (from the zip already on your Drive)"),
code("""import glob
SRC = '/content/datasets/flood_c2a_coco'
if not os.path.exists(f'{SRC}/train/_annotations.coco.json'):
    os.makedirs(SRC, exist_ok=True)
    !unzip -q "{C2A_ZIP}" -d {SRC}
for split in ['train', 'valid', 'test']:
    print(f"{split:6s} {len(glob.glob(f'{SRC}/{split}/*.jpg'))} images")
# expected: train 20790 | valid 541 | test 109"""),

md("""## 5. VisDrone train + val (~1.5 GB, local disk)

Same mirror as Phase 1."""),
code("""VD = '/content/datasets/VisDrone'
os.makedirs(VD, exist_ok=True)
VD_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0'
for zipname in ['VisDrone2019-DET-train.zip', 'VisDrone2019-DET-val.zip']:
    if not os.path.isdir(f'{VD}/{zipname[:-4]}'):
        !curl -L -o /content/{zipname} {VD_URL}/{zipname}
        !unzip -q /content/{zipname} -d {VD}
        !rm /content/{zipname}
!echo "train:" && ls {VD}/VisDrone2019-DET-train/images | wc -l
!echo "val:"   && ls {VD}/VisDrone2019-DET-val/images   | wc -l"""),

md("## 6. Write the scripts"),
code("%%writefile /content/visdrone_to_yolo.py\n" + conv_vd),
code("%%writefile /content/train_visdrone.py\n" + train_vd),
code("%%writefile /content/prepare_flood_c2a.py\n" + prep),
code("%%writefile /content/train_flood.py\n" + train),

md("## 7. Convert VisDrone (same converter as Phase 1)"),
code("""!python /content/visdrone_to_yolo.py --root {VD} --yaml-out {VD}/visdrone.yaml \\
    --stats-out /content/visdrone_conversion_stats.json"""),

md("""## 8. Convert C2A - one copy per scene + vehicle pseudo-labels

`--copies-per-source 1` (v1 used 2): 2,079 C2A images next to 6,471 VisDrone images,
so roughly 3 real images for every synthetic one."""),
code("""DATA = '/content/datasets/flood_c2a_r'
!python /content/prepare_flood_c2a.py \\
    --src {SRC} \\
    --out {DATA} \\
    --copies-per-source 1 \\
    --vehicle-weights {BASE} \\
    --vehicle-conf 0.5 \\
    --stats-out {OUT}/flood_c2a_rehearsal_stats.json"""),

md("## 9. The mixed dataset"),
code("""from pathlib import Path
MIX = '/content/datasets/flood_rehearsal.yaml'
Path(MIX).write_text(f'''# Phase 2b rehearsal: VisDrone + C2A mixed
path: /content/datasets
train:
  - VisDrone/VisDrone2019-DET-train/images
  - flood_c2a_r/train/images
val:
  - VisDrone/VisDrone2019-DET-val/images
  - flood_c2a_r/val/images
test: flood_c2a_r/test/images   # real flood photos - the reported comparison
nc: 3
names:
  0: person
  1: vehicle
  2: animal
''')
n = {k: len(glob.glob(p)) for k, p in {
    'VisDrone train': f'{VD}/VisDrone2019-DET-train/images/*.jpg',
    'C2A train':      f'{DATA}/train/images/*.jpg',
    'VisDrone val':   f'{VD}/VisDrone2019-DET-val/images/*.jpg',
    'C2A val':        f'{DATA}/val/images/*.jpg',
    'flood test':     f'{DATA}/test/images/*.jpg'}.items()}
print(n)
print(f"real : synthetic in training = {n['VisDrone train'] / n['C2A train']:.1f} : 1")
# expected: 6471 | 2079 | 548 | 541 | 109  ->  3.1 : 1"""),

md("""## 10. Fine-tune with rehearsal + compare

10 epochs (each ~6-7 min on a T4: 8,550 images), lr0 0.001, last 3 epochs without
mosaic. Writes `models/flood_rehearsal.pt` and `models/flood_rehearsal_metrics.json` -
v1's files are left untouched.

> **Run once.** Disconnected -> **10b**. Already finished -> **10c**."""),
code("""if os.path.exists(f'{MODELS}/flood_rehearsal.pt'):
    raise SystemExit('Already finished (models/flood_rehearsal.pt exists). Use 10b or 10c.')

!python /content/train_flood.py \\
    --data {MIX} \\
    --weights {BASE} \\
    --visdrone-data {VD}/visdrone.yaml \\
    --project {RUNS} \\
    --name flood_rehearsal \\
    --epochs 10 \\
    --patience 4 \\
    --close-mosaic 3 \\
    --export-to {MODELS} \\
    --export-name flood_rehearsal"""),

md("### 10b. Resume after a disconnect (re-run cells 1-9 first)"),
code("""!python /content/train_flood.py \\
    --data {MIX} --weights {BASE} --visdrone-data {VD}/visdrone.yaml \\
    --project {RUNS} --name flood_rehearsal --resume \\
    --export-to {MODELS} --export-name flood_rehearsal"""),

md("### 10c. Finished - rerun the comparison only"),
code("""!python /content/train_flood.py \\
    --data {MIX} --weights {BASE} --visdrone-data {VD}/visdrone.yaml \\
    --project {RUNS} --name flood_rehearsal --val-only \\
    --finetuned {MODELS}/flood_rehearsal.pt \\
    --export-to {MODELS} --export-name flood_rehearsal"""),

md("## 11. Results: Phase 1 vs v1 (C2A only) vs rehearsal"),
code("""import json
r = json.load(open(f'{MODELS}/flood_rehearsal_metrics.json'))
v1p = f'{MODELS}/phase2_flood_metrics.json'
v1 = json.load(open(v1p)) if os.path.exists(v1p) else None

def get(d, block, model, key, cls=None):
    x = d[block][model]
    return x['per_class_mAP50'].get(cls) if cls else x[key]

rows = [('flood test  person mAP50', 'flood_test', 'mAP50', None),
        ('flood test  mAP50-95',     'flood_test', 'mAP50-95', None),
        ('flood test  recall',       'flood_test', 'recall', None),
        ('VisDrone    mAP50',        'visdrone_val', 'mAP50', None),
        ('VisDrone    person mAP50', 'visdrone_val', None, 'person'),
        ('VisDrone    vehicle mAP50','visdrone_val', None, 'vehicle')]
print(f"{'':28s}{'Phase 1':>9s}{'v1 C2A':>9s}{'rehearsal':>11s}")
for name, block, key, cls in rows:
    b = get(r, block, 'base', key, cls)
    f = get(r, block, 'finetuned', key, cls)
    o = get(v1, block, 'finetuned', key, cls) if v1 else float('nan')
    print(f'{name:28s}{b:9.3f}{o:9.3f}{f:11.3f}')

from IPython.display import Image as IPImage, display
p = f'{RUNS}/flood_rehearsal/results.png'
if os.path.exists(p):
    display(IPImage(p, width=900))"""),

md("## 12. Side-by-side on real test photos"),
code("""from ultralytics import YOLO
import random, cv2, matplotlib.pyplot as plt
models = [('Phase 1', BASE), ('v1 C2A only', V1), ('rehearsal', f'{MODELS}/flood_rehearsal.pt')]
models = [(n, p) for n, p in models if os.path.exists(p)]
picks = random.Random(1).sample(sorted(glob.glob(f'{DATA}/test/images/*.jpg')), 3)
fig, axes = plt.subplots(len(models), 3, figsize=(22, 7 * len(models)))
for row, (tag, w) in zip(axes, models):
    for ax, res in zip(row, YOLO(w).predict(picks, imgsz=r['imgsz'], conf=0.25, verbose=False)):
        ax.imshow(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
        ax.set_title(f'{tag}: {len(res.boxes)} detections'); ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/phase1_v1_rehearsal_test.jpg', dpi=70)
plt.show()"""),

md("## 13. Download"),
code("""from google.colab import files
files.download(f'{MODELS}/flood_rehearsal.pt')
files.download(f'{MODELS}/flood_rehearsal_metrics.json')
files.download(f'{OUT}/flood_c2a_rehearsal_stats.json')
files.download(f'{OUT}/phase1_v1_rehearsal_test.jpg')"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
out = ROOT / "notebooks" / "Phase2b_Rehearsal_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
