"""Generates notebooks/Phase2_FloodC2A_Colab.ipynb from the Phase-2 scripts.

The notebook embeds prepare_flood_c2a.py and train_flood.py (plus the two
Phase-1 scripts they import from) via %%writefile, so the .py files in
scripts/ stay the single source of truth. Re-run this after editing any of
them:   python scripts/_gen_phase2_notebook.py
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
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s}


cells = [
md("""# Phase 2 - Fine-tune Model 1 on Flood C2A

**Flood Victim Detection & Dynamic Rescue Priority Mapping**

Starts from the Phase-1 VisDrone model (`models/visdrone_base.pt` on Drive) and
fine-tunes it on the flood subset of **C2A** (Nihal et al., ICPR 2024 - ref [3]):
aerial flood scenes with people composited in. Then scores **both** models on the
same test data, which is the "fine-tuning helps" result for Phase 9.

Run every cell top to bottom **once**. Expected total: **~2 h on T4**, **~45 min on A100**.

**Coming back to a finished run?** Run cells 1-8, then **9c**. Never re-run cell 9
once training has finished.

Before you start: `Runtime > Change runtime type > GPU`.

| split | images | what it is | used for |
|---|---|---|---|
| train | 4,158 | 2,079 aerial flood scenes x 2 augmented copies | training |
| val | 541 | same backgrounds as train, different pasted people | picking the best epoch only |
| test | 109 | **real** flood/landslide photos, people labelled | **the reported comparison** |"""),

md("## 1. Check the GPU"),
code("!nvidia-smi"),

md("## 2. Mount Google Drive"),
code("""from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT = '/content/drive/MyDrive/FloodRescue'
RUNS    = f'{PROJECT}/runs'
MODELS  = f'{PROJECT}/models'
OUT     = f'{PROJECT}/outputs/phase2'
os.makedirs(RUNS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

BASE = f'{MODELS}/visdrone_base.pt'
assert os.path.exists(BASE), f'{BASE} missing - Phase 1 must have finished (or upload visdrone_base.pt there)'
print('base model:', BASE)"""),

md("## 3. Install dependencies"),
code("""!pip -q install "ultralytics>=8.3.0"
import ultralytics, torch
ultralytics.checks()
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"""),

md("""## 4. Unzip Flood C2A from Drive

Upload `flood_c2a_coco.zip` (made locally from the Roboflow export: `train/ valid/ test/`
at the top level) to **`MyDrive/FloodRescue/data/`** before running this.
It is unzipped to local Colab disk, not used from Drive - Drive I/O would
throttle every training epoch."""),
code("""import glob
SRC = '/content/datasets/flood_c2a_coco'
ZIP = f'{PROJECT}/data/flood_c2a_coco.zip'

if not os.path.exists(f'{SRC}/train/_annotations.coco.json'):
    assert os.path.exists(ZIP), f'{ZIP} not found - upload flood_c2a_coco.zip to MyDrive/FloodRescue/data/'
    os.makedirs(SRC, exist_ok=True)
    !unzip -q "{ZIP}" -d {SRC}

for split in ['train', 'valid', 'test']:
    n = len(glob.glob(f'{SRC}/{split}/*.jpg'))
    print(f'{split:6s} {n} images')
# expected: train 20790 | valid 541 | test 109"""),

md("## 5. Write the scripts"),
code("%%writefile /content/visdrone_to_yolo.py\n" + conv_vd),
code("%%writefile /content/train_visdrone.py\n" + train_vd),
code("%%writefile /content/prepare_flood_c2a.py\n" + prep),
code("%%writefile /content/train_flood.py\n" + train),

md("""## 6. Convert to YOLO + pseudo-label vehicles

* keeps 2 of the 10 augmented copies of each train image (Ultralytics augments on the fly)
* maps every C2A label to `person`
* C2A labels **no vehicles**. Left like that, fine-tuning would teach the model that
  cars are background. So the Phase-1 model's confident (>=0.5) vehicle detections
  are added to **train/val** as pseudo-labels. **Never to test** - scoring a model
  against its own predictions would be circular.

~3 min for conversion + ~3 min for pseudo-labelling on a T4."""),
code("""DATA = '/content/datasets/flood_c2a'
!python /content/prepare_flood_c2a.py \\
    --src {SRC} \\
    --out {DATA} \\
    --vehicle-weights {BASE} \\
    --vehicle-conf 0.5 \\
    --stats-out {OUT}/flood_c2a_stats.json

print()
!cat {DATA}/flood_c2a.yaml"""),

md("""### 6b. Sanity check - overlay converted labels

Green = person (C2A labels), orange = vehicle (pseudo-labels). If green boxes do
not sit on people, or orange boxes are mostly on non-vehicles, **stop** - lower or
raise `--vehicle-conf` in cell 6 and re-run it."""),
code("""import random, cv2, matplotlib.pyplot as plt

COLORS = [(0, 255, 0), (255, 160, 0), (255, 0, 255)]

def show(split, k=3, seed=None):
    rng = random.Random(seed)
    imgs = rng.sample(sorted(glob.glob(f'{DATA}/{split}/images/*.jpg')), k)
    fig, axes = plt.subplots(1, k, figsize=(22, 7.5))
    for ax, p in zip(axes, imgs):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        n = [0, 0, 0]
        for line in open(p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'):
            c, xc, yc, w, h = line.split()
            c = int(c); xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
            cv2.rectangle(img, (int((xc - w / 2) * W), int((yc - h / 2) * H)),
                          (int((xc + w / 2) * W), int((yc + h / 2) * H)), COLORS[c], 2)
            n[c] += 1
        ax.imshow(img); ax.set_title(f'{split}: {n[0]} person, {n[1]} vehicle'); ax.axis('off')
    plt.tight_layout(); plt.show()

show('train', seed=0)
show('test', seed=0)"""),

md("""## 7. VisDrone val - for the forgetting check

Fine-tuning on flood data can erode what Phase 1 learned (especially vehicles,
which C2A only has as pseudo-labels). Both models are scored on VisDrone val too.
Only the val split is downloaded (~80 MB)."""),
code("""import sys
from pathlib import Path
VD = '/content/datasets/VisDrone'
VD_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0'
os.makedirs(VD, exist_ok=True)
if not os.path.isdir(f'{VD}/VisDrone2019-DET-val'):
    !curl -L -o /content/VisDrone2019-DET-val.zip {VD_URL}/VisDrone2019-DET-val.zip
    !unzip -q /content/VisDrone2019-DET-val.zip -d {VD}
    !rm /content/VisDrone2019-DET-val.zip

sys.path.insert(0, '/content')
from visdrone_to_yolo import convert_split
s = convert_split(Path(f'{VD}/VisDrone2019-DET-val'))
print('VisDrone val:', s['images'], 'images', s['per_class'])

VD_YAML = f'{VD}/visdrone_val.yaml'
Path(VD_YAML).write_text(
    f'path: {VD}\\n'
    'train: VisDrone2019-DET-val/images   # unused - val only\\n'
    'val: VisDrone2019-DET-val/images\\n'
    'nc: 3\\nnames:\\n  0: person\\n  1: vehicle\\n  2: animal\\n')"""),

md("""## 8. Expected time

Same auto-picked resolution as Phase 1 (960 on T4, 1280 on A100), so both models
are compared at the resolution the base was trained at.

| GPU | EPOCHS | PATIENCE | ~time/epoch | ~total incl. comparison |
|---|---|---|---|---|
| A100 | 30 | 8 | ~1 min | ~45 min |
| L4 | 30 | 8 | ~2 min | ~1.2 h |
| **T4** | **30** | **8** | **~3 min** | **~1.8 h** |

Order-of-magnitude only. Leave the tab open: a free Colab session is reclaimed
after ~90 min of no interaction."""),

md("""## 9. Fine-tune + compare

`lr0=0.001` - 10x below Phase 1, so the VisDrone knowledge is adjusted, not
overwritten. When training ends the script scores **base vs fine-tuned** on the
flood test split and VisDrone val, and writes `phase2_flood_metrics.json`.

> **Run this cell exactly once.** Disconnected mid-training -> **9b**.
> Training already done -> **9c**."""),
code("""EPOCHS   = 30
PATIENCE = 8

if os.path.exists(f'{MODELS}/flood_finetuned.pt'):
    raise SystemExit('Training already finished (models/flood_finetuned.pt exists). '
                     'Use 9b to resume a disconnect or 9c to regenerate the comparison.')

!python /content/train_flood.py \\
    --data {DATA}/flood_c2a.yaml \\
    --weights {BASE} \\
    --visdrone-data {VD_YAML} \\
    --project {RUNS} \\
    --name flood_finetune \\
    --epochs {EPOCHS} \\
    --patience {PATIENCE} \\
    --export-to {MODELS}"""),

md("""### 9b. If Colab disconnected - resume

Re-run cells 1-8 (download and convert skip or rebuild quickly), then run this
**instead of** cell 9."""),
code("""!python /content/train_flood.py \\
    --data {DATA}/flood_c2a.yaml \\
    --weights {BASE} \\
    --visdrone-data {VD_YAML} \\
    --project {RUNS} \\
    --name flood_finetune \\
    --resume \\
    --export-to {MODELS}"""),

md("""### 9c. Training already finished - rerun the comparison only

Scores `models/flood_finetuned.pt` against `models/visdrone_base.pt` again and
rewrites `phase2_flood_metrics.json`. ~3 min."""),
code("""!python /content/train_flood.py \\
    --data {DATA}/flood_c2a.yaml \\
    --weights {BASE} \\
    --visdrone-data {VD_YAML} \\
    --project {RUNS} \\
    --name flood_finetune \\
    --val-only \\
    --finetuned {MODELS}/flood_finetuned.pt \\
    --export-to {MODELS}"""),

md("## 10. Results - the numbers for your report"),
code("""import json
from IPython.display import Image as IPImage, display

r = json.load(open(f'{MODELS}/phase2_flood_metrics.json'))

def row(name, d, key):
    b, f, dl = d['base'][key], d['finetuned'][key], d['delta'][key]
    print(f'{name:28s} {b:8.3f} {f:11.3f} {dl:+8.3f}')

print(f'{"":28s} {"base":>8s} {"fine-tuned":>11s} {"delta":>8s}')
for k in ['mAP50', 'mAP50-95', 'precision', 'recall']:
    row(f'flood test   {k}', r['flood_test'], k)
if r['visdrone_val']:
    for k in ['mAP50', 'mAP50-95']:
        row(f'VisDrone val {k}', r['visdrone_val'], k)
    v = r['visdrone_val']
    for c in ['person', 'vehicle']:
        print(f'VisDrone val {c:8s} mAP50-95  {v["base"]["per_class_mAP50-95"][c]:8.3f} '
              f'{v["finetuned"]["per_class_mAP50-95"][c]:11.3f} '
              f'{v["delta"]["per_class_mAP50-95"][c]:+8.3f}')

for png in ['flood_finetune/results.png',
            'flood_finetune_test_base/BoxPR_curve.png',
            'flood_finetune_test_finetuned/BoxPR_curve.png']:
    p = f'{RUNS}/{png}'
    if os.path.exists(p):
        print('\\n' + png)
        display(IPImage(p, width=900))"""),

md("""### How to read this

* **flood test** is person-only (C2A has no vehicle/animal labels), so its mAP is
  the person mAP. `vehicle` and `animal` show `null` - not evaluated, correct.
* **Fine-tuning helped** if flood-test mAP rose. That is the Phase-9 claim.
* **Forgetting** shows as a drop on VisDrone val. A small drop (a few points) is the
  normal price of specialising; a large `vehicle` drop means the pseudo-labels were
  too sparse - re-run cell 6 with a lower `--vehicle-conf` (e.g. 0.35).
* The test photos are real; the training people are pasted in. Any gain here is a
  synthetic -> real transfer, which is the honest framing for the report."""),

md("## 11. Side-by-side on real test photos"),
code("""from ultralytics import YOLO

base_m = YOLO(BASE)
ft_m   = YOLO(f'{MODELS}/flood_finetuned.pt')
picks  = random.Random(1).sample(sorted(glob.glob(f'{DATA}/test/images/*.jpg')), 3)

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
for row_axes, (tag, m) in zip(axes, [('Phase 1 (VisDrone)', base_m), ('Phase 2 (fine-tuned)', ft_m)]):
    for ax, res in zip(row_axes, m.predict(picks, imgsz=r['imgsz'], conf=0.25, verbose=False)):
        ax.imshow(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
        ax.set_title(f'{tag}: {len(res.boxes)} detections'); ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/base_vs_finetuned_test.jpg', dpi=80)
plt.show()"""),

md("""## 12. Download for the local repo

Put `flood_finetuned.pt` into `E:/Flood_major_7/models/` and the JSONs into
`models/` and `outputs/phase2/`. Phase 3 (SAHI) runs on top of this model."""),
code("""from google.colab import files
files.download(f'{MODELS}/flood_finetuned.pt')
files.download(f'{MODELS}/phase2_flood_metrics.json')
files.download(f'{OUT}/flood_c2a_stats.json')
files.download(f'{OUT}/base_vs_finetuned_test.jpg')"""),

md("""---
### Phase 2 deliverables checklist

- [ ] `models/flood_finetuned.pt` - fine-tuned detector (input to Phase 3)
- [ ] `models/phase2_flood_metrics.json` - base vs fine-tuned, flood test + VisDrone val
- [ ] `outputs/phase2/flood_c2a_stats.json` - split sizes, box sizes, pseudo-label counts
- [ ] `outputs/phase2/base_vs_finetuned_test.jpg` - report figure
- [ ] `runs/flood_finetune/results.png` - training curve

Next: **Phase 3** - SAHI sliced inference on top of the fine-tuned model."""),
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

out = ROOT / "notebooks" / "Phase2_FloodC2A_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
