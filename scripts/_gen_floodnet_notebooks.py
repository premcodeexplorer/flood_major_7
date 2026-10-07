"""Generates the three Model-2 (FloodNet segmentation) Colab notebooks.

  notebooks/Model2_FloodNet_1_Prep_Colab.ipynb   - run ONCE by one person
  notebooks/Model2_FloodNet_2_Train_Teammate{1,2}_Colab.ipynb - 2 models each, 4 in total

The scripts are embedded via %%writefile so scripts/*.py stay the single source
of truth. Re-run after editing floodnet_prep.py or floodnet_train.py:
    python scripts/_gen_floodnet_notebooks.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
S = ROOT / "scripts"
prep = (S / "floodnet_prep.py").read_text()
train = (S / "floodnet_train.py").read_text()

import sys  # noqa: E402
sys.path.insert(0, str(S))
from floodnet_train import RUNS  # noqa: E402

DROPBOX = ("https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc"
           "?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1")
ZIP_BYTES = 13057835433


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def notebook(cells, gpu=True):
    nb = {"cells": cells,
          "metadata": {"colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    if gpu:
        nb["metadata"]["accelerator"] = "GPU"
        nb["metadata"]["colab"]["gpuType"] = "T4"
    for c in nb["cells"]:
        lines = c["source"].split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return nb


SHOW_MASKS = """import glob, random, cv2, numpy as np, matplotlib.pyplot as plt
PALETTE = np.array([[0,0,0],[255,0,0],[180,120,120],[160,150,20],[140,140,140],
                    [61,230,250],[0,82,255],[255,0,245],[255,235,0],[4,250,7]], np.uint8)
NAMES = ['background','building-flooded','building-non-flooded','road-flooded','road-non-flooded',
         'water','tree','vehicle','pool','grass']
picks = random.Random(0).sample(sorted(glob.glob(f'{DATA}/train/images/*.jpg')), 3)
fig, axes = plt.subplots(2, 3, figsize=(22, 11))
for k, p in enumerate(picks):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    m = cv2.imread(p.replace('/images/', '/masks/').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
    axes[0, k].imshow(img); axes[0, k].axis('off'); axes[0, k].set_title(p.split('/')[-1])
    axes[1, k].imshow((0.5 * img + 0.5 * PALETTE[m]).astype(np.uint8)); axes[1, k].axis('off')
handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i] / 255) for i in range(10)]
fig.legend(handles, NAMES, loc='lower center', ncol=10)
plt.tight_layout(); plt.show()"""

# =============================================================== notebook 1: prep
prep_cells = [
md(f"""# Model 2 - FloodNet: 1. Download + prepare (run ONCE, by one person)

Downloads the **official FloodNet-Supervised v1.0** release (BinaLab, CDLA-Permissive 1.0,
reference [4]), verifies every image has a mask, resizes everything once to **1024x768**,
and saves a ~1 GB zip to your Drive. Teammates train from that zip with their Teammate notebook.

**No GPU needed** - `Runtime > Change runtime type > CPU` saves your GPU quota.
~30 min total (13 GB download, unzip, resize).

Use the official train/val/test folders exactly as given - never re-split, or the
numbers stop being comparable with the FloodNet paper."""),

md("## 1. Mount Drive"),
code("""from google.colab import drive
drive.mount('/content/drive')
import os
FS = '/content/drive/MyDrive/FloodSeg'
os.makedirs(FS, exist_ok=True)
RAW, DATA = '/content/floodnet_raw', '/content/floodnet_1024'"""),

md(f"""## 2. Download FloodNet-Supervised v1.0 (13.06 GB)

The direct-download form of the Dropbox link in the official README
(github.com/BinaLab/FloodNet-Supervised_v1.0). Goes to local Colab disk."""),
code(f"""ZIP = '/content/FloodNet.zip'
URL = '{DROPBOX}'
if not os.path.isdir(RAW):
    if not os.path.exists(ZIP) or os.path.getsize(ZIP) != {ZIP_BYTES}:
        !wget -q --show-progress -O {{ZIP}} "{{URL}}"
    size = os.path.getsize(ZIP)
    print(f'{{size:,}} bytes')
    assert size == {ZIP_BYTES:_}, 'incomplete download - re-run this cell'"""),

md("## 3. Unzip (including any zips nested inside)"),
code("""import glob, subprocess
if not os.path.isdir(RAW):
    !unzip -q {ZIP} -d {RAW}
    for z in glob.glob(f'{RAW}/**/*.zip', recursive=True):
        print('nested:', z)
        subprocess.run(['unzip', '-q', '-o', z, '-d', os.path.dirname(z)], check=True)
        os.remove(z)
    os.remove(ZIP)      # free 13 GB
!find {RAW} -maxdepth 4 -type d | head -30"""),

md("""## 4. Verify + resize to 1024x768

Fails loudly on a missing mask, an unreadable file, a size mismatch or a class id
outside 0-9. Expected split (official, 2,343 images): **1445 / 450 / 448** - if the
counts differ, tell the team before anyone trains."""),
code("%%writefile /content/floodnet_prep.py\n" + prep),
code("""!python /content/floodnet_prep.py --src {RAW} --out {DATA}
!cp {DATA}/floodnet_stats.json {FS}/"""),

md("""### 4b. Check the masks line up

Bottom row = mask colours over the photo. If colours do not sit on the right
objects (red = flooded building, cyan = water, magenta = vehicle, yellow = pool),
**stop** - something is wrong with the pairing."""),
code(SHOW_MASKS),

md("## 5. Zip to Drive (~1 GB)"),
code("""!cd {DATA} && zip -q -r -0 /content/floodnet_1024.zip .
!cp /content/floodnet_1024.zip {FS}/floodnet_1024.zip
!ls -lh {FS}"""),

md("""## 6. Share with teammates

1. Open Google Drive > `MyDrive/FloodSeg/floodnet_1024.zip`
2. Right-click > **Share** > General access: **Anyone with the link** (Viewer)
3. **Copy link** and send it to everyone - they paste it into `DATA_LINK` in their Teammate notebook.

Also send `floodnet_stats.json` (class pixel shares) to whoever writes the paper -
it is the class-imbalance table."""),
]

# =============================================================== notebook 2: train (one per teammate)
TEAM = {
    1: ["unet_r50", "deeplabv3p_r50"],
    2: ["pspnet_r50", "unet_r50_scse_hybrid"],
}
assert sorted(r for rs in TEAM.values() for r in rs) == sorted(RUNS), "every run assigned exactly once"
DESC = {
    "unet_r50": "U-Net, ResNet-50, cross-entropy - plain version of ours",
    "deeplabv3p_r50": "DeepLabV3+, ResNet-50, cross-entropy - baseline from the FloodNet paper",
    "pspnet_r50": "PSPNet, ResNet-50, cross-entropy - the FloodNet paper's best model",
    "unet_r50_scse_hybrid": "**ours**: U-Net R50 + scSE attention + Dice/Focal loss",
}


def train_cells(n, runs):
    table = "\n".join(f"| {i + 1} | `{r}` | {DESC[r]} |" for i, r in enumerate(runs))
    return [
md(f"""# Model 2 - FloodNet: Teammate {n} - train 2 models

This notebook trains **{runs[0]}** and then **{runs[1]}**, one after the other.
The other teammate's notebook trains the other 2 of the 4 models; together they
fill the comparison table. Every model uses the same data, crop, schedule,
epochs and seed - only the model/loss differs - so the 4 results are fair to compare.

| # | model | what it is |
|---|---|---|
{table}

`Runtime > Change runtime type > GPU (T4)`. Expect **~2-3 h per model**.

**Disconnected, or the session ran out?** Run every cell again from the top.
A finished model is skipped and an unfinished one resumes from its last epoch
(checkpoints are on your Drive). Nothing to change."""),

md("## 1. Settings - the only cell you edit"),
code(f"""DATA_LINK = ''   # @param {{type:"string"}}
# DATA_LINK: the share link of floodnet_1024.zip you were sent.
# Leave empty if floodnet_1024.zip is already in your own MyDrive/FloodSeg/.

RUNS = {json.dumps(runs)}   # fixed - do not change
EPOCHS, SEED = 80, 0        # fixed - must match the other teammate"""),

md("## 2. GPU + Drive"),
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os
FS = '/content/drive/MyDrive/FloodSeg'
os.makedirs(FS, exist_ok=True)
DATA = '/content/floodnet_1024'"""),

md("## 3. Get the prepared dataset onto local disk (~1 min)"),
code("""if not os.path.isdir(f'{DATA}/train/images'):
    ZIP = f'{FS}/floodnet_1024.zip'
    if not os.path.exists(ZIP):
        assert DATA_LINK, 'Set DATA_LINK in cell 1 (or put floodnet_1024.zip in MyDrive/FloodSeg/)'
        ZIP = '/content/floodnet_1024.zip'
        !gdown -q --fuzzy "{DATA_LINK}" -O {ZIP}
    !mkdir -p {DATA} && unzip -q {ZIP} -d {DATA}
for s in ['train', 'val', 'test']:
    print(s, len(os.listdir(f'{DATA}/{s}/images')), 'images')
# expected: train 1445 | val 450 | test 448"""),

md("## 4. Install + write the training script"),
code("""!pip -q install "segmentation-models-pytorch>=0.5" "albumentations>=1.4"
import segmentation_models_pytorch as smp, torch
print('smp', smp.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"""),
code("%%writefile /content/floodnet_train.py\n" + train),

md(f"""## 5. Train both models

Trains `{runs[0]}`, then `{runs[1]}`. Each prints one line per epoch (`sec=` is the
epoch time); validation runs every 2 epochs, and the best checkpoint is scored once
on the **test** split at the end of each model.

Leave the tab open: free Colab reclaims idle sessions after ~90 min."""),
code("""for RUN in RUNS:
    if os.path.exists(f'{FS}/results/{RUN}.json'):
        print(f'[skip] {RUN} already finished')
        continue
    print(f'\\n=============== {RUN} ===============')
    !python /content/floodnet_train.py --run {RUN} --data {DATA} --out {FS} --epochs {EPOCHS} --seed {SEED}"""),

md("## 6. Results"),
code("""import json, matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display
for RUN in RUNS:
    p = f'{FS}/results/{RUN}.json'
    if not os.path.exists(p):
        print(f'{RUN}: not finished yet'); continue
    r = json.load(open(p))
    print(f"\\n{RUN}: test mIoU(9) {r['test']['mIoU9']}  hard-4 {r['hard_class_mIoU']}  "
          f"params {r['params_M']}M  fps {r['fps']} ({r['gpu']})")
    for c, v in r['test']['per_class_IoU'].items():
        print(f'   {c:22s} {v}')
    h = json.load(open(f'{FS}/runs/{RUN}/history.json'))
    ev = [x for x in h if 'val_mIoU9' in x]
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
    ax[0].plot([x['epoch'] + 1 for x in h], [x['loss'] for x in h]); ax[0].set_title(f'{RUN}: train loss')
    ax[1].plot([x['epoch'] + 1 for x in ev], [x['val_mIoU9'] for x in ev]); ax[1].set_title(f'{RUN}: val mIoU (9)')
    plt.show()
    display(IPImage(f'{FS}/runs/{RUN}/test_predictions.jpg', width=1100))  # photo | ground truth | prediction"""),

md("""## 7. Send your results

Downloads 3 small files per model - send all 6 to the person building the table."""),
code("""from google.colab import files
for RUN in RUNS:
    files.download(f'{FS}/results/{RUN}.json')
    files.download(f'{FS}/runs/{RUN}/history.json')
    files.download(f'{FS}/runs/{RUN}/test_predictions.jpg')
# best.pt (~100 MB each) stays on your Drive - only the winning model's is needed later."""),
    ]


notebooks = [("Model2_FloodNet_1_Prep_Colab.ipynb", prep_cells, False)]
notebooks += [(f"Model2_FloodNet_2_Train_Teammate{n}_Colab.ipynb", train_cells(n, runs), True)
              for n, runs in TEAM.items()]
for name, cells, gpu in notebooks:
    out = ROOT / "notebooks" / name
    out.write_text(json.dumps(notebook(cells, gpu), indent=1))
    print(f"wrote {out}  ({len(cells)} cells)")
