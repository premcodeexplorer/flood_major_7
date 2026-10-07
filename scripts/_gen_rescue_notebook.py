"""Generates notebooks/Phase3_Rescue_Pipeline_Colab.ipynb - the full pipeline on a GPU.

Embeds scripts/rescue/*.py and the run/eval scripts via %%writefile, so the
notebook is self-contained. Re-run after editing any of them:
    python scripts/_gen_rescue_notebook.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
S = ROOT / "scripts"
PKG = ["__init__.py", "detect.py", "track.py", "motion.py", "severity.py", "memory.py",
       "risk.py", "render.py", "pipeline.py"]
SCRIPTS = ["run_rescue.py", "eval_sahi.py", "eval_priority.py", "make_demo_video.py",
           "_test_rescue.py", "visdrone_to_yolo.py"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def writefile(path, text):
    return code(f"%%writefile {path}\n{text}")


cells = [
md("""# Phase 3-5 - Rescue pipeline (Model 1 + Model 2 + tracking + memory + priority)

```
frame -> Model 1 detector -> ByteTrack -> camera-motion compensation
      -> group people -> memory (time stranded, water trend)
      -> Model 2 flood map (severity, isolation) -> risk 0-1 -> Critical/High/Medium/Low
      -> annotated video, priority map, victims.csv, alerts.json
```

Nothing is trained here. Needs on Drive (`MyDrive/FloodRescue/`):

| file | from |
|---|---|
| `models/flood_rehearsal.pt` | Phase 2b (already there) |
| `models/flood_severity_unet_scse.pt` | **upload** `outputs/model2/runs/unet_r50_scse_hybrid/best.pt` from the repo, renamed |
| `data/flood_c2a_coco.zip` | Phase 2 (already there) |
| `videos/*.mp4` | **upload** 1-3 real aerial flood clips (see section 8) |

`Runtime > Change runtime type > GPU` (T4 is fine). Skip section 6 (optional).
Section 8 runs every video in `videos/` (roughly 8-12 processed fps on a T4).
Full checklist: `docs/FINAL_DEMO_PLAN.md`."""),

md("## 1. GPU"),
code("!nvidia-smi"),

md("## 2. Drive and paths"),
code("""from google.colab import drive
drive.mount('/content/drive')
import os, glob
PROJECT = '/content/drive/MyDrive/FloodRescue'
MODELS, OUT = f'{PROJECT}/models', f'{PROJECT}/outputs/rescue'
DET = f'{MODELS}/flood_rehearsal.pt'
SEV = f'{MODELS}/flood_severity_unet_scse.pt'
C2A_ZIP = f'{PROJECT}/data/flood_c2a_coco.zip'
VIDEOS = f'{PROJECT}/videos'
os.makedirs(OUT, exist_ok=True); os.makedirs(VIDEOS, exist_ok=True)
for p in [DET, SEV, C2A_ZIP]:
    assert os.path.exists(p), f'missing {p}'
print('ok; videos on Drive:', [os.path.basename(v) for v in glob.glob(f'{VIDEOS}/*')])"""),

md("## 3. Install"),
code("""!pip -q install "ultralytics>=8.3.0" "sahi>=0.11.20" "segmentation-models-pytorch>=0.3.3" pycocotools lapx
import ultralytics, sahi, segmentation_models_pytorch as smp, torch
print(ultralytics.__version__, sahi.__version__, smp.__version__, torch.cuda.get_device_name(0))"""),

md("## 4. Write the code, run the logic tests"),
code("!mkdir -p /content/rescue"),
]
cells += [writefile(f"/content/rescue/{f}", (S / "rescue" / f).read_text()) for f in PKG]
cells += [writefile(f"/content/{f}", (S / f).read_text()) for f in SCRIPTS]
cells += [
code("%cd /content\n!python _test_rescue.py"),

md("## 5. Flood test photos (used by sections 7 and 10)"),
code("""SRC = '/content/datasets/flood_c2a_coco'
if not os.path.exists(f'{SRC}/test/_annotations.coco.json'):
    os.makedirs(SRC, exist_ok=True)
    !unzip -q "{C2A_ZIP}" "test/*" -d {SRC}
print(len(glob.glob(f'{SRC}/test/*.jpg')), 'test photos')"""),

md("""## 6. (OPTIONAL - skip it) SAHI on VisDrone val

**SAHI is dropped from the system**: on the 109 flood test photos it lowered AP50 from 0.404
to 0.326 (`outputs/rescue/sahi_eval_flood_cpu.md`). This only checks whether it would help
on high-resolution images (VisDrone val, 1-2K px). Nice for the discussion, not required."""),
code("""VD = '/content/datasets/VisDrone'
if not os.path.isdir(f'{VD}/VisDrone2019-DET-val'):
    os.makedirs(VD, exist_ok=True)
    !curl -sL -o /content/vdval.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip
    !unzip -q /content/vdval.zip -d {VD} && rm /content/vdval.zip
if not os.path.isdir(f'{VD}/VisDrone2019-DET-val/labels'):
    import sys; from pathlib import Path; sys.path.insert(0, '/content')
    from visdrone_to_yolo import convert_split            # same converter as Phase 1, val only
    print(convert_split(Path(f'{VD}/VisDrone2019-DET-val'))['per_class'])
print(len(glob.glob(f'{VD}/VisDrone2019-DET-val/labels/*.txt')), 'label files')
!python eval_sahi.py --images {VD}/VisDrone2019-DET-val/images \\
    --yolo-labels {VD}/VisDrone2019-DET-val/labels \\
    --weights {DET} --slices 512 640 --out {OUT}/sahi_eval_visdrone.json"""),

md("""## 7. Single photos - every flood test photo through the pipeline

No tracking or memory (one photo has no history). Writes `annotated/`, `victims.csv`."""),
code("""!python run_rescue.py --source {SRC}/test --out {OUT}/test_photos \\
    --detector {DET} --severity {SEV} | tail -25
from IPython.display import Image as IPImage, display
for p in sorted(glob.glob(f'{OUT}/test_photos/annotated/*.jpg'))[:4]:
    display(IPImage(p, width=900))"""),

md("""## 8. Video - the full system on EVERY clip in `videos/`

**Upload real aerial flood footage** to `MyDrive/FloodRescue/videos/` first - see
`docs/FINAL_DEMO_PLAN.md` for what makes a good clip. Top-down views work best (Model 2 was
trained on top-down FloodNet imagery).

No video yet? Falls back to a **simulated** drone pass over a test photo - label anything
from it "simulated camera motion over a still image" (nothing moves in it)."""),
code("""vids = sorted(glob.glob(f'{VIDEOS}/*.mp4') + glob.glob(f'{VIDEOS}/*.mov') + glob.glob(f'{VIDEOS}/*.MP4'))
if not vids:
    STILL = glob.glob(f'{SRC}/test/400a88b7*.jpg')[0]      # flooded plaza, ~50 people
    !python make_demo_video.py --image "{STILL}" \
        --out /content/simulated_pass.mp4 --seconds 30 --fps 10 --size 640 360 --view 0.95 --rotate 2
    vids = ['/content/simulated_pass.mp4']
NAMES = []
for VIDEO in vids:
    NAME = os.path.splitext(os.path.basename(VIDEO))[0]; NAMES.append(NAME)
    print('=' * 20, NAME)
    # --stride 3: 30 fps footage processed at 10 fps - plenty for people who barely move
    !python run_rescue.py --source "{VIDEO}" --out "{OUT}/{NAME}/full" --detector {DET} --severity {SEV} \
        --stride 3 | grep -E "video|ALERT|fps_processed|unique_tracks|final_groups|done"
print(NAMES)"""),

md("### 8b. Watch them"),
code("""import base64, json, pandas as pd
from IPython.display import HTML, display
def h264(mp4):
    out = mp4.replace('.mp4', '_h264.mp4')
    if not os.path.exists(out):
        !ffmpeg -y -loglevel error -i "{mp4}" -vcodec libx264 -pix_fmt yuv420p -crf 23 "{out}"
    return out
def show(mp4, width=1000):
    b = base64.b64encode(open(h264(mp4), 'rb').read()).decode()
    return HTML(f'<video width={width} controls><source src="data:video/mp4;base64,{b}" type="video/mp4"></video>')
for NAME in NAMES:
    print('=' * 20, NAME)
    display(show(f'{OUT}/{NAME}/full/annotated.mp4'))
    display(pd.read_csv(f'{OUT}/{NAME}/full/victims.csv').head(8))
    print(json.dumps(json.load(open(f'{OUT}/{NAME}/full/alerts.json'))[:5], indent=1))"""),

md("""## 9. Ablations (planned comparisons 3 and 4) on the main clip

Same clip, one module off at a time: does the ranking / do the alerts change? Pick the clip
with the most people as `MAIN`."""),
code("""MAIN = NAMES[0]          # <- change to the best clip's name
VIDEO = [v for v in vids if os.path.splitext(os.path.basename(v))[0] == MAIN][0]
for tag, flags in [('no_memory', '--no-memory'), ('no_severity', '--no-severity'),
                  ('no_motion', '--no-motion')]:
    print('=====', tag)
    !python run_rescue.py --source "{VIDEO}" --out "{OUT}/{MAIN}/{tag}" --detector {DET} --severity {SEV} \
        --stride 3 {flags} --no-panel | grep -E "fps_processed|unique_tracks|alerts|final_groups"
rows = []
for tag in ['full', 'no_memory', 'no_severity', 'no_motion']:
    p = f'{OUT}/{MAIN}/{tag}'
    s = json.load(open(f'{p}/summary.json'))
    v = pd.read_csv(f'{p}/victims.csv') if os.path.getsize(f'{p}/victims.csv') else pd.DataFrame()
    rows.append(dict(variant=tag, fps=s['fps_processed'], tracks=s.get('unique_tracks'),
                     alerts=s['alerts'], groups=s['final_groups'],
                     top_group=None if v.empty else f"{v.iloc[0]['group']} ({v.iloc[0]['people']} ppl)",
                     top_risk=None if v.empty else v.iloc[0]['risk'], **s['final_levels']))
abl = pd.DataFrame(rows); display(abl)
abl.to_csv(f'{OUT}/{MAIN}/ablation.csv', index=False)"""),

md("""## 10. Presentation pack - keyframes, tables, videos in one zip

Saves 4 keyframes per clip (PNG, for slides), the H.264 videos (play in PowerPoint), the
victim tables, alerts and the ablation table, then downloads `presentation_pack.zip`."""),
code("""import cv2, shutil
PACK = '/content/presentation_pack'
shutil.rmtree(PACK, ignore_errors=True); os.makedirs(PACK)
for NAME in NAMES:
    d = f'{OUT}/{NAME}/full'
    shutil.copy(h264(f'{d}/annotated.mp4'), f'{PACK}/{NAME}_demo.mp4')
    for f in ['victims.csv', 'alerts.json', 'summary.json']:
        shutil.copy(f'{d}/{f}', f'{PACK}/{NAME}_{f}')
    cap = cv2.VideoCapture(f'{d}/annotated.mp4'); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k, q in enumerate([0.1, 0.4, 0.7, 0.98]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(q * (n - 1))); ok, fr = cap.read()
        if ok: cv2.imwrite(f'{PACK}/{NAME}_frame{k + 1}.png', fr)
if os.path.exists(f'{OUT}/{MAIN}/ablation.csv'):
    shutil.copy(f'{OUT}/{MAIN}/ablation.csv', f'{PACK}/{MAIN}_ablation.csv')
for p in sorted(glob.glob(f'{OUT}/test_photos/annotated/*.jpg'))[:12]:
    os.makedirs(f'{PACK}/photos', exist_ok=True); shutil.copy(p, f'{PACK}/photos/')
shutil.make_archive('/content/presentation_pack', 'zip', PACK)
shutil.copy('/content/presentation_pack.zip', f'{OUT}/presentation_pack.zip')
print(sorted(os.listdir(PACK)))
from google.colab import files
files.download('/content/presentation_pack.zip')"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
out = ROOT / "notebooks" / "Phase3_Rescue_Pipeline_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
