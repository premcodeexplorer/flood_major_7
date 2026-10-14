# Handoff: what's left before the presentation

Written 2026-10-14. Presentation in ~2 days. **Code and both models are finished.**
What remains is running the pipeline on real footage, collecting the numbers it produces,
and updating the slides. Do the steps in order; step 1 blocks almost everything else.

---

## 0. What you are receiving

| Where | What |
|---|---|
| **GitHub repo** (this folder) | All code, notebooks, docs, figures, result tables |
| **Google Drive folder `FloodRescue/`** (shared separately; built from `SEND_VIA_DRIVE/` on Prem's PC) | The trained weights + test data. They are too big for GitHub (Model 2 = 135 MB, limit 100 MB) |

Drive folder contents (the Colab notebook expects exactly this layout in **My Drive**):
```
My Drive/FloodRescue/models/flood_rehearsal.pt            (Model 1, 22 MB)
My Drive/FloodRescue/models/flood_severity_unet_scse.pt   (Model 2, 135 MB)
My Drive/FloodRescue/data/flood_c2a_coco.zip              (flood test photos)
My Drive/FloodRescue/videos/SOURCES.txt                   (clip sources, fill in)
My Drive/FloodRescue/extras/simulated_video.mp4           (fallback demo, label as "simulated")
```
If it shows up under **"Shared with me"**, right-click it, then **Organize > Add shortcut > My Drive**,
so it appears as `My Drive/FloodRescue`. If the notebook still can't find it, download the
folder and upload it into your own My Drive.

### Results already done (use these in the slides)

| Part | Result | Source file |
|---|---|---|
| Model 1: victim detector (YOLOv8, VisDrone → flood, rehearsal fine-tune) | Flood mAP50 **0.324 → 0.399 (+23%)**, forgetting fixed | `models/flood_rehearsal_metrics.json`, `docs/PHASE2_README.md` |
| Model 2: flood segmentation (U-Net R50 + scSE + Dice/Focal, FloodNet) | **75.67 mIoU**, beats our DeepLabV3+ / PSPNet / U-Net | `outputs/model2/comparison.md`, `ablation.md`, `per_class.png` |
| SAHI test | Dropped: AP50 0.404 → 0.326, ~5× slower | `outputs/rescue/sahi_eval_flood_cpu.md` |
| Integrated pipeline (detect → track → memory → severity → risk → ranking) | Built and tested | `docs/RESCUE_PIPELINE_README.md` |
| Example outputs | Annotated photos + simulated video | `outputs/rescue/local_demo/` |

Full plan with the reasoning behind each step: `docs/FINAL_DEMO_PLAN.md`.

---

## 1. Find 1–3 real flood drone clips (most important, ~1–2 h)

Without a real clip the demo only shows a simulated pan over a still photo.

- **Aerial drone view of a real flood**, ideally **top-down**. People **clearly visible**
  (on roofs, wading, in boats), at least 25–30 px tall in 1080p.
- **Steady** (hover or slow pan), **20–60 s**, **one continuous shot**, **1080p MP4**. Not 4K.
- Avoid: news captions, cuts, night footage, clips with no people.
- Sources: **Pexels**, **Pixabay** ("flood drone", "flooded city aerial", "flood rescue aerial").
  YouTube is fine **with a citation**.
- Put clips in `My Drive/FloodRescue/videos/` and fill in `videos/SOURCES.txt`
  (filename | site | title | URL | date | trimmed to).
- Trim a long clip (Colab cell or local):
  `ffmpeg -ss 00:00:12 -to 00:00:52 -i raw.mp4 -vf scale=-2:1080 -c:v libx264 -crf 20 -an clip1.mp4`

- [ ] At least one clip in Drive, source written down

## 2. Run the Colab notebook (~30–60 min)

1. colab.research.google.com → **File > Upload notebook** → `notebooks/Phase3_Rescue_Pipeline_Colab.ipynb`
2. **Runtime > Change runtime type > T4 GPU**
3. Run the cells top to bottom:

| Section | Expect |
|---|---|
| 1. GPU | Tesla T4 table |
| 2. Drive | Allow access; must print `ok; videos on Drive: [...]` |
| 3. Install | 1–2 min, ignore pip warnings |
| 4. Code + tests | Tests pass |
| 5. Test photos | `109 test photos` |
| **6. SAHI** | **SKIP. Do not run.** |
| 7. Photos | 4 annotated photos |
| 8. Videos | A few minutes per clip |
| 8b. Watch | **Watch every output** and pick the best clip |
| 9. Ablations | Set `MAIN = '<best clip name>'` on its first line, then run |
| 10. Pack | Downloads `presentation_pack.zip` (copy also in `My Drive/FloodRescue/outputs/rescue/`) |

If it breaks: disconnected → rerun from section 1. "No GPU" → quota used up, try later or use
another Google account. Empty video (no boxes) → people too small or not aerial, use another clip.

- [ ] `presentation_pack.zip` downloaded
- [ ] Main clip chosen, section 9 ablation table saved

## 3. Rater labels (~15 min per person, 3 people)

- Instructions: `outputs/rescue/priority_eval/README_RATERS.md`
- 40 photos in `outputs/rescue/priority_eval/sheet/`. For each, pick the numbered group you'd
  rescue first (`0` if no box contains a real person). Do this **independently**.
- Fill `rater1`, `rater2`, `rater3` in `outputs/rescue/priority_eval/labels.csv`, save as CSV.
- Then run: `python scripts/eval_priority.py score` → writes `priority_eval.json`.

If time runs out, skip this step and its slide. The demo video matters more.

- [ ] `labels.csv` all 3 columns filled
- [ ] `priority_eval.json` produced

## 4. Fill the results tables

**Pipeline ablation, main clip** (from `ablation.csv` in the pack)

| Variant | Groups | Alerts | Top group | Top risk | Critical / High / Medium / Low |
|---|---|---|---|---|---|
| Full system | | | | | |
| No memory | | | | | |
| No flood severity | | | | | |
| No camera-motion compensation | | | | | |

**Priority vs human raters** (from `priority_eval.json`)

| Method | Top-1 agreement with rater majority |
|---|---|
| Random pick | |
| People count only (no Model 2) | |
| **Full system** | |
| Rater vs rater (ceiling) | |

Report whatever comes out, even if "people count only" wins.

**Speed:** processed fps on T4 (`summary.json` in the pack): ______

## 5. Update the slide deck (`FloodRescue_Major_Project_PPT_1st.pptx`)

Fixes to existing slides:
- [ ] Slide 11 Tech stack: remove "+ SAHI" (say "SAHI tested, not adopted")
- [ ] Slides 4 and 12: "humans, vehicles and animals" → "persons and vehicles". There is no
      animal training data, so list it as a limitation.
- [ ] Slide 10 Timeline: show what is actually done, not the old plan
- [ ] Architecture diagram (`Pipeline_Architecture (1).png`): remove "+ SAHI"

New slides to add:
- [ ] **Model 1 results**: base vs fine-tuned vs rehearsal table, +23%; one line on SAHI
      (figure: `outputs/phase2b/figures/phase1_v1_rehearsal_test.png`)
- [ ] **Model 2 results**: comparison table + per-class chart (`outputs/model2/per_class.png`,
      `docs/_brief_figs/`)
- [ ] **How it works**: architecture + risk-factor table (`docs/RESCUE_PIPELINE_README.md`)
- [ ] **Live demo**: embed the `_demo.mp4` from the pack (H.264, plays in PowerPoint) + 4 keyframes
- [ ] **Annotated photos**: `outputs/rescue/local_demo/photos/street.jpg`, `plaza.jpg`
- [ ] **What each module adds**: ablation table (step 4)
- [ ] **Does it rank like a human?**: rater table (step 4), if done
- [ ] **Limitations**: pixels not GPS; water extent not depth; best on top-down imagery;
      no animal data; no labelled flood video for tracking metrics (MOTA/IDF1)

## 6. Dry run

- [ ] Timed run-through. Demo script (~3 min) is in `docs/FINAL_DEMO_PLAN.md` §6
- [ ] Backup: keyframe PNGs in the deck in case the video won't play; pack on a USB stick + Drive
- [ ] Viva prep: "Design decisions" section of `docs/RESCUE_PIPELINE_README.md`,
      plus `docs/EVALUATION_METRICS.md` and `FloodRescue_Technical_QA.pdf`

---

## Running locally (optional, CPU)

```bash
pip install -r requirements.txt
python scripts/_test_rescue.py
# quick check of a clip: copy both .pt files from Drive into models/ first
python scripts/run_rescue.py --source clip1.mp4 --out outputs/rescue/check --stride 10 --max-frames 15 \
    --severity models/flood_severity_unet_scse.pt
```
