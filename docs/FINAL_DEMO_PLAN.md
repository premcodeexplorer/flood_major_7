# Final demo plan (5 days)

**Goal:** a working end-to-end demo on **real flood video**, backed by numbers. All code is
done. What is left is good input (videos), human labels (raters), running the notebook,
and turning the outputs into report and slide material.

SAHI is **dropped** (it lowered AP50 from 0.404 to 0.326 on our flood photos; see
`Project_Master_Document.md` §10). That saves time: skip notebook section 6.

---

## Day-by-day

| Day | Task | Who | Done when |
|---|---|---|---|
| **1** | Find and prepare 2–3 flood video clips (rules below) | 1 teammate | Clips in `MyDrive/FloodRescue/videos/`, sources written in `videos/SOURCES.txt` |
| **1** | Upload Model 2 to Drive: `outputs/model2/runs/unet_r50_scse_hybrid/best.pt` → `MyDrive/FloodRescue/models/flood_severity_unet_scse.pt` | You | Notebook cell 2 prints `ok` |
| **1–2** | Rate the 40 photos (`outputs/rescue/priority_eval/README_RATERS.md`) | 3 teammates, separately | `labels.csv` has all 3 columns filled |
| **2** | Run `notebooks/Phase3_Rescue_Pipeline_Colab.ipynb` on a T4 (skip §6) | You | `presentation_pack.zip` downloaded |
| **2** | Watch every output video. Pick the best clip as the **main demo**. Re-run §9 with `MAIN` set to it | You | Ablation table for the main clip |
| **3** | `python scripts/eval_priority.py score` → priority result | You | `priority_eval.json` |
| **3** | Fill in the results tables (template below) | Team | Every table has numbers |
| **3–4** | Report chapters and slides (mapping below) | Team | Draft complete |
| **4** | Dry run of the presentation, timed | Team | Under the time limit |
| **5** | Buffer: fix what the dry run exposed, rehearse viva questions | Team | — |

---

## 1. Video requirements (the most important input)

Output quality depends mostly on the clip. A bad clip gives an empty or confusing demo.

**Must have:**
- **Aerial drone view of a real flood**, preferably **top-down or steeply angled**.
  Model 2 was trained on top-down FloodNet images; side views give a poor flood map.
- **People clearly visible**: on roofs, wading, in boats, on a dry patch. At least
  **25–30 px tall in a 1080p frame**. If you can barely see them, the detector can't either.
- **Steady footage**: hovering or a slow pan. Fast spins and zooms break tracking.
- **20–60 seconds** per clip. Longer only slows processing; the stuck-time counter is
  visible after about 20 s.
- **1080p**, MP4 (H.264). Not 4K: frames are scaled to 960 px for detection, so in 4K
  footage people become too small.

**Avoid:** news overlays and big captions, cuts between shots (trim to one continuous
shot), night footage, clips with no people.

**Where to find clips**
- **Pexels** / **Pixabay** (free licence, safest): search "flood drone", "flooded city aerial",
  "flood rescue aerial", "flooded street drone".
- **YouTube** news or rescue drone footage: fine for academic use **with a citation**. Note the
  channel, title, URL and date. Never present it as your own footage.
- Write every source in `videos/SOURCES.txt` (it goes in the report's references).

**Trimming and converting** (ffmpeg, in Colab or locally):
```bash
# keep 0:12–0:52 of one continuous shot, scale to 1080p, H.264
ffmpeg -ss 00:00:12 -to 00:00:52 -i raw.mp4 -vf scale=-2:1080 -c:v libx264 -crf 20 -an clip1.mp4
```

**Quick check before the full run** (local, CPU, ~1 minute):
```bash
python scripts/run_rescue.py --source clip1.mp4 --out outputs/rescue/check --stride 10 --max-frames 15
```
If `annotated.mp4` shows people boxed and a sensible flood overlay, the clip is good.

**If no usable real clip is found:** the notebook falls back to a simulated drone pass
over a real flood photo. It is acceptable, but it must be labelled **"simulated camera
motion over a still photo"** on the slide, and it cannot show people moving or water
rising.

---

## 2. Rater labels

Instructions: `outputs/rescue/priority_eval/README_RATERS.md`. In short: 40 photos, pick
the group to rescue first (0 if no box has a real person), about 15 minutes each,
**independently**. Three raters give a majority vote, and their agreement with each other
gives an upper bound on what any method can reach.

The rating sheets and `labels.csv` are already generated in the repo. Nothing needs to run
before rating.

---

## 3. What "proper output" means (checklist)

Tick every line before the presentation:

**Main demo video**
- [ ] Real flood footage (or clearly labelled simulated), with the source cited
- [ ] People boxed, and boxes stay on people (no wild jumping)
- [ ] Groups drawn with level colours, and the ranked panel readable on a projector
- [ ] The "stuck" time visibly counts up, and at least one alert or High group appears (if none does, explain why: short clip, little water)
- [ ] Plays inside PowerPoint (use the `_demo.mp4` H.264 files from the pack)

**Numbers**
- [ ] Model 1: Phase 1 vs v1 vs rehearsal table (`models/flood_rehearsal_metrics.json`)
- [ ] SAHI test table and the "why dropped" paragraph
- [ ] Model 2: comparison and ablation (`outputs/model2/comparison.md`, `ablation.md`)
- [ ] Pipeline ablation on the main clip (`ablation.csv`): full / no memory / no severity / no motion
- [ ] Priority agreement with raters (`priority_eval.json`), next to random and rater-vs-rater
- [ ] Speed: processed fps on a T4 (`summary.json`)

**Pictures for slides**
- [ ] 4 keyframes from the main clip (pack: `*_frame1-4.png`)
- [ ] 2–3 annotated test photos (pack: `photos/`), ideally the flooded street and the plaza
- [ ] Architecture diagram (existing `Pipeline_Architecture (1).png`, with "+ SAHI" removed)

---

## 4. Results tables to fill (copy into report and slides)

**Pipeline ablation, main clip** (from `ablation.csv`)

| Variant | Groups | Alerts | Top group | Top risk | Critical / High / Medium / Low |
|---|---|---|---|---|---|
| Full system | | | | | |
| No memory | | | | | |
| No flood severity | | | | | |
| No camera-motion compensation | | | | | |

How to read it: *no memory* has no "stuck" time and no water trend, so long-stranded
groups rank lower. *No severity* ranks by head count alone, so a big dry group outranks a
small group in water. *No motion* breaks tracks when the drone moves (compare `tracks`).

**Priority ranking vs human raters** (from `priority_eval.json`)

| Method | Top-1 agreement with rater majority |
|---|---|
| Random pick | |
| People count only (no Model 2) | |
| Flood factors only | |
| **Full system** | |
| Rater vs rater (ceiling) | |

Report whatever comes out. If "people count only" ties or wins, say so: on single photos
most raters may simply pick the biggest group, and the flood factors matter more in video
(water trend, time stranded).

---

## 5. Where each output goes

| Output | Report chapter | Slide |
|---|---|---|
| Model 1 table (3 variants) + forgetting story | Results: detection | "Model 1 results" |
| SAHI table + why dropped | Results: detection (discussion) | one line on the Model 1 slide |
| Model 2 comparison + ablation + per-class chart | Results: segmentation | "Model 2 results" |
| Pipeline architecture + factor table (`docs/RESCUE_PIPELINE_README.md`) | System design | "How it works" |
| Keyframes + demo video | Results: system demo | "Live demo" (play the video) |
| Pipeline ablation | Results: ablations | "What each module adds" |
| Priority vs raters | Results: evaluation | "Does it rank like a human?" |
| Honest limits (README) | Limitations / future work | "Limitations" |

---

## 6. Presentation demo script (about 3 minutes)

1. **Problem (15 s):** rescuers see dozens of people in drone footage. Who first?
2. **Play the main demo video (60 s)**, narrating:
   "boxes are people from Model 1 → coloured overlay is the flood map from Model 2 →
   groups are ranked on the right → watch the stuck time count up → this group turns High
   because it is surrounded by water and has not moved."
3. **Show one annotated photo (20 s):** the flooded street, with 98 % water around the people.
4. **Numbers (60 s):** Model 1 +23 %, Model 2 75.67 mIoU, ablation, rater agreement.
5. **Limits (20 s):** pixels not GPS, water extent not depth, top-down imagery works best.

**Backup:** if the video won't play live, use the keyframe PNGs, and keep
`presentation_pack.zip` on a USB stick and on Drive.

---

## 7. Fixes still needed in the documents

- [x] SAHI removed from the architecture and novelty sections, with "why dropped" added (`Project_Master_Document.md`)
- [ ] Slides: remove "+ SAHI" from the architecture diagram and the Model 1 slide
- [ ] "Detects ≥ 3 classes (person/vehicle/animal)": there is **no animal training data**.
      Say "person and vehicle; animal class reserved, no public flood-animal labels" (limitation).
- [ ] Timeline slide: show what is actually done, not the old Nov–Mar plan, if your mentor reviews progress now.
