# Phases 3-5 — Rescue Pipeline

The two trained models joined into one system. Nothing here is trained.

```
frame ─► Model 1: detector (flood_rehearsal.pt)
      ─► ByteTrack (in camera-compensated scene coordinates)
      ─► group people into rescue units
      ─► memory: time stranded, water trend          ◄─ Model 2: FloodNet U-Net + scSE
      ─► risk = weighted factors in [0, 1]                 (severity, isolation)
      ─► Critical > 0.8 · High 0.5–0.8 · Medium 0.3–0.5 · Low < 0.3
      ─► annotated video + priority panel, victims.csv, tracks.csv, alerts.json
```

## Files

| File | What it does |
|---|---|
| `scripts/rescue/detect.py` | Model 1; SAHI available behind `--sahi` but not used (see Results) |
| `scripts/rescue/track.py` | ByteTrack from Ultralytics, fed any detections (also SAHI's) |
| `scripts/rescue/motion.py` | Camera-motion compensation: LK optical flow + RANSAC similarity, chained to the first frame |
| `scripts/rescue/severity.py` | Model 2 wrapper; severity and isolation lookups on its class map |
| `scripts/rescue/memory.py` | Per-person memory, re-identification by place, grouping |
| `scripts/rescue/risk.py` | The risk formula, levels, ablation switches |
| `scripts/rescue/render.py` | Annotated frame + ranked priority panel |
| `scripts/rescue/pipeline.py` | One frame through all modules |
| `scripts/run_rescue.py` | CLI: video / frame folder / photos → all outputs |
| `scripts/eval_sahi.py` | Plain YOLO vs YOLO + SAHI, COCO protocol (AP50, AP_small …) |
| `scripts/eval_priority.py` | Human-judgement evaluation of the ranking (rating sheets → top-1 agreement) |
| `scripts/make_demo_video.py` | Simulated drone pass over a still photo (testing only) |
| `scripts/_test_rescue.py` | Logic tests: grouping, memory, re-ID, risk, severity, camera motion |
| `notebooks/Phase3_Rescue_Pipeline_Colab.ipynb` | Everything above on a Colab GPU (`scripts/_gen_rescue_notebook.py`) |

```bash
python scripts/_test_rescue.py
python scripts/run_rescue.py --source flood.mp4 --out outputs/rescue/demo --stride 3
python scripts/run_rescue.py --source data/flood_c2a/test --out outputs/rescue/test_photos
python scripts/run_rescue.py --source flood.mp4 --out ... --no-memory      # ablation
```

Needs `ultralytics sahi segmentation-models-pytorch lapx pycocotools` (see `requirements.txt`).

## How each factor is measured

| Factor | Source | Definition | Scaled to 1 at |
|---|---|---|---|
| size | Model 1 | people in the group | 10 people |
| severity | Model 2 | share of flooded pixels (water, flooded road, flooded building) in a disk of 4 body sizes around the group, background ignored | fully surrounded |
| isolation | Model 2 | distance to "mainland": dry ground (dry road, grass, dry building, unlabelled) connected to the edge of the view | ¼ of the frame diagonal |
| stranded | memory | seconds since the group last moved more than one body size (scene coordinates) | 300 s |
| trend | memory | slope of severity around the person over the last 60 s | +0.10 per minute |

Weights: size 0.20, severity 0.30, stranded 0.25, isolation 0.10, trend 0.15. They are
a **design choice**, not fitted, because no ground truth for “who to rescue first” exists.
They put the water first (severity + isolation + trend = 0.55). Turning a module off
drops its factors and renormalises the rest (`--no-memory`, `--no-severity`).

Groups: people join when they are within 3 body sizes of each other (single linkage).
Clusters wider than 12 body sizes are split, because one boat cannot cover a whole crowd.

## Design decisions worth explaining in the viva

- **Tracking in scene coordinates.** ByteTrack matches by box overlap. When a drone pans,
  a 15 px person shifts more than their own width between frames, and the track breaks.
  Boxes are therefore moved into first-frame coordinates before tracking. On a simulated
  pan this kept **2.3× more people tracked per frame** (7.4 → 17.4). Registration error
  was about 3 grey levels, against about 55 without compensation, with 0 failures.
- **Re-identification by place.** A half-submerged person's detection flickers, and
  ByteTrack then hands out a new ID. A new track that appears where a lost one stood
  inherits its memory. Without this, “time stranded” restarted at 0 on every break
  (6 s instead of 20 s after 20 s).
- **Model 2 every second, not every frame.** Water does not change in 1/30 s. Positions are
  carried into the older map through the camera transform.
- **Mainland, not just “dry road”.** The first version looked for dry road only. Many
  photos contain none, so every group scored isolation 1. Dry ground connected to the
  edge of the view is the honest version of “a way out”.

## Results so far

### SAHI on the real flood test photos (109 photos, 1,330 people; CPU run)

| Method | AP50 | AP50-95 | AP_small | AR@300 | s / image (CPU) |
|---|---|---|---|---|---|
| YOLO | **0.404** | **0.178** | **0.152** | **0.335** | 1.2 |
| YOLO + SAHI (320 px tiles) | 0.326 | 0.139 | 0.116 | 0.306 | 5.9 |

The plain-YOLO AP50 (0.404, pycocotools) matches `train_flood.py`'s 0.399 (Ultralytics),
which confirms the evaluator.

**Decision: SAHI is dropped.** The pipeline uses standard detection; `--sahi` stays as an
option for genuinely high-resolution footage. Why:

1. **It lowered accuracy by 8 AP50 points** (0.404 → 0.326), and small-person AP too (0.152 → 0.116).
2. **The cause is resolution.** The test photos are 640 px. The detector already up-samples
   them to 960 px, and 320 px tiles enlarge each region about 3× more, so people blur and
   get cut at tile borders. SAHI is designed for 2K–4K footage where people are a few pixels tall.
3. **It costs about 5× the compute** per image (1.2 s → 5.9 s on CPU), which matters for video.

For the viva: *"We tested SAHI rather than assuming it helps. On our data it reduced AP50
from 0.404 to 0.326, because the images are low-resolution and tiling over-enlarges them,
so we kept standard detection."*

### Still to fill in (Colab)

- Real-video run + ablations (§8–9): full / no memory / no severity / no motion
- (optional) SAHI on VisDrone val (§6), only if time allows
- Priority-ranking agreement with human raters (§10)

## Honest limits

- **Positions are image pixels, not GPS.** No telemetry in the data. With drone GPS and
  camera pose, the same centres would project to latitude and longitude.
- **Severity means water around the group, not depth.** Depth cannot be read from one RGB image.
- **Model 2 is trained on nadir FloodNet imagery** (Hurricane Harvey, ~200 ft). It is good
  on top-down views (a flooded street scored 98 % flooded around the people), weaker on
  oblique news photos. Check the overlay before trusting the numbers.
- **Tracking metrics (MOTA/IDF1) need a labelled flood video.** None exists publicly. Tracking is
  shown qualitatively, plus track-lifetime statistics from `tracks.csv`.
- **The simulated video moves the camera over a still photo.** It tests tracking and
  compensation, but cannot show people moving or water rising.
