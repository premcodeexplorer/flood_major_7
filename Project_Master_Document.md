# Memory-Augmented Temporal Reasoning for Flood Victim Detection and Dynamic Rescue Priority Mapping

**A complete technical master document — for building the project presentation and report.**

Team: Prem Baba (53), Purva Bhoyar (57), Adhishree Shiledar (01), Palak Zade (68)
Guide: Prof. Nisarg Gandhewar (Associate Professor)
Dept. of Computer Science & Engineering, RCOEM Nagpur — Session 2026–2027, Semester 7

---

## 1. PROJECT IN ONE LINE (THE OBJECTIVE)

When people, vehicles, and animals get stranded in a flood, our system detects them from a drone, judges how dangerous each location is, and tells the rescue team **who to save first** — as a ranked rescue-priority map.

---

## 2. COMPLETE PROJECT IDEA (DETAILED)

During floods, rescue teams face two life-threatening problems: (1) many affected areas become physically unreachable because roads submerge and currents turn deadly, and (2) responders lack real-time information on how many people are stranded and where. Ground-based search is slow, dangerous, and cannot cover large inundated regions quickly.

Drones (UAVs) can survey inaccessible flood areas from above and stream high-resolution imagery. However, existing UAV systems typically perform only a single task in isolation — they either detect victims, or segment flood water, or classify a scene — and they describe only the present moment. They do not track victims over time, do not remember how each victim's situation is evolving, and do not convert observations into an actionable, ranked rescue plan. Additionally, from drone altitude, victims appear as tiny, partially-occluded objects that standard detectors frequently miss — which is especially dangerous in rescue, where a single missed detection can cost a life.

Our project builds an **integrated, temporally-aware pipeline** that:
1. **Detects** stranded people, vehicles, and animals from drone footage — including tiny and occluded targets.
2. **Tracks** each victim across frames with a stable identity.
3. **Remembers** each victim's recent state over time (how long stranded, movement, trend).
4. **Assesses flood severity** — how deeply flooded the area around each victim group is.
5. **Fuses** all of this into a rescue-priority score and produces a **dynamic rescue-priority map** with a victim summary table and alerts.

The system does not just detect — it goes from **perception all the way to a prioritized rescue decision**.

---

## 3. MOTIVATION (WHY THIS PROJECT)

**Social / Humanitarian:**
- Floods kill thousands every year; faster, smarter, prioritized rescue directly reduces deaths.
- Helps rescue teams reach the unreachable and focus effort where it matters most.
- Protects the most vulnerable during disasters.

**Technical:**
- The hardest technical problem (detecting tiny, hidden victims from height) is *also* the most important thing for rescue — our difficulty and our purpose align.
- No existing flood-detection work combines detection + tracking + memory + flood-severity + prioritization. This integration is a genuine gap.

**Feasibility:**
- Built on proven, well-documented tools (YOLO, SAHI, ByteTrack, U-Net) and available datasets (VisDrone, Roboflow flood, FloodNet) — so it is buildable within a student project.

---

## 4. PROBLEM STATEMENT

Existing UAV flood-response solutions largely stop at single-frame detection or segmentation and do not provide rescue teams with a temporally-informed, prioritized decision output. There is a need for a system that detects stranded victims (including small and occluded targets) from drone imagery, tracks them across time, remembers how their situation is changing, incorporates flood severity, and produces a dynamic rescue-priority map. This project proposes an integrated framework combining aerial victim detection (humans, vehicles, animals), multi-object tracking, short-term temporal memory, and flood-severity awareness to estimate a rescue-priority score for each affected location and generate an interpretable priority map that guides responders to the highest-risk victims first.

---

## 5. AIM

To develop a drone-based system that detects, tracks, and remembers stranded flood victims, fuses flood-severity information, and generates a dynamic rescue-priority map that guides teams to the most critical locations first.

---

## 6. OBJECTIVES

1. To build an aerial detector for stranded humans, vehicles, and animals, with special handling for tiny and partially-occluded targets using slicing-aided inference (SAHI).
2. To adapt a general aerial detector to flood conditions via two-stage training (pre-training on VisDrone, then fine-tuning on flood-specific data).
3. To track detected victims across frames and maintain a short-term memory of each victim's evolving state (time stranded, movement, trend).
4. To integrate flood-severity segmentation so victims in more severely flooded areas are prioritized appropriately.
5. To compute a rescue-priority score per location and generate a dynamic rescue-priority map with a victim summary.
6. To evaluate the system against baseline detectors and through ablation studies measuring each module's contribution.

---

## 7. OUR SOLUTION (HIGH LEVEL)

A single integrated pipeline of **two trained deep-learning models** and **three logic-based modules**:

```
Drone Flood Footage
   │
   ├──> MODEL 1: Victim Detection (YOLOv8 + SAHI)  [trained]
   │         └──> Tracking (ByteTrack)              [logic]
   │                  └──> Short-Term Memory        [logic]
   │
   └──> MODEL 2: Flood Severity (U-Net)             [trained]
   │
   └──────────────> RISK & PRIORITY (fusion formula) [logic]
                          │
                          └──> Rescue-Priority Map + Table + Alerts
```

---

## 8. LITERATURE SURVEY (10+ PAPERS)

| # | Paper (Year) | Key Contribution / Method | Limitation / Gap |
|---|---|---|---|
| 1 | Prabhu et al., 2022 — **RescueNet** | YOLO-based detection AND counting of flood survivors from UAV imagery; high reported mAP | Single-frame only; no tracking, memory, or rescue prioritization |
| 2 | Zhao, Sui & Zhu, 2024 — **Real-time Rescue Target Detection** | Real-time UAV method that delineates flood extent and detects trapped pedestrians and vehicles | No temporal tracking, no victim-state memory, no ranked priority output |
| 3 | Nihal et al., 2024 — **C2A dataset** | Synthetic UAV disaster human-detection dataset; fine-tuning on it improves SAR detection substantially | General disaster (not flood-specific); detection only; synthetic/composite data |
| 4 | Rahnemoonfar et al., 2021 — **FloodNet** | High-resolution UAV imagery dataset for post-flood scene understanding (segmentation, classification, VQA) | Built for scene understanding, not victim detection or rescue ranking |
| 5 | Comparative study, 2025 — **YOLO for flood human position** | Compares YOLO variants (YOLO12) + optical flow + Kalman for detecting people on water | Maritime/water focus; no victim memory or rescue-priority mapping |
| 6 | Fine-tuned study, 2025 — **UAV rooftop victim detection** | Fine-tuned YOLOv9 (~87% AP) for people stranded on rooftops/in buildings during floods | Detection only; no counting, tracking, or decision output |
| 7 | Akyon, Altinuc & Temizel, 2022 — **SAHI** | Slicing-Aided Hyper Inference: slices images into tiles to greatly improve small-object detection | A technique, not a full rescue system; increases inference time |
| 8 | Zhang et al., 2022 — **ByteTrack** | Multi-object tracking that associates every detection box (high + low confidence) for robustness | Generic tracker; no flood context or prioritization logic |
| 9 | Kyrkou & Theocharides, 2020 — **AIDER** | CNN-based aerial image classification for emergency response (fire, flood, collapse) | Scene-level classification only; no victim localization or ranking |
| 10 | Hassan et al., 2024 — **Aerial flood scene (South Asia)** | YOLOv8 to detect humans/houses in flood imagery (~67% human accuracy) | Low human-detection accuracy; no tracking, memory, or priority |
| 11 | Zhu et al., 2021 — **VisDrone** | Large-scale aerial detection & tracking benchmark from drones (people, vehicles) | Urban, not flood; dry-city pedestrians, not flood victims |

**Key Insight / Research Gap:** All existing work solves only ONE piece of the problem. Most operate on single frames and stop at detection or counting; those with tracking are built for maritime/thermal settings, not flood terrain; scene-level datasets do not localize or rank victims. The field also lacks a standard flood-victim benchmark. **No unified, temporally-aware system detects flood victims (including tiny/occluded ones), tracks and remembers their evolving state, fuses flood severity, and produces a ranked rescue-priority map. This project fills that gap.**

---

## 9. THE TWO MODELS — WHAT THEY ARE

### MODEL 1 — Victim Detector
- **Type:** YOLOv8 (object detection)
- **Job:** detect and locate people, vehicles, animals → bounding boxes with class labels
- **Special technique:** SAHI (Slicing-Aided Hyper Inference) for tiny, far, occluded victims
- **Trained on:** VisDrone (base) → fine-tuned on Roboflow flood dataset
- **Output:** boxes + classes + confidence per frame

### MODEL 2 — Flood Severity
- **Type:** U-Net (semantic segmentation)
- **Job:** segment the water and estimate how deeply flooded each area is (water extent/severity), from the current image
- **Trained on:** FloodNet
- **Output:** a water/severity map per frame
- **Important:** Model 2 reads the *present* water condition around a location — it does NOT predict future water levels.

**Everything else is NOT a trained model:**
- Tracking (ByteTrack) = algorithm
- Memory = data structure (per-victim state buffer)
- Risk & Priority = weighted scoring formula

---

## 10. HOW WE TRAIN THE TWO MODELS (TRAINING ARCHITECTURE)

### MODEL 1 — Two-Stage Training + SAHI

**Why two stages:** VisDrone teaches the general skill of "detecting people/vehicles from drone height." Flood data teaches the specific skill of "recognizing victims against flood backgrounds (water, rooftops, debris)." Training on both (general → specific) generalizes better than either alone.

```
STAGE 1 — Base training on VisDrone
  - Dataset: VisDrone (aerial, urban, people + vehicles)
  - Convert VisDrone annotations to YOLO format
  - Merge vehicle sub-classes into one "vehicle" class; keep "person"
  - Train YOLOv8 → model learns general overhead object appearance
  - Output: base detector weights

STAGE 2 — Fine-tuning on flood data (Roboflow)
  - Dataset: Roboflow flood set (Person-Flood / Rooftop / Boat, animals, vehicles)
  - Convert to YOLO format; align/remap classes (person, vehicle, animal)
  - Continue training (fine-tune) the base detector on flood images
  - Output: flood-adapted detector weights

SMALL-OBJECT HANDLING (applied during inference + fine-tuning) — SAHI:
  - Slice each large drone image into overlapping tiles
  - Run the detector on each zoomed-in tile (tiny victims now appear larger)
  - Merge tile detections back with overlap handling (NMS)
  - Result: significantly more tiny / occluded victims detected
```

**Training details / hyperparameters (typical, to be tuned):**
- Framework: PyTorch + Ultralytics YOLOv8
- Input resolution: high (e.g., 1280) to preserve small-object detail
- Augmentation: mosaic, flips, scaling, HSV jitter
- Optimizer: SGD/AdamW; learning-rate warmup then decay
- Loss: YOLO box + class + objectness loss
- Fine-tuning: lower learning rate than base training (transfer learning)

**Model 1 metrics:** mAP, mAP@0.5, small-object AP, precision, recall.

### MODEL 2 — Segmentation Training on FloodNet

```
STAGE — Train U-Net on FloodNet
  - Dataset: FloodNet (high-res UAV flood imagery with water/scene masks)
  - Prepare water masks (flooded vs non-flooded regions)
  - Train U-Net (encoder-decoder) to segment water extent
  - Output: severity/segmentation model weights
```

**Training details:**
- Architecture: U-Net (encoder downsamples, decoder upsamples, skip connections)
- Loss: Dice loss / cross-entropy (segmentation)
- Metric: IoU (Intersection over Union), mean IoU
- Augmentation: flips, scaling, brightness

### What is NOT trained
- ByteTrack (tracking) — works on Model 1's outputs directly
- Memory module — a per-victim state buffer, updated each frame
- Risk & Priority — a weighted formula

---

## 11. SYSTEM ARCHITECTURE (FULL PIPELINE) & DATA FLOW

```
INPUT: Drone flood footage (video / image frames)
        │
        ├──────────────────────────────┐
        ▼                              ▼
[MODULE 1] Victim Detection      [MODULE 4] Flood Severity
   YOLOv8 + SAHI                    U-Net segmentation
   (trained: VisDrone→flood)        (trained: FloodNet)
   → boxes: person/vehicle/animal   → water severity per region
        │                              │
        ▼                              │
[MODULE 2] Tracking (ByteTrack)        │
   → stable ID + trajectory per victim │
        │                              │
        ▼                              │
[MODULE 3] Short-Term Memory           │
   → per-victim state: time stranded,  │
     movement, trend across frames     │
        │                              │
        └──────────────┬───────────────┘
                       ▼
[MODULE 5] Risk & Priority (fusion formula)
   risk = f(group size, flood severity, time stranded,
            isolation, rising-trend)  → score in [0,1]
   rank into Critical / High / Medium / Low
                       │
                       ▼
OUTPUT:
   - Dynamic Rescue-Priority Map (color-coded markers)
   - Victim Summary Table (ID, position, count, risk, priority)
   - Alerts for critical locations
```

**Component summary:**

| Module | Type | Trained? | Tool |
|---|---|---|---|
| 1. Victim Detection | Deep-learning model | Yes | YOLOv8 + SAHI |
| 2. Tracking | Algorithm | No | ByteTrack |
| 3. Short-Term Memory | Data structure | No | Custom logic |
| 4. Flood Severity | Deep-learning model | Yes | U-Net |
| 5. Risk & Priority | Formula | No | Weighted scoring |

---

## 12. HOW IT WORKS — STEP BY STEP (RUNTIME)

1. **Input:** drone footage enters frame by frame.
2. **Detect (Model 1 + SAHI):** find people/vehicles/animals; SAHI catches tiny/hidden ones.
3. **Track (ByteTrack):** assign each victim a stable ID across frames → trajectories.
4. **Group / cluster:** nearby victims are grouped into rescue units (e.g., "8 people on this rooftop").
5. **Memory (Module 3):** for each victim/group, store recent history (time stranded, movement, whether water is rising — compared across frames).
6. **Severity (Model 2):** segment the water; find how deeply flooded the area around each group is.
7. **Fusion — Risk score (Module 5):** combine group size + flood severity + time stranded + isolation + rising-trend → a risk score in [0,1].
8. **Rank:** sort groups; classify into Critical (>0.8), High (0.5–0.8), Medium (0.3–0.5), Low (<0.3).
9. **Output:** priority map + victim summary table + alerts telling the team who to rescue first.

---

## 13. KEY TECHNICAL POINTS & HONEST SCOPE (IMPORTANT)

**Location = relative position, not GPS.**
From a drone image we get each victim's **pixel position (x, y)** — the box center. Since all victims share the same frame, we know their **relative positions**, which we plot on a scene map. Exact GPS (lat/long) would need the drone's GPS + camera telemetry, which the datasets don't include — so GPS geolocation is noted as a **deployment/future extension**.

**Memory = per-individual, powered by tracking.**
Tracking gives each victim a stable ID; memory stores that ID's history over frames. Memory needs **video** (multiple frames) to build history. It is used for **continuous monitoring across a multi-victim, multi-hour rescue operation** — updating priorities as conditions change — NOT for delaying any single rescue.

**"Rising water over time" = compare frames at two times.**
We run Model 2 on frames at different times, store readings in memory, and compare them to detect if water is rising. This assumes a **steady drone viewpoint** (or frame alignment); otherwise "more water" could just mean the drone moved.

**We do NOT predict the future (honest boundary).**
We do not forecast exact future flood levels or "survival in the next hour" — no dataset supports that. We use **current severity + observed rising-trend** only.

**Model 2 gives water severity around a location, not the exact floor.**
A top-down view shows water extent/depth around buildings — not "water reached the 3rd floor." Use the floor idea only to *explain the concept*.

**Multimodal inputs (thermal/sensors/GIS) are OUT of scope.**
No paired flood datasets with thermal/sensors/GIS exist for training — so we use **drone RGB footage only**. Multimodal is future work.

---

## 14. DATASETS (WHAT, WHY, FROM WHERE)

| Dataset | Role in project | Used for | Notes |
|---|---|---|---|
| **VisDrone** | Base training (Model 1) | General aerial detection of people/vehicles | Direct download (GoogleDrive mirror); urban, aerial |
| **Roboflow flood set** | Fine-tuning (Model 1) | Adapt to flood victims (rooftop/boat/water, animals, vehicles) | Detection-ready (YOLO export); verify it is aerial + enough images |
| **FloodNet** | Training (Model 2) | Flood-water severity segmentation | High-res post-flood UAV imagery; prepare water masks |

**Honest note:** There is no single public "flood video with labeled stranded people" benchmark, so most flood-rescue works use their own data. Model 1 is fine-tuned on flood **images**; tracking + memory are demonstrated on **video** (a short flood clip or aerial video), which we state clearly.

---

## 15. TECHNOLOGY STACK

| Category | Tools |
|---|---|
| Language | Python |
| Detection | Ultralytics YOLOv8 + SAHI |
| Tracking | ByteTrack |
| Segmentation | U-Net (flood severity) |
| Frameworks/Libraries | PyTorch, OpenCV, NumPy |
| Datasets | VisDrone, Roboflow flood, FloodNet |
| Platform | Google Colab / GPU-enabled workstation |
| Version Control | Git / GitHub |

---

## 16. WHAT MAKES IT UNIQUE (NOVELTY)

**Headline novelty:** First to combine **Detection + Tracking + Memory + Flood-Severity + Rescue Prioritization** in ONE flood system — from perception all the way to prioritized action. No existing flood-detection work does all of this together.

**Supporting contributions:**
1. **Tiny/occluded-victim detection** — SAHI + occlusion-aware fine-tuning catches the small, hidden victims standard detectors miss (the most rescue-critical case).
2. **Two-stage training** — VisDrone → flood fine-tuning, shown better than single-dataset training (proven via VisDrone-only vs fine-tuned comparison).
3. **Memory-based risk reasoning** — uses each victim's changing situation over time, not just one frame.
4. **Severity-aware prioritization** — fuses how flooded an area is into the rescue ranking; no prior work does this.

**Honest framing:** The novelty is in the **integration + application + decision-useful output**, not in inventing a new neural network. This is a valid, respected type of contribution.

---

## 17. EVALUATION PLAN (HOW WE PROVE IT WORKS)

**Metrics:**
- Detection: mAP, mAP@0.5, small-object AP, precision, recall
- Tracking: MOTA, IDF1
- Segmentation (Model 2): IoU / mean IoU
- Priority: priority-ranking accuracy (do the right locations rank highest)
- Speed: FPS (if claiming near real-time)

**Comparisons / ablations (6 planned):**
1. VisDrone-only detector vs flood-fine-tuned detector (proves fine-tuning helps)
2. Plain YOLO vs YOLO+SAHI (proves small-object handling helps)
3. Risk with vs without memory (ablation)
4. Risk with vs without flood-severity (ablation)
5. Capability comparison table vs closest prior work (RescueNet / Real-time / YOLO-flood)
6. Test on a public set (FloodNet / NITC) for context

**Honest caution:** Because there is no common flood-victim benchmark, we compare fairly by re-running a baseline on our own data; we cite others' numbers only as context (different datasets), never as head-to-head wins.

---

## 18. EXPECTED OUTCOMES & DELIVERABLES

**Outcomes:**
- A trained victim detector (person/vehicle/animal) with improved small-object detection over a plain-YOLO baseline.
- A flood-severity segmentation model estimating water extent per region.
- An integrated pipeline: detect → track → memory → severity → priority.

**Deliverables:**
- Trained model weights (detector + segmenter)
- Documented Git repository (full pipeline source code)
- Annotated output videos, a victim summary table (ID, position, count, risk, priority), and a rescue-priority map
- Evaluation report with the comparisons/ablations above

**Functional specs:**
- Input: aerial flood video/images → Output: annotated frames + priority map + alerts
- Detects and counts ≥ 3 classes; assigns each victim a stable unique ID
- Risk score = f(count, flood severity, time stranded, isolation, trend), normalized to [0,1]
- Priority levels: Critical (>0.8), High (0.5–0.8), Medium (0.3–0.5), Low (<0.3)

---

## 19. PROJECT SCOPE

**In scope:** drone-captured imagery of flood-affected areas; detection + tracking + short-term memory + flood-severity fusion + priority mapping for visible stranded victims (humans, vehicles, animals) in open/semi-open flood scenes; a decision-support tool for rescue teams and disaster command centres.

**Out of scope / future work:** control of the drone itself; dense-forest-canopy scenarios (victims physically hidden from optical cameras); multi-sensor fusion (thermal, GIS, live sensors); exact GPS geolocation; long-term historical-flood memory; and minute-level future-risk forecasting.

---

## 20. TIMELINE (TILL MARCH 2027)

| Phase | Duration | Milestone |
|---|---|---|
| Phase 1 — Literature review & requirements | Aug–Sep 2026 | Problem, datasets, baseline finalized |
| Phase 2 — Design & methodology | Oct 2026 | Architecture & module design frozen |
| Phase 3 — Detection + Tracking | Nov–Dec 2026 | Train YOLO+SAHI, fine-tune on flood, add ByteTrack |
| Phase 4 — Memory + Severity | Jan 2027 | Short-term memory, FloodNet U-Net, risk scoring |
| Phase 5 — Integration + Testing | Feb 2027 | Priority map, baseline & ablation results, metrics |
| Phase 6 — Documentation | Mar 2027 | Final report, demo, presentation |

---

## 21. REFERENCES (IEEE)

[1] B. V. B. Prabhu, R. Lakshmi, R. Ankitha, M. S. Prateeksha, and N. C. Priya, "RescueNet: YOLO-based object detection model for detection and counting of flood survivors," Model. Earth Syst. Environ., vol. 8, no. 4, pp. 4509–4516, 2022.
[2] B. Zhao, H. Sui, and Y. Zhu, "Real-time rescue target detection based on UAV imagery for flood emergency response," J. Geodesy and Geoinformation Science, vol. 7, no. 1, pp. 74–89, 2024.
[3] R. A. Nihal, B. Yen, K. Itoyama, and K. Nakadai, "UAV-Enhanced Combination to Application: benchmarking of a human detection dataset for disaster scenarios," in Proc. Int. Conf. Pattern Recognition (ICPR), 2024.
[4] M. Rahnemoonfar, T. Chowdhury, A. Sarkar, D. Varshney, M. Yari, and R. R. Murphy, "FloodNet: a high-resolution aerial imagery dataset for post-flood scene understanding," IEEE Access, vol. 9, pp. 89644–89654, 2021.
[5] "Evaluation of YOLO models for human position recognition with UAVs during a flood," Applied System Innovation, vol. 9, no. 1, 2025.
[6] "Fine-tuned deep models for people detection in UAV building images to aid flood rescue," Int. J. Applied Earth Observation and Geoinformation, 2025.
[7] F. C. Akyon, S. O. Altinuc, and A. Temizel, "Slicing aided hyper inference and fine-tuning for small object detection," in Proc. IEEE Int. Conf. Image Processing (ICIP), 2022.
[8] Y. Zhang et al., "ByteTrack: multi-object tracking by associating every detection box," in Proc. European Conf. Computer Vision (ECCV), 2022.
[9] C. Kyrkou and T. Theocharides, "Deep-learning-based aerial image classification for emergency response applications using UAVs (AIDER)," in Proc. IEEE CVPR Workshops, 2020.
[10] I. Hassan et al., "Aerial flood scene classification using fine-tuned attention-based architecture for flood-prone countries in South Asia," arXiv:2411.00169, 2024.
[11] P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, and H. Ling, "Detection and tracking meet drones challenge (VisDrone)," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 7380–7399, 2021.

---

## 22. SUGGESTED PPT SLIDE ORDER (FOR THE PPT TOOL)

1. Title
2. Contents
3. Introduction (problem)
4. Motivation
5. Literature Survey (table of 10+ papers)
6. Research Gap
7. Aim & Objectives
8. Proposed Solution / System Architecture (diagram)
9. The Two Models (what they are)
10. Training Architecture — Model 1 (two-stage + SAHI)
11. Training Architecture — Model 2 (U-Net on FloodNet)
12. How It Works — step by step
13. Datasets
14. Technology Stack
15. Uniqueness / Novelty
16. Evaluation Plan
17. Expected Outcomes & Deliverables
18. Scope (in-scope + future work)
19. Timeline (till March 2027)
20. References
21. Thank You

---

*Honest reminder: the priority-map and detection visuals are illustrative mockups of intended output; actual results will come after training and testing. Multimodal inputs, GPS geolocation, and future-risk forecasting are future work, not current claims.*
