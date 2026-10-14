# FloodRescue: context for Claude

Final-year major project (Group 7, RCOEM): drone-based flood victim detection, tracking,
flood-severity fusion and rescue-priority ranking. **The presentation is ~2026-10-16.**

**Start with `HANDOFF_TODO.md`.** It is the task list. Work through it in order and tick boxes as
steps finish. Background and reasoning: `docs/FINAL_DEMO_PLAN.md`, `docs/RESCUE_PIPELINE_README.md`.

## Ground rules
- **No training, no new models, no new features.** Both models are final. The remaining work is
  running the existing pipeline, collecting numbers, and updating slides and the report.
- **Never invent numbers.** Every result in slides or report must come from a file:
  `models/flood_rehearsal_metrics.json`, `outputs/model2/comparison.md`, `ablation.md`,
  `outputs/rescue/sahi_eval_flood_cpu.md`, and (after the Colab run) `ablation.csv`,
  `summary.json`, `priority_eval.json`. If a number is missing, leave the cell blank and say so.
- SAHI was tested and **dropped**. Skip notebook section 6, and remove "+ SAHI" from slides.
- No animal training data exists. Say "persons and vehicles", and list animals as a limitation.
- The simulated video must be labelled "simulated camera motion over a still photo".
- Cite every video clip used (`videos/SOURCES.txt`).

## Where things are
- GPU work runs on **Google Colab (T4)**: `notebooks/Phase3_Rescue_Pipeline_Colab.ipynb`.
  It reads weights and data from `My Drive/FloodRescue/` (sent separately via Google Drive, not in git).
- Local machine: CPU only. The weights are not in git; copy them into `models/` from the Drive folder.
- `scripts/eval_priority.py score` needs only opencv and numpy, no weights.
- Slides: `FloodRescue_Major_Project_PPT_1st.pptx` (first-review deck, needs a results section).
  Full project write-up: `Project_Master_Document.md`.
