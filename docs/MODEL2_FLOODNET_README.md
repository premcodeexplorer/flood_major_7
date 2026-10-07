# Model 2 — Flood Segmentation on FloodNet

Model 2 serves two purposes: it is the flood-severity model in the main pipeline, and it is
the subject of a separate research paper. Everything here produces numbers that
are fair to compare: one fixed split, one resolution and one schedule for every model.

## Workflow

| step | who | notebook | GPU? | time |
|---|---|---|---|---|
| 1. Download, verify and resize | **one person, once** | `Model2_FloodNet_1_Prep_Colab.ipynb` | no (CPU runtime) | ~30 min |
| 2. Train 2 models each | teammates 1 / 2 / 3 | `Model2_FloodNet_2_Train_Teammate{1,2,3}_Colab.ipynb` | T4 | ~2–3 h per model |
| 3. Build the table | you, locally | `python scripts/floodnet_compare.py --results outputs/model2/results` | no | seconds |

Notebook 1 leaves `MyDrive/FloodSeg/floodnet_1024.zip` (~1 GB). Share it as
"Anyone with the link"; teammates paste that link into `DATA_LINK` in their notebook.
After each model, teammates send back `results/<run>.json`, `history.json` and
`test_predictions.jpg`. Collect them in `outputs/model2/results/` and
`outputs/model2/runs/<run>/`.

### Who trains what (6 models, 2 per teammate)

| notebook | model 1 | model 2 |
|---|---|---|
| Teammate 1 | `unet_r50` | `deeplabv3p_r50` |
| Teammate 2 | `pspnet_r50` | `unet_r50_scse_hybrid` (ours) |
| Teammate 3 | `unet_r50_scse` (ablation) | `unet_r50_hybrid` (ablation) |

Each notebook trains its two models one after the other. If a session ends, re-running
every cell skips the finished model and resumes the other from its last epoch.

## Source of truth

| File | Role |
|---|---|
| `scripts/floodnet_prep.py` | Finds and pairs images and masks, validates them, resizes to 1024×768 once, writes class-frequency stats |
| `scripts/floodnet_train.py` | The 6 runs, training with per-epoch checkpoint and resume, test evaluation, params and FPS, results JSON |
| `scripts/floodnet_compare.py` | Paper table, ablation table, per-class bar chart |
| `scripts/_gen_floodnet_notebooks.py` | Rebuilds the 4 notebooks from the scripts (the teammate split is `TEAM` there) |
| `scripts/_test_floodnet.py` | End-to-end smoke test on synthetic data: all runs, resume, guards, compare |

The notebooks are **generated**. Edit the `.py` files, then:

```
python scripts/_test_floodnet.py        # needs torch + smp + albumentations (CPU is fine)
python scripts/_gen_floodnet_notebooks.py
```

## Dataset

**FloodNet-Supervised v1.0**: the official release, github.com/BinaLab/FloodNet-Supervised_v1.0
(Dropbox, `FloodNet.zip`, 13.06 GB, CDLA-Permissive 1.0). Do **not** use the
EarthVision 2021 challenge release, which is semi-supervised with only ~400 labelled images.

- 2,343 UAV images (~4000×3000) after Hurricane Harvey, 10 classes:
  background, building-flooded, building-non-flooded, road-flooded,
  road-non-flooded, water, tree, vehicle, pool, grass
- **Official split, used as given:** expected 1445 / 450 / 448. Prep prints the
  real counts; report those.
- Cached once at **1024×768** (images: area interpolation, masks: nearest). Training
  uses random **512×512 crops**; evaluation uses the **full 1024×768 frame**.

## Protocol (identical for every run)

| | |
|---|---|
| Optimiser | AdamW, lr 2e-4, weight decay 1e-4 |
| Schedule | 1 epoch linear warm-up, then cosine, per iteration |
| Epochs, batch | 80, 8 |
| Augmentation | scale ±25 %, 512 crop, h/v flip, 90° rotation, brightness/contrast |
| Precision | AMP (fp16) |
| Encoder init | ImageNet |
| Model selection | best **val** mIoU(9), checked every 2 epochs |
| Reported | **test** split, once, with the selected checkpoint |
| Seed | 0 (1, 2 for repeats) |

### Runs

| run | model | loss | role |
|---|---|---|---|
| `unet_r50` | U-Net, ResNet-50 | CE | baseline and the plain version of ours |
| `deeplabv3p_r50` | DeepLabV3+, ResNet-50 | CE | baseline (also in the FloodNet paper) |
| `pspnet_r50` | PSPNet, ResNet-50 | CE | baseline (the FloodNet paper's best) |
| `unet_r50_scse_hybrid` | U-Net, ResNet-50 + scSE attention | Dice + Focal | **ours** |
| `unet_r50_scse` | U-Net, ResNet-50 + scSE attention | CE | ablation: attention only |
| `unet_r50_hybrid` | U-Net, ResNet-50 | Dice + Focal | ablation: loss only |

All six share the ResNet-50 encoder, so the differences come from the decoder,
the attention and the loss, not from a bigger backbone. `unet_r50` → ours shows
what our two changes add **together**; the two ablation runs (Teammate 3) show
what scSE alone and Dice/Focal alone add, so `ablation.md` splits the gain. scSE is concurrent spatial and channel
squeeze-and-excitation (Roy et al., MICCAI 2018) on every U-Net decoder block.

## Metrics

- **mIoU(9)**: mean IoU over the 9 non-background classes. This is the definition used
  in FloodNet Table 4 (PSPNet: 79.69 = mean of its 9 per-class values), so it is the
  **headline metric**.
- mIoU(10) including background, pixel accuracy, and per-class IoU.
- **Hard-4**: mean IoU of building-flooded, road-flooded, vehicle and pool, the
  classes every paper finds hardest. This is the focus metric for the paper's claim.
- Params (M) and FPS (batch 1, 1024×768, fp16). **FPS depends on the GPU.** If runs
  happened on different GPUs, the compare script warns you. Re-measure FPS for all
  models on one GPU (`--eval-only` on the same machine) before publishing.

## Published numbers (context only)

FloodNet paper, Table 4 (verified from arXiv:2012.02951). These rows go in the table
**marked as cited, not as a like-for-like comparison**. The paper used 713×713 crops,
PSPNet with ResNet-101, and its own schedule.

| method | Bld-F | Bld-NF | Road-F | Road-NF | Water | Tree | Vehicle | Pool | Grass | mIoU |
|---|---|---|---|---|---|---|---|---|---|---|
| ENet | 6.94 | 47.35 | 12.49 | 48.43 | 48.95 | 68.36 | 32.26 | 42.49 | 76.23 | 42.61 |
| DeepLabV3+ | 32.7 | 72.8 | 52.00 | 70.2 | 75.2 | 77.00 | 42.5 | 47.1 | 84.3 | 61.53 |
| PSPNet (R101) | 68.93 | 89.75 | 82.16 | 91.18 | 92.00 | 89.55 | 46.15 | 64.19 | 93.29 | 79.69 |

## Notes for the paper

- **Defensible claim:** "improves over baselines we trained under identical
  conditions, especially on the hard classes." Do not claim to beat 79.69 unless
  the number actually does. The setups differ (resolution, backbone), so even
  a win is not a clean comparison.
- **State the resolution.** Images are downsampled ~3.9× (4000→1024), and vehicles
  shrink with them. That is the main reason vehicle IoU will be low, and an honest
  limitation (future work: tiled full-resolution training).
- **Label noise:** other work (LightFormer, arXiv 2504.10834) reports missed or
  mislabelled vehicles in FloodNet annotations. Cite this when discussing vehicle IoU.
- Before writing "no prior work does X", do a proper search (Scholar, IEEE, arXiv)
  and phrase it as "to the best of our knowledge".
- `floodnet_stats.json` from notebook 1 gives the class pixel shares, i.e. the
  class-imbalance table that motivates the hybrid loss.

## Use in the main pipeline

The best checkpoint (`runs/<best run>/best.pt`) becomes the flood-severity model.
It stores the run name, config, class list and frame size, so it can be rebuilt
with `floodnet_train.build_model(config, weights=None)` and `load_state_dict`.

## Citation

> M. Rahnemoonfar, T. Chowdhury, A. Sarkar, D. Varshney, M. Yari, and R. R. Murphy,
> "FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene
> Understanding," IEEE Access, vol. 9, pp. 89644–89654, 2021.
