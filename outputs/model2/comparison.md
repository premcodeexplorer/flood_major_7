| Model | mIoU (9) | Bld-F | Bld-NF | Road-F | Road-NF | Water | Tree | Vehicle | Pool | Grass | Hard-4 | Params (M) | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ENet [FloodNet] * | 42.61 | 6.94 | 47.35 | 12.49 | 48.43 | 48.95 | 68.36 | 32.26 | 42.49 | 76.23 | 23.55 | - | - |
| DeepLabV3+ [FloodNet] * | 61.53 | 32.70 | 72.80 | 52.00 | 70.20 | 75.20 | 77.00 | 42.50 | 47.10 | 84.30 | 43.58 | - | - |
| PSPNet-R101 [FloodNet] * | 79.69 | 68.93 | 89.75 | 82.16 | 91.18 | 92.00 | 89.55 | 46.15 | 64.19 | 93.29 | 65.36 | - | - |
| U-Net (R50) | 71.72 | 73.70 | 79.49 | 54.65 | 82.14 | 69.46 | 80.97 | 55.36 | 61.98 | 87.70 | 61.42 | 32.52 | 24.7 |
| DeepLabV3+ (R50) | 73.44 | 73.60 | 80.12 | 63.46 | 83.18 | 75.68 | 81.39 | 53.54 | 61.28 | 88.69 | 62.97 | 26.68 | 34.6 |
| PSPNet (R50) | 69.57 | 69.99 | 75.35 | 57.91 | 78.92 | 74.88 | 81.89 | 43.60 | 55.47 | 88.10 | 56.74 | 24.34 | 62.6 |
| **Ours**: U-Net (R50) + scSE + Dice/Focal | 75.67 | 73.67 | 80.06 | 63.96 | 83.01 | 75.66 | 82.60 | 65.14 | 68.02 | 88.90 | 67.70 | 33.82 | 17.9 |

\* Published, FloodNet Table 4: 713x713 crops, PSPNet uses ResNet-101, different training setup - context only, not a like-for-like comparison.
All other rows: trained by us on the official FloodNet split (1445/450/448), frames 1024x768, 512px crops, 80 epochs, identical schedule; test split. mIoU (9) excludes background, as in the paper. Hard-4 = mean IoU of Bld-F, Road-F, Vehicle, Pool. FPS: batch 1, 1024x768, fp16 on Tesla T4.