| Method | AP50 | AP50-95 | AP75 | AP_small | AP_medium | AR@300 | sec_per_image |
|---|---|---|---|---|---|---|---|
| YOLO | 0.404 | 0.178 | 0.128 | 0.152 | 0.256 | 0.335 | 1.19 |
| YOLO + SAHI (320px tiles) | 0.326 | 0.139 | 0.102 | 0.116 | 0.206 | 0.306 | 5.92 |

109 real flood photos (1330 people), COCO protocol via pycocotools, conf >= 0.01, imgsz 960. AP_small = people under 32x32 px. sec_per_image depends on the GPU.
