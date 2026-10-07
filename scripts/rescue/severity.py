"""
Module 4 - flood severity: Model 2 (U-Net R50 + scSE, trained on FloodNet).

The segmenter labels every pixel with one of FloodNet's 10 classes. Two
numbers are read from that map around each rescue group:

  severity   share of FLOODED pixels (water, flooded road, flooded building)
             in a disk around the group, ignoring background. 0 = dry
             surroundings, 1 = surrounded by water. It says how much water
             is around the group, NOT how deep it is - depth cannot be read
             from a single RGB image.
  isolation  distance from the group to the nearest "mainland": dry ground
             (dry road, grass, dry building, unlabelled background) that is
             connected to the edge of the view, i.e. leads somewhere. As a
             fraction of `isolation_full` x frame diagonal, capped at 1.
             On mainland -> 0. On a dry island or a roof surrounded by
             water -> grows with the water between them and dry land.
             No mainland anywhere in view -> 1.

FloodNet is nadir imagery from ~200 ft after Hurricane Harvey. On oblique or
very different footage the map gets worse; check the overlay before trusting
the numbers.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CLASSES = ["background", "building-flooded", "building-non-flooded", "road-flooded",
           "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]
FLOODED = [CLASSES.index(c) for c in ("building-flooded", "road-flooded", "water")]
DRY = [CLASSES.index(c) for c in ("background", "building-non-flooded", "road-non-flooded", "grass")]
BACKGROUND = 0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PALETTE = np.array([[0, 0, 0], [255, 0, 0], [180, 120, 120], [160, 150, 20], [140, 140, 140],
                    [61, 230, 250], [0, 82, 255], [255, 0, 245], [255, 235, 0], [4, 250, 7]],
                   np.uint8)   # RGB, same as floodnet_train.py


def build_model(cfg):
    import segmentation_models_pytorch as smp
    kw = dict(encoder_name=cfg["encoder"], encoder_weights=None, classes=len(CLASSES))
    if cfg["arch"] == "Unet":
        return smp.Unet(decoder_attention_type=cfg["attention"], **kw)
    return getattr(smp, cfg["arch"])(**kw)


class SeverityModel:
    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("classes", CLASSES) != CLASSES:
            raise SystemExit(f"{ckpt_path}: class list differs from FloodNet's")
        self.model = build_model(ck["config"]).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.frame = tuple(ck.get("frame", (1024, 768)))     # (W, H) it was trained at
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.run = ck.get("run", "?")

    @torch.no_grad()
    def segment(self, frame_bgr):
        """Class-id map (H, W) uint8 at the frame's own resolution."""
        h, w = frame_bgr.shape[:2]
        W, H = self.frame
        if (h > w) != (H > W):            # portrait footage: segment it as landscape
            W, H = H, W
        x = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        x = x.half() if self.half else x
        logits = self.model(x)
        pred = logits.argmax(1)[0].byte().cpu().numpy()
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


class SeverityMap:
    """Precomputed lookups on one segmentation, so each group costs O(disk area)."""

    def __init__(self, seg, isolation_full=0.25):
        self.seg = seg
        self.flooded = np.isin(seg, FLOODED)
        self.valid = seg != BACKGROUND
        diag = float(np.hypot(*seg.shape))
        self.norm = isolation_full * diag
        main = self._mainland(seg, diag)
        # distance from every pixel to the nearest mainland pixel
        self.dist = cv2.distanceTransform(1 - main, cv2.DIST_L2, 5) if main.any() else None
        self.counts = np.bincount(seg.ravel(), minlength=len(CLASSES))

    @staticmethod
    def _mainland(seg, diag):
        dry = np.isin(seg, DRY).astype(np.uint8)
        k = max(3, int(diag * 0.01) | 1)          # bridge people, cars, canopy over a road
        dry = cv2.morphologyEx(dry, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab = cv2.connectedComponents(dry, connectivity=8)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        return np.isin(lab, edge[edge > 0]).astype(np.uint8)

    def severity(self, cx, cy, radius):
        h, w = self.seg.shape
        r = int(max(4, radius))
        x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        valid = self.valid[y0:y1, x0:x1] & disk
        n = valid.sum()
        return float((self.flooded[y0:y1, x0:x1] & valid).sum() / n) if n else 0.0

    def isolation(self, cx, cy):
        if self.dist is None:
            return 1.0
        h, w = self.seg.shape
        d = self.dist[min(h - 1, max(0, int(cy))), min(w - 1, max(0, int(cx)))]
        return float(min(1.0, d / self.norm))

    def flooded_share(self):
        n = self.counts[1:].sum()
        return float(self.counts[FLOODED].sum() / n) if n else 0.0


def colorize(seg):
    """BGR colour image of a class map."""
    return PALETTE[seg][:, :, ::-1].copy()
