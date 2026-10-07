"""
Module 1 - victim detection: Model 1 (YOLOv8, flood_rehearsal.pt), optionally with SAHI.

SAHI cuts the frame into overlapping tiles, runs the detector on each tile and
on the full frame, and merges everything with NMS. A person 15 px tall in the
frame becomes ~3x larger inside a tile, which is where plain YOLO loses them.

Merging uses NMS on IoU, not SAHI's default (greedy merge on intersection over
the smaller box): the default fuses neighbours in a crowd into one big box,
and the head count of a group matters for the risk score.
"""

import numpy as np
from ultralytics import YOLO

PERSON, VEHICLE, ANIMAL = 0, 1, 2
NAMES = {PERSON: "person", VEHICLE: "vehicle", ANIMAL: "animal"}


class Dets:
    """Detections in one frame. Indexable the way ByteTrack expects (conf / cls / xywh)."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.asarray(cls, np.float32).reshape(-1)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return Dets(self.xyxy[i], self.conf[i], self.cls[i])


class Detector:
    def __init__(self, weights, imgsz=960, conf=0.25, device=None,
                 sahi=False, slice_size=512, overlap=0.2, nms_iou=0.5):
        self.imgsz, self.conf, self.sahi = imgsz, conf, sahi
        self.slice_size, self.overlap, self.nms_iou = slice_size, overlap, nms_iou
        self.device = device
        if sahi:
            from sahi import AutoDetectionModel
            self.model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(weights), confidence_threshold=conf,
                device=device or ("cuda:0" if _cuda() else "cpu"), image_size=imgsz)
        else:
            self.model = YOLO(str(weights))

    def __call__(self, frame_bgr) -> Dets:
        if not self.sahi:
            r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                   max_det=1000, verbose=False)[0].boxes.cpu().numpy()
            return Dets(r.xyxy, r.conf, r.cls)

        from sahi.predict import get_sliced_prediction
        res = get_sliced_prediction(
            np.ascontiguousarray(frame_bgr[:, :, ::-1]), self.model,      # SAHI wants RGB
            slice_height=self.slice_size, slice_width=self.slice_size,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
            perform_standard_pred=True, postprocess_type="NMS",
            postprocess_match_metric="IOU", postprocess_match_threshold=self.nms_iou,
            verbose=0)
        p = res.object_prediction_list
        if not p:
            return Dets(np.zeros((0, 4)), [], [])
        return Dets([o.bbox.to_xyxy() for o in p], [o.score.value for o in p],
                    [o.category.id for o in p])


def _cuda():
    import torch
    return torch.cuda.is_available()
