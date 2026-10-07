"""
One frame through all five modules. Used by run_rescue.py and the evaluation scripts.

Model 2 is the slow part (a ResNet-50 U-Net at 1024x768), and flood water does
not change in a fraction of a second, so it runs every `severity_every_s`
seconds. Between runs the last map is reused: group positions are carried
into that older frame through the camera-motion transform, so a drifting
drone still reads the water at the right place.
"""

import numpy as np

from . import risk
from .detect import ANIMAL, PERSON, VEHICLE, Dets
from .memory import Memory, group_people
from .motion import CameraMotion
from .severity import SeverityMap
from .track import Tracker


class RescuePipeline:
    def __init__(self, detector, severity_model=None, fps=30.0, video=True,
                 use_memory=True, use_severity=True, use_motion=True,
                 severity_every_s=1.0, link_factor=3.0, radius_factor=4.0,
                 isolation_full=0.25, memory_kw=None):
        self.detector, self.sev_model = detector, severity_model
        self.fps, self.video = fps, video
        self.use_memory = use_memory and video          # one photo has no history
        self.use_severity = use_severity and severity_model is not None
        self.link_factor, self.radius_factor = link_factor, radius_factor
        self.isolation_full = isolation_full
        self.sev_every = max(1, int(round(severity_every_s * fps))) if video else 1
        self.tracker = Tracker(fps=fps) if video else None
        self.motion = CameraMotion(enabled=use_motion and video)
        self.memory = Memory(**(memory_kw or {}))
        self.smap, self.seg, self.seg_T = None, None, np.eye(3)
        self.i = -1

    # ------------------------------------------------------------------ helpers
    def _to_seg_frame(self, pts):
        """Current-frame points -> coordinates of the frame the severity map came from."""
        scene = self.motion.to_scene(pts)
        inv = np.linalg.inv(self.seg_T)
        return scene @ inv[:2, :2].T + inv[:2, 2]

    def _tracks(self, frame, dets: Dets):
        if self.tracker is not None:
            if not self.motion.enabled:
                return self.tracker.update(dets, frame)
            # Track in scene coordinates: ByteTrack matches by box overlap, and a
            # 15 px person shifts more than their own width when the drone pans.
            # Compensated, a still person stays put and the Kalman filter only
            # has to follow real movement. Boxes go back to frame coordinates after.
            z = self.motion.zoom
            c = self.motion.to_scene(dets.xywh[:, :2]) if len(dets) else np.zeros((0, 2))
            wh = dets.xywh[:, 2:] * z
            scene = Dets(np.hstack([c - wh / 2, c + wh / 2]), dets.conf, dets.cls)
            t = self.tracker.update(scene, frame)
            if len(t):
                inv = np.linalg.inv(self.motion.T)
                cc = (t[:, :2] + t[:, 2:4]) / 2
                cc = cc @ inv[:2, :2].T + inv[:2, 2]
                hw = (t[:, 2:4] - t[:, :2]) / (2 * z)
                t[:, :4] = np.hstack([cc - hw, cc + hw])
            return t
        ids = np.arange(1, len(dets) + 1, dtype=np.float32)
        return np.column_stack([dets.xyxy, ids, dets.conf, dets.cls]) if len(dets) else np.zeros((0, 7))

    # ------------------------------------------------------------------ main
    def step(self, frame):
        self.i += 1
        t = self.i / self.fps
        T = self.motion.update(frame)
        dets = self.detector(frame)
        tracks = self._tracks(frame, dets)
        persons = tracks[tracks[:, 6] == PERSON]
        others = tracks[np.isin(tracks[:, 6], [VEHICLE, ANIMAL])]

        fresh = False
        if self.use_severity and (self.smap is None or self.i % self.sev_every == 0):
            self.seg = self.sev_model.segment(frame)
            self.smap = SeverityMap(self.seg, self.isolation_full)
            self.seg_T, fresh = T.copy(), True

        if len(persons):
            b = persons[:, :4]
            centres = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
            sizes = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * self.motion.zoom
            self.memory.observe_frame(t, persons[:, 4], self.motion.to_scene(centres), sizes)

        groups = group_people(persons, self.link_factor)
        for g in groups:
            sev = iso = 0.0
            if self.smap is not None:
                cx, cy = self._to_seg_frame(g.center)[0]
                radius = max(self.radius_factor * g.person_size, g.spread + g.person_size)
                sev, iso = self.smap.severity(cx, cy, radius), self.smap.isolation(cx, cy)
                if fresh:
                    for tid in g.members:
                        self.memory.observe_severity(tid, t, sev)
            stranded = float(np.median([self.memory.stranded_s(m, t) for m in g.members]))
            trend = float(np.mean([self.memory.trend_per_min(m, t) for m in g.members]))
            f = risk.factors(g.count, sev, stranded, iso, trend)
            g.risk = risk.score(f, self.use_memory, self.use_severity)
            g.level = risk.level(g.risk)
            g.features = dict(severity=sev, isolation=iso, stranded_s=stranded,
                              trend_per_min=trend, **{f"f_{k}": v for k, v in f.items()})
        groups.sort(key=lambda g: -g.risk)
        self.memory.forget(t)
        return dict(t=t, persons=persons, others=others, groups=groups, dets=dets,
                    seg=self.seg if self.use_severity else None)
