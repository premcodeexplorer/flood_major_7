"""
Camera-motion compensation.

A drone drifts and pans. In raw pixels a person sitting still on a roof looks
like they are moving, and the water around them seems to change because the
view changed. So every frame is registered to the first one: corners are
tracked with Lucas-Kanade optical flow and a similarity transform (shift,
rotation, scale) is fitted with RANSAC. Chaining those gives a transform from
the current frame into "scene" coordinates (the first frame), in which
stillness and the water trend are measured.

Valid for a mostly flat scene seen from above, which is what a flood plain
from drone height is. It fails on fast pans with little texture (open water);
then the last good transform is kept and the frame is counted as unreliable.
"""

import cv2
import numpy as np


class CameraMotion:
    def __init__(self, enabled=True, work_width=640, max_corners=400, min_inliers=30):
        self.enabled, self.work_width = enabled, work_width
        self.max_corners, self.min_inliers = max_corners, min_inliers
        self.prev = None
        self.scale = 1.0
        self.T = np.eye(3)            # current frame -> scene (first frame)
        self.failures = 0

    def update(self, frame_bgr):
        if not self.enabled:
            return self.T
        h, w = frame_bgr.shape[:2]
        self.scale = min(1.0, self.work_width / w)
        g = cv2.cvtColor(cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            step = self._step(self.prev, g)
            if step is None:
                self.failures += 1
            else:
                S = np.diag([self.scale, self.scale, 1.0])
                # step maps current -> previous in the downscaled image; lift to full res
                self.T = self.T @ (np.linalg.inv(S) @ step @ S)
        self.prev = g
        return self.T

    def _step(self, prev, cur):
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8)
        if p0 is None or len(p0) < self.min_inliers:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None)
        ok = st.reshape(-1) == 1
        if ok.sum() < self.min_inliers:
            return None
        M, inl = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl is None or inl.sum() < self.min_inliers:
            return None
        return np.vstack([M, [0, 0, 1]])

    def to_scene(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        return pts @ self.T[:2, :2].T + self.T[:2, 2]

    @property
    def zoom(self):
        """Scene pixels per frame pixel (<1 after the camera zooms in or descends)."""
        return float(np.sqrt(abs(np.linalg.det(self.T[:2, :2]))))
