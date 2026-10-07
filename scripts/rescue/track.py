"""
Module 2 - tracking: ByteTrack (the implementation bundled with Ultralytics).

Not trained. ByteTrack matches boxes frame to frame with a Kalman filter and
IoU, in two passes: confident detections first, then low-confidence ones,
which is what keeps a half-submerged person tracked when their score dips.
Running it here (instead of model.track) lets it take SAHI detections too.
"""

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from .detect import Dets


class Tracker:
    def __init__(self, fps=30.0, high=0.25, low=0.1, new=0.25, buffer_s=3.0, match=0.8):
        frames = max(1, int(round(buffer_s * fps)))
        args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=high, track_low_thresh=low,
                               new_track_thresh=new, track_buffer=frames,
                               match_thresh=match, fuse_score=True)
        self.bt = BYTETracker(args)
        # Older Ultralytics rescales the buffer by frame_rate/30; pin it to frames.
        self.bt.max_time_lost = self.bt.max_frames_lost = frames

    def update(self, dets: Dets, frame):
        """Returns confirmed tracks as an (N, 7) array: x1, y1, x2, y2, track_id, score, cls."""
        out = np.asarray(self.bt.update(dets, frame), np.float32)
        return out[:, :7] if out.ndim == 2 and len(out) else np.zeros((0, 7), np.float32)
