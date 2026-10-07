"""
Module 3 - short-term memory, and grouping people into rescue units.

Memory keeps a small state per tracked person (by ByteTrack ID), in scene
coordinates (see motion.py), so a single frame is never the whole story:

  stranded_s   seconds since the person last moved more than `move_factor`
               body sizes. Someone who has not moved for minutes is more
               likely stuck than someone walking out.
  trend        slope of the flood severity around them, in severity units
               per minute, over the last `trend_window_s` seconds. Positive
               = the water around them is rising (or spreading).
  seen_s       how long they have been in view.

Re-identification by place: ByteTrack gives a person a new ID when their
detection drops out for longer than its buffer (common for a half-submerged
person). In scene coordinates a still person stays at the same spot, so a
NEW track that appears within `move_factor` body sizes of a track that was
lost earlier inherits that track's memory (time stranded, water history)
instead of starting from zero.

A track that has been gone longer than `ttl_s` is forgotten. Not trained.
"""

from collections import deque

import numpy as np


class TrackState:
    __slots__ = ("first_t", "last_t", "anchor", "still_since", "size", "sev", "pos")

    def __init__(self, t, pos, size):
        self.first_t = self.last_t = self.still_since = t
        self.anchor = np.asarray(pos, np.float64)
        self.pos = self.anchor.copy()
        self.size = size
        self.sev = deque(maxlen=512)            # (t, severity)


class Memory:
    def __init__(self, move_factor=1.0, trend_window_s=60.0, trend_min_s=10.0, ttl_s=10.0):
        self.move_factor, self.trend_window_s = move_factor, trend_window_s
        self.trend_min_s, self.ttl_s = trend_min_s, ttl_s
        self.tracks: dict[int, TrackState] = {}

    def observe_frame(self, t, ids, scene_pos, scene_size):
        """All people of one frame. Known tracks first, so a new track can only
        inherit from a track that is NOT in this frame."""
        ids = [int(i) for i in ids]
        new = [k for k, tid in enumerate(ids) if tid not in self.tracks]
        for k, tid in enumerate(ids):
            if tid in self.tracks:
                self.observe(tid, t, scene_pos[k], scene_size[k])
        present = set(ids)
        for k in new:
            pos = np.asarray(scene_pos[k], np.float64)
            lost = [(np.linalg.norm(s.pos - pos), old) for old, s in self.tracks.items()
                    if old not in present and s.last_t < t
                    and np.linalg.norm(s.pos - pos) <= self.move_factor * max(s.size, scene_size[k])]
            if lost:
                old = min(lost)[1]
                self.tracks[ids[k]] = self.tracks.pop(old)
                present.add(old)
            self.observe(ids[k], t, pos, scene_size[k])

    def observe(self, tid, t, scene_pos, scene_size):
        s = self.tracks.get(tid)
        if s is None:
            self.tracks[tid] = TrackState(t, scene_pos, scene_size)
            return
        s.last_t, s.pos = t, np.asarray(scene_pos, np.float64)
        s.size = 0.8 * s.size + 0.2 * scene_size           # smooth box jitter
        if np.linalg.norm(s.pos - s.anchor) > self.move_factor * s.size:
            s.anchor, s.still_since = s.pos.copy(), t

    def observe_severity(self, tid, t, value):
        if tid in self.tracks:
            self.tracks[tid].sev.append((t, value))

    def forget(self, t):
        for tid in [k for k, s in self.tracks.items() if t - s.last_t > self.ttl_s]:
            del self.tracks[tid]

    def stranded_s(self, tid, t):
        s = self.tracks.get(tid)
        return t - s.still_since if s else 0.0

    def seen_s(self, tid, t):
        s = self.tracks.get(tid)
        return t - s.first_t if s else 0.0

    def trend_per_min(self, tid, t):
        s = self.tracks.get(tid)
        if s is None:
            return 0.0
        pts = [(ti, v) for ti, v in s.sev if t - ti <= self.trend_window_s]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < self.trend_min_s:
            return 0.0
        ts, vs = np.array(pts).T
        return float(np.polyfit(ts, vs, 1)[0] * 60.0)


class Group:
    """People close enough together to be rescued as one unit."""

    def __init__(self, members, boxes):
        self.members = members                          # track ids
        self.boxes = boxes                              # (n, 4) frame xyxy
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        self.center = np.array([cx.mean(), cy.mean()])
        sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        self.person_size = float(np.median(sizes))
        self.spread = float(np.max(np.hypot(cx - self.center[0], cy - self.center[1])))
        self.gid = min(members)
        self.features, self.risk, self.level = {}, 0.0, "Low"

    @property
    def count(self):
        return len(self.members)

    @property
    def bbox(self):
        return np.array([self.boxes[:, 0].min(), self.boxes[:, 1].min(),
                         self.boxes[:, 2].max(), self.boxes[:, 3].max()])


def group_people(tracks, link_factor=3.0, max_extent=12.0):
    """Single-linkage clustering: two people join when their centres are closer than
    link_factor x their mean body size. tracks: (N, 7) rows x1 y1 x2 y2 id score cls.

    Single linkage chains through a crowd, so a cluster wider than max_extent
    body sizes is split in two at the median of its longest axis, repeatedly:
    one rescue unit is people one boat or team can reach together."""
    n = len(tracks)
    if n == 0:
        return []
    b = tracks[:, :4]
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    size = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    link = link_factor * (size[:, None] + size[None, :]) / 2
    for i, j in zip(*np.nonzero(np.triu(d <= link, 1))):
        parent[find(i)] = find(j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    out, todo = [], [np.array(v) for v in clusters.values()]
    while todo:
        idx = todo.pop()
        pts = c[idx]
        span = pts.max(0) - pts.min(0)
        if len(idx) < 2 or span.max() <= max_extent * np.median(size[idx]):
            out.append(idx)
            continue
        ax = int(np.argmax(span))
        order = idx[np.argsort(pts[:, ax], kind="stable")]
        half = len(order) // 2
        todo += [order[:half], order[half:]]
    return [Group([int(tracks[i, 4]) for i in idx], b[idx]) for idx in out]
