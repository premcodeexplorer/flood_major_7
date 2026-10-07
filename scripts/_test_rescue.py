"""
Tests for the rescue pipeline's logic modules (no GPU, no weights needed):
grouping, memory (stranded time, water trend), risk formula and ablations,
severity lookups, camera-motion compensation.

    python scripts/_test_rescue.py
"""

import numpy as np

from rescue import risk
from rescue.memory import Memory, group_people
from rescue.motion import CameraMotion
from rescue.severity import CLASSES, SeverityMap


def box(cx, cy, s, tid):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, tid, 0.9, 0]


def test_grouping():
    t = np.array([box(100, 100, 20, 1), box(130, 100, 20, 2), box(160, 110, 20, 3),   # chain -> 1 group
                  box(600, 600, 20, 4)], np.float32)                                    # alone
    gs = sorted(group_people(t, link_factor=3.0), key=lambda g: -g.count)
    assert [g.count for g in gs] == [3, 1], [g.count for g in gs]
    assert gs[0].gid == 1 and set(gs[0].members) == {1, 2, 3}
    assert group_people(np.zeros((0, 7)), 3.0) == []
    crowd = np.array([box(100 + 30 * k, 100, 20, k + 1) for k in range(40)], np.float32)   # 1200 px line
    gs = group_people(crowd, link_factor=3.0, max_extent=12.0)
    assert sum(g.count for g in gs) == 40 and len(gs) > 1
    assert all(g.boxes[:, 2].max() - g.boxes[:, 0].min() <= 12 * 20 + 20 for g in gs)


def test_memory_stranded_and_trend():
    m = Memory(move_factor=1.0, trend_window_s=60, trend_min_s=10, ttl_s=5)
    for k in range(61):                       # still for 60 s, water rising 0.2 -> 0.5
        t = float(k)
        m.observe(7, t, (100 + 0.2 * np.sin(k), 100), 20)
        m.observe_severity(7, t, 0.2 + 0.005 * k)
    assert abs(m.stranded_s(7, 60) - 60) < 1e-6
    assert abs(m.trend_per_min(7, 60) - 0.3) < 1e-6          # 0.005/s = 0.3/min
    m.observe(7, 61, (200, 100), 20)                         # walks 5 body sizes
    assert m.stranded_s(7, 61) == 0
    m.forget(100)
    assert 7 not in m.tracks
    # re-identification by place: track 7 lost, track 9 appears on the same spot
    m = Memory(move_factor=1.0, ttl_s=10)
    for k in range(20):
        m.observe_frame(float(k), [7, 8], [(100, 100), (400, 100)], [20, 20])
    m.observe_frame(21.0, [9, 8], [(103, 98), (400, 100)], [20, 20])
    assert 7 not in m.tracks and abs(m.stranded_s(9, 21) - 21) < 1e-6
    m.observe_frame(22.0, [10, 8, 9], [(400, 104), (400, 100), (103, 98)], [20, 20, 20])
    assert m.stranded_s(10, 22) == 0                         # 8 is present -> 10 is a new person
    m2 = Memory()
    m2.observe(1, 0, (0, 0), 10)
    m2.observe_severity(1, 0, 0.5)
    assert m2.trend_per_min(1, 0) == 0.0                     # too little history -> no trend


def test_risk():
    hi = risk.factors(count=12, severity=0.9, stranded_s=600, isolation=1.0, trend_per_min=0.2)
    lo = risk.factors(count=1, severity=0.0, stranded_s=0, isolation=0.0, trend_per_min=-0.1)
    assert risk.score(hi) > 0.95 and risk.level(risk.score(hi)) == "Critical"
    assert risk.score(lo) < 0.05 and risk.level(risk.score(lo)) == "Low"
    assert abs(sum(risk.WEIGHTS.values()) - 1) < 1e-9
    f = risk.factors(count=5, severity=0.8, stranded_s=0, isolation=0.5, trend_per_min=0)
    # no memory: stranded/trend dropped and weights renormalised
    w = {k: v for k, v in risk.WEIGHTS.items() if k not in risk.MEMORY_FACTORS}
    expect = sum(w[k] * f[k] for k in w) / sum(w.values())
    assert abs(risk.score(f, use_memory=False) - expect) < 1e-9
    # no severity: only size + stranded remain
    assert abs(risk.score(f, use_severity=False) - (0.2 * 0.5) / (0.2 + 0.25)) < 1e-9
    assert [risk.level(x) for x in (0.81, 0.8, 0.5, 0.31, 0.3, 0.0)] == \
        ["Critical", "High", "Medium", "Medium", "Low", "Low"]


def test_severity_map():
    W = CLASSES.index("water")
    seg = np.full((200, 200), CLASSES.index("grass"), np.uint8)   # dry mainland on the left
    seg[:, 100:] = W
    seg[90:110, 160:180] = CLASSES.index("building-non-flooded")  # dry island (roof) in the water
    s = SeverityMap(seg, isolation_full=0.25)
    assert s.severity(185, 30, 10) == 1.0
    assert s.severity(20, 100, 15) == 0.0
    assert 0.4 < s.severity(100, 30, 30) < 0.6
    assert s.isolation(20, 100) == 0.0                                 # on mainland
    assert s.isolation(170, 100) > 0.8                                 # on the island: ~70 px of water
    assert s.isolation(120, 100) < s.isolation(170, 100)
    no_road = SeverityMap(np.full((50, 50), CLASSES.index("water"), np.uint8))
    assert no_road.isolation(10, 10) == 1.0
    bg = SeverityMap(np.zeros((50, 50), np.uint8))           # all background -> no evidence
    assert bg.severity(25, 25, 10) == 0.0


def test_camera_motion():
    rng = np.random.default_rng(0)
    base = (rng.random((480, 640)) * 255).astype(np.uint8)
    base = np.dstack([cv2_blur(base)] * 3)
    cm = CameraMotion(work_width=640)
    cm.update(base)
    M = np.float32([[1, 0, -12], [0, 1, 7]])                 # camera moved: scene shifted by (-12, +7)
    import cv2
    moved = cv2.warpAffine(base, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
    cm.update(moved)
    p = cm.to_scene([[300, 200]])[0]                         # pixel in moved frame -> first frame
    assert np.allclose(p, [312, 193], atol=0.6), p


def cv2_blur(x):
    import cv2
    return cv2.GaussianBlur(x, (0, 0), 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all rescue tests passed")
