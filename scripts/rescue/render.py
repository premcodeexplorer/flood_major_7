"""
Outputs: annotated frame + priority panel (the "rescue-priority map"), drawn with OpenCV.

Positions are image positions (pixels), not GPS - the datasets carry no drone
telemetry. With GPS + camera pose, the same group centres would project to
lat/long; that is a deployment step, noted in the project document.
"""

import cv2
import numpy as np

from .severity import colorize

LEVEL_BGR = {"Critical": (40, 40, 230), "High": (0, 140, 255), "Medium": (0, 215, 255), "Low": (80, 190, 60)}
VEHICLE_BGR = (230, 200, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale, color, thick=1, bg=None):
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_frame(frame, persons, others, groups, seg=None, overlay_alpha=0.35, t=None):
    img = frame.copy()
    h, w = img.shape[:2]
    if seg is not None and overlay_alpha > 0:
        img = cv2.addWeighted(img, 1 - overlay_alpha, colorize(seg), overlay_alpha, 0)
    s = max(0.45, w / 2200)
    lw = max(1, int(round(w / 900)))

    for x1, y1, x2, y2, *_ in others:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), VEHICLE_BGR, lw)
    level_of = {tid: g.level for g in groups for tid in g.members}
    for x1, y1, x2, y2, tid, *_ in persons:
        c = LEVEL_BGR[level_of.get(int(tid), "Low")]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, lw)

    for rank, g in enumerate(groups, 1):
        c = LEVEL_BGR[g.level]
        pad = 0.5 * g.person_size
        x1, y1, x2, y2 = g.bbox + np.array([-pad, -pad, pad, pad])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, lw + 1)
        label_y = y1 - 4 if y1 - 4 > 22 * s / 0.45 else y2 + 20 * s / 0.45   # below the box at the top edge
        _text(img, f"#{rank} {g.level} {g.risk:.2f} ({g.count})", (max(2, x1), label_y), s, (255, 255, 255),
              max(1, lw - 1), bg=c)

    n_people = int(sum(g.count for g in groups))
    hud = f"people {n_people}   groups {len(groups)}"
    if t is not None:
        hud = f"t={t:6.1f}s   " + hud
    _text(img, hud, (10, h - 10), s * 1.1, (255, 255, 255), lw, bg=(30, 30, 30))
    return img


def priority_panel(img, groups, title="Rescue priority", width=None, max_rows=12, use_memory=True):
    """Appends a ranked table to the right of the image."""
    h, w = img.shape[:2]
    pw = width or max(380, int(w * 0.32))
    panel = np.full((h, pw, 3), 28, np.uint8)
    s = pw / 700
    y = int(40 * s / 0.55)
    _text(panel, title, (14, y), s * 1.15, (255, 255, 255), 2)
    y += int(36 * s / 0.55)
    cols = [("#", 0.0), ("level", 0.06), ("risk", 0.30), ("ppl", 0.42), ("flood", 0.53),
            ("stuck", 0.67), ("isol.", 0.82)]
    for name, fx in cols:
        _text(panel, name, (32 + fx * (pw - 40), y), s * 0.8, (170, 170, 170), 1)
    y += int(8 * s / 0.55)
    row_h = int(34 * s / 0.55)
    for rank, g in enumerate(groups[:max_rows], 1):
        y += row_h
        f = g.features
        cells = [str(rank), g.level, f"{g.risk:.2f}", str(g.count), f"{f.get('severity', 0):.0%}",
                 f"{f.get('stranded_s', 0):.0f}s" if use_memory else "-", f"{f.get('isolation', 0):.2f}"]
        cv2.rectangle(panel, (14, y - int(16 * s / 0.55)), (24, y + 2), LEVEL_BGR[g.level], -1)
        for (_, fx), cell in zip(cols, cells):
            _text(panel, cell, (32 + fx * (pw - 40), y), s * 0.8, (235, 235, 235), 1)
    if len(groups) > max_rows:
        _text(panel, f"... {len(groups) - max_rows} more", (32, y + row_h), s * 0.8, (170, 170, 170), 1)
    y = h - int(20 * s / 0.55)
    for i, (name, c) in enumerate(LEVEL_BGR.items()):
        x = 14 + i * (pw - 14) // 4
        cv2.rectangle(panel, (x, y - int(14 * s / 0.55)), (x + 12, y), c, -1)
        _text(panel, name, (x + 18, y), s * 0.75, (220, 220, 220), 1)
    return np.hstack([img, panel])
