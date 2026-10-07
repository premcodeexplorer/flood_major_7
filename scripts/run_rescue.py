"""
Phases 3-5: run the full rescue pipeline on drone footage or photos.

    detect (Model 1 [+ SAHI]) -> track (ByteTrack) -> group -> memory
    -> flood severity (Model 2) -> risk score -> priority map / table / alerts

Inputs:
  * a video                    -> tracked over time; memory gives time stranded and water trend
  * a folder of frames, --sequence  -> same, frames in name order at --fps
  * a photo or folder of photos -> each photo on its own (no tracking, no memory)

Outputs in --out:
  annotated.mp4 / annotated/*.jpg   frames with boxes, groups, flood overlay and the priority panel
  priority_map.jpg                  the last (video) or each (photo) frame with the ranked table
  victims.csv                       one row per group: rank, people, position, every factor, risk, level
  tracks.csv                        (video) every person box per frame with its track and group
  alerts.json                       (video) when a group first reached High / Critical
  summary.json                      settings, counts, speed

Usage:
    python scripts/run_rescue.py --source flood.mp4 --out outputs/rescue/demo
    python scripts/run_rescue.py --source data/flood_c2a/test --out outputs/rescue/test_photos
    python scripts/run_rescue.py --source flood.mp4 --sahi --no-memory   # ablation
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from rescue.detect import Detector
from rescue.pipeline import RescuePipeline
from rescue.render import draw_frame, priority_panel
from rescue.severity import SeverityModel

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
GROUP_FIELDS = ["rank", "group", "people", "cx", "cy", "level", "risk", "severity", "isolation",
                "stranded_s", "trend_per_min", "f_size", "f_severity", "f_stranded", "f_isolation",
                "f_trend", "members"]


def frames_of(src: Path, sequence: bool, max_frames: int):
    """Yields (name, frame). Video -> sequence; folder -> photos unless --sequence."""
    if src.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(src))
        i = 0
        while i < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            yield f"{i:06d}", f
            i += 1
        cap.release()
        return
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    for p in files[:max_frames]:
        f = cv2.imread(str(p))
        if f is not None:
            yield p.stem, f


def group_rows(groups, name=None):
    rows = []
    for rank, g in enumerate(groups, 1):
        f = g.features
        r = dict(rank=rank, group=f"G{g.gid}", people=g.count, cx=round(float(g.center[0]), 1),
                 cy=round(float(g.center[1]), 1), level=g.level, risk=round(g.risk, 3),
                 members=" ".join(map(str, g.members)),
                 **{k: round(float(f[k]), 3) for k in GROUP_FIELDS if k in f})
        if name is not None:
            r = {"image": name, **r}
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video, image, or folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default=str(ROOT / "models/flood_rehearsal.pt"))
    ap.add_argument("--severity", default=str(ROOT / "outputs/model2/runs/unet_r50_scse_hybrid/best.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sahi", action="store_true", help="tiled inference for small people")
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--sequence", action="store_true", help="treat an image folder as video frames")
    ap.add_argument("--fps", type=float, default=0, help="frame rate (default: the video's; 2 for frame folders)")
    ap.add_argument("--stride", type=int, default=1, help="process every n-th video frame")
    ap.add_argument("--max-frames", type=int, default=10**9)
    ap.add_argument("--severity-every", type=float, default=1.0, help="seconds between Model 2 runs")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--no-severity", action="store_true")
    ap.add_argument("--no-motion", action="store_true", help="assume a fixed camera")
    ap.add_argument("--overlay", type=float, default=0.35, help="flood-map opacity (0 = off)")
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.out)
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    out.mkdir(parents=True, exist_ok=True)
    video = src.suffix.lower() in VID_EXT or args.sequence

    fps = args.fps
    if src.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(src))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        fps = fps or src_fps / args.stride
        print(f"[video] {src.name}: {n_src} frames @ {src_fps:.1f} fps -> processing at {fps:.1f} fps")
    fps = fps or 2.0

    print(f"[model 1] {args.detector}" + ("  + SAHI" if args.sahi else ""))
    det = Detector(args.detector, imgsz=args.imgsz, conf=args.conf, device=args.device,
                   sahi=args.sahi, slice_size=args.slice)
    sev = None
    if not args.no_severity:
        sev = SeverityModel(args.severity, device=args.device)
        print(f"[model 2] {args.severity} ({sev.run}, {sev.device})")

    def new_pipeline():
        return RescuePipeline(det, sev, fps=fps, video=video, use_memory=not args.no_memory,
                              use_severity=not args.no_severity, use_motion=not args.no_motion,
                              severity_every_s=args.severity_every)

    pipe = new_pipeline()
    writer, last_img, last_groups = None, None, []
    group_csv, track_rows, alerts, best_level = [], [], [], {}
    rank_of_level = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    t0, n = time.time(), 0
    if not video:
        (out / "annotated").mkdir(exist_ok=True)

    raw = frames_of(src, args.sequence, args.max_frames * args.stride)
    for k, (name, frame) in enumerate(raw):
        if src.suffix.lower() in VID_EXT and k % args.stride:
            continue
        if not video:
            pipe = new_pipeline()                          # photos are independent
        r = pipe.step(frame)
        n += 1
        img = draw_frame(frame, r["persons"], r["others"], r["groups"], r["seg"],
                         overlay_alpha=args.overlay, t=r["t"] if video else None)
        if not args.no_panel:
            img = priority_panel(img, r["groups"], use_memory=pipe.use_memory)
        last_img, last_groups = img, r["groups"]

        if video:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
            writer.write(img)
            gid_of = {m: g.gid for g in r["groups"] for m in g.members}
            for x1, y1, x2, y2, tid, sc, cl in np.vstack([r["persons"], r["others"]]):
                track_rows.append(dict(frame=pipe.i, t=round(r["t"], 2), track=int(tid), cls=int(cl),
                                       score=round(float(sc), 3), x1=round(float(x1), 1), y1=round(float(y1), 1),
                                       x2=round(float(x2), 1), y2=round(float(y2), 1),
                                       group=f"G{gid_of[int(tid)]}" if int(tid) in gid_of else ""))
            for g in r["groups"]:
                if rank_of_level[g.level] >= 2 and rank_of_level[g.level] > best_level.get(g.gid, -1):
                    alerts.append(dict(t=round(r["t"], 2), frame=pipe.i, group=f"G{g.gid}", level=g.level,
                                       risk=round(g.risk, 3), people=g.count,
                                       position=[round(float(v), 1) for v in g.center],
                                       severity=round(g.features["severity"], 3),
                                       stranded_s=round(g.features["stranded_s"], 1)))
                    print(f"  ALERT t={r['t']:.1f}s  G{g.gid} -> {g.level} (risk {g.risk:.2f}, {g.count} people)")
                best_level[g.gid] = max(best_level.get(g.gid, -1), rank_of_level[g.level])
            if n % 50 == 0:
                print(f"  frame {n}  {n / (time.time() - t0):.2f} fps  people {len(r['persons'])}  "
                      f"groups {len(r['groups'])}")
        else:
            cv2.imwrite(str(out / "annotated" / f"{name}.jpg"), img)
            group_csv += group_rows(r["groups"], name)
            print(f"  {name[:40]:40s} people {len(r['persons']):3d}  groups {len(r['groups']):2d}  "
                  f"top {r['groups'][0].level if r['groups'] else '-'}")

    if n == 0:
        raise SystemExit("no frames read")
    elapsed = time.time() - t0
    if writer is not None:
        writer.release()
    cv2.imwrite(str(out / "priority_map.jpg"), last_img)

    if video:
        group_csv = group_rows(last_groups)
        _write_csv(out / "tracks.csv", track_rows)
        (out / "alerts.json").write_text(json.dumps(alerts, indent=2))
    _write_csv(out / "victims.csv", group_csv)

    summary = dict(source=str(src), mode="video" if video else "photos", frames=n,
                   seconds=round(elapsed, 1), fps_processed=round(n / elapsed, 2),
                   detector=args.detector, sahi=args.sahi, severity=None if args.no_severity else args.severity,
                   use_memory=pipe.use_memory, use_severity=pipe.use_severity,
                   camera_motion=not args.no_motion and video,
                   motion_failures=pipe.motion.failures if video else None,
                   alerts=len(alerts), final_groups=len(last_groups),
                   final_levels={lv: sum(g.level == lv for g in last_groups) for lv in rank_of_level})
    if video:
        summary["unique_tracks"] = len({row["track"] for row in track_rows if row["cls"] == 0})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] {out}")


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
