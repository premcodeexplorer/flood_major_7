"""
Phase 5 evaluation: does the risk score rank groups the way people would?

There is no public benchmark of "who should be rescued first", so this uses
human judgement, the standard fallback:

  1. `prepare`: runs Models 1 + 2 on photos that contain 2+ groups and draws
     every group with a number, in RANDOM order (no risk shown, so the
     raters are not anchored). Writes sheet/<photo>.jpg and labels.csv.
  2. Each rater opens the sheets and fills in their column of labels.csv
     with the group number they would send the boat to FIRST (0 = no box
     contains real people; see priority_eval/README_RATERS.md).
  3. `score`: compares every ranking variant with the raters' choices:
       full          all factors (single photos have no memory)
       no_severity   people count only (Model 2 switched off)
       severity_only flood factors only (severity + isolation)
       random        expected top-1 for a random pick = mean(1 / groups)
     Reports top-1 agreement with the majority vote, and how often the raters
     agree with each other (the ceiling any method can reach).

Memory (time stranded, water trend) needs video and is not covered here;
its effect is shown on the demo video instead.

Usage:
    python scripts/eval_priority.py prepare --images data/flood_c2a/test --out outputs/rescue/priority_eval
    # raters fill labels.csv columns rater1, rater2, rater3
    python scripts/eval_priority.py score      # plain Python + numpy + opencv is enough
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from rescue import risk          # models are imported inside prepare(), so `score` runs anywhere

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {
    "full": dict(use_memory=False, use_severity=True),
    "no_severity": dict(use_memory=False, use_severity=False),
    "severity_only": dict(weights=dict(severity=0.75, isolation=0.25)),
}


def prepare(args):
    from rescue.detect import Detector
    from rescue.pipeline import RescuePipeline
    from rescue.render import _text
    from rescue.severity import SeverityModel
    det = Detector(args.detector, imgsz=args.imgsz, conf=args.conf, device=args.device)
    sev = SeverityModel(args.severity, device=args.device)
    out = Path(args.out)
    (out / "sheet").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    rows, groups_json = [], {}
    for p in files:
        frame = cv2.imread(str(p))
        r = RescuePipeline(det, sev, video=False).step(frame)
        gs = [g for g in r["groups"]]
        if len(gs) < 2:
            continue
        rng.shuffle(gs)                                   # numbering carries no information
        img = frame.copy()
        s = max(0.6, frame.shape[1] / 1100)
        for k, g in enumerate(gs, 1):
            pad = 0.5 * g.person_size
            x1, y1, x2, y2 = (g.bbox + np.array([-pad, -pad, pad, pad])).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            _text(img, str(k), (x1 + 2, max(18, y1 - 4)), s, (0, 0, 0), 2, bg=(255, 255, 255))
        cv2.imwrite(str(out / "sheet" / f"{p.stem}.jpg"), img)
        groups_json[p.stem] = [dict(number=k, people=g.count, **{f: g.features[f] for f in
                                    ("severity", "isolation")}) for k, g in enumerate(gs, 1)]
        rows.append(dict(photo=p.stem, groups=len(gs), rater1="", rater2="", rater3=""))
        if len(rows) >= args.max_photos:
            break
    (out / "groups.json").write_text(json.dumps(groups_json, indent=1))
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[prepare] {len(rows)} photos with 2+ groups -> {out / 'sheet'}")
    print(f"Raters: open each sheet, write the number of the group to rescue FIRST in labels.csv.")


def rank_top(groups, variant):
    v = VARIANTS[variant]
    def sc(g):
        f = risk.factors(g["people"], g["severity"], 0, g["isolation"], 0)
        if "weights" in v:
            return risk.score(f, weights=v["weights"])
        return risk.score(f, v["use_memory"], v["use_severity"])
    best = max(sc(g) for g in groups)
    return {g["number"] for g in groups if abs(sc(g) - best) < 1e-9}       # ties count as a hit


def score(args):
    out = Path(args.out)
    groups = json.loads((out / "groups.json").read_text())
    rows = list(csv.DictReader(open(out / "labels.csv")))
    raters = [c for c in rows[0] if c.startswith("rater")]
    used, hits, chance, pair_agree, pairs, no_people = 0, Counter(), 0.0, 0, 0, 0
    for r in rows:
        votes = [int(r[c]) for c in raters if r[c].strip()]
        if not votes:
            continue
        top, n = Counter(votes).most_common(1)[0]
        if list(Counter(votes).values()).count(n) > 1:
            continue                                           # no majority -> skip the photo
        if top == 0:
            no_people += 1                                     # majority: no real people in any box
            continue
        used += 1
        for v in VARIANTS:
            top_set = rank_top(groups[r["photo"]], v)
            hits[v] += (top in top_set) / len(top_set)        # a tie of k groups earns 1/k
        chance += 1 / len(groups[r["photo"]])
        for i in range(len(votes)):
            for j in range(i + 1, len(votes)):
                pairs += 1
                pair_agree += votes[i] == votes[j]
    if not used:
        raise SystemExit("no labelled photos with a majority vote yet")
    res = dict(photos=used, skipped_no_real_people=no_people, raters=raters,
               top1_vs_majority={v: round(hits[v] / used, 3) for v in VARIANTS},
               random_top1=round(chance / used, 3),
               rater_pairwise_agreement=round(pair_agree / pairs, 3) if pairs else None)
    (out / "priority_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "score"])
    ap.add_argument("--images", default=str(ROOT / "data/flood_c2a/test"))
    ap.add_argument("--out", default=str(ROOT / "outputs/rescue/priority_eval"))
    ap.add_argument("--detector", default=str(ROOT / "models/flood_rehearsal.pt"))
    ap.add_argument("--severity", default=str(ROOT / "outputs/model2/runs/unet_r50_scse_hybrid/best.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-photos", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else score(args)


if __name__ == "__main__":
    main()
