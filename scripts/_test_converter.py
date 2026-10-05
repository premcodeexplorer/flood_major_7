"""Smoke test for visdrone_to_yolo.py against a synthetic VisDrone fixture."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp())

for split, folder in [("train", "VisDrone2019-DET-train"), ("val", "VisDrone2019-DET-val")]:
    d = tmp / folder
    (d / "images").mkdir(parents=True)
    (d / "annotations").mkdir(parents=True)
    Image.new("RGB", (1360, 765)).save(d / "images" / "0000001_00000_d_0000001.jpg")
    (d / "annotations" / "0000001_00000_d_0000001.txt").write_text(
        "\n".join([
            "100,100,20,40,1,1,0,0",      # pedestrian  -> person
            "200,200,18,35,1,2,0,0",      # people      -> person
            "300,300,80,60,1,4,0,0",      # car         -> vehicle
            "400,400,120,90,1,9,0,0",     # bus         -> vehicle
            "0,0,500,500,0,0,0,0",        # ignored region (score 0, cat 0) -> skip
            "600,600,30,30,1,11,0,0",     # others      -> skip
            "700,700,0,10,1,1,0,0",       # degenerate  -> skip
            "1340,700,100,200,1,4,0,0",   # overflows frame -> clamped, kept
            "500,500,25,25,0,1,0,0",      # score 0     -> skip
        ])
    )
    # a second image with no annotation file at all -> empty label, counted
    Image.new("RGB", (640, 480)).save(d / "images" / "0000002_00000_d_0000001.jpg")

r = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "visdrone_to_yolo.py"), "--root", str(tmp)],
    capture_output=True, text=True,
)
print(r.stdout, r.stderr)
assert r.returncode == 0, "converter crashed"

lbl = (tmp / "VisDrone2019-DET-train" / "labels" / "0000001_00000_d_0000001.txt").read_text()
lines = lbl.strip().split("\n")
print("labels:\n" + lbl)

assert len(lines) == 5, f"expected 5 kept boxes, got {len(lines)}"
assert [l.split()[0] for l in lines] == ["0", "0", "1", "1", "1"], "class mapping wrong"
for l in lines:
    vals = [float(v) for v in l.split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in vals), f"un-normalised/out-of-range: {l}"
    xc, yc, w, h = vals
    assert xc - w / 2 >= -1e-6 and xc + w / 2 <= 1 + 1e-6, f"x out of frame: {l}"
    assert yc - h / 2 >= -1e-6 and yc + h / 2 <= 1 + 1e-6, f"y out of frame: {l}"

# clamped box: original 1340,700,100,200 in a 1360x765 image -> w=20px, h=65px
clamped = [float(v) for v in lines[-1].split()[1:]]
assert abs(clamped[2] * 1360 - 20) < 0.1, f"clamp width wrong: {clamped}"
assert abs(clamped[3] * 765 - 65) < 0.1, f"clamp height wrong: {clamped}"

empty = (tmp / "VisDrone2019-DET-train" / "labels" / "0000002_00000_d_0000001.txt")
assert empty.exists() and empty.read_text() == "", "missing-annotation image not handled"

stats = json.loads((tmp / "visdrone_conversion_stats.json").read_text())
assert stats["train"]["boxes_kept"] == 5
assert stats["train"]["per_class"] == {"person": 2, "vehicle": 3}
assert stats["train"]["images"] == 2
assert stats["train"]["images_with_no_objects"] == 1
assert stats["train"]["skipped_ignored_region"] == 2   # cat-0 region + score-0 pedestrian
assert stats["train"]["skipped_other_class"] == 1
assert stats["train"]["skipped_degenerate"] == 1
# 20x40=800 and 18x35=630 -> both area < 32^2, so COCO-small
assert stats["train"]["small_boxes_coco"] == 2, stats["train"]
# 80x60=4800 and the clamped 20x65=1300 -> COCO-medium (32^2 <= area < 96^2)
assert stats["train"]["medium_boxes_coco"] == 2, stats["train"]
# 120x90=10800 -> COCO-large (>= 96^2)
assert stats["train"]["large_boxes_coco"] == 1, stats["train"]

yaml = (tmp / "visdrone.yaml").read_text()
assert "nc: 3" in yaml and "0: person" in yaml and "2: animal" in yaml
print("\nyaml:\n" + yaml)

shutil.rmtree(tmp, ignore_errors=True)
print("ALL CONVERTER TESTS PASSED")
