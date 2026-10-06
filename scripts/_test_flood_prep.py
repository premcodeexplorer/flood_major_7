"""Smoke test for prepare_flood_c2a.py against a synthetic Roboflow COCO fixture."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_flood_c2a import iou, source_key  # noqa: E402

assert source_key("flood_image0036_0_png.rf.fe626a.jpg") == "flood_image0036_0_png"
assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9

tmp = Path(tempfile.mkdtemp())
src, out = tmp / "export", tmp / "yolo"
CATS = [{"id": 0, "name": "human", "supercategory": "none"},
        {"id": 1, "name": "0", "supercategory": "human"},
        {"id": 2, "name": "1", "supercategory": "human"}]


def make_split(name, files, boxes):
    d = src / name
    d.mkdir(parents=True)
    images, anns = [], []
    for i, f in enumerate(files):
        Image.new("RGB", (640, 640)).save(d / f)
        images.append({"id": i, "file_name": f, "width": 640, "height": 640})
        for cat, bbox in boxes.get(f, []):
            anns.append({"id": len(anns), "image_id": i, "category_id": cat, "bbox": bbox})
    (d / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": anns, "categories": CATS}))


# train: source A has 4 augmented copies, B has 1 -> keep 2 of A + B
train_files = [f"a_png.rf.{h}.jpg" for h in "1234"] + ["b_png.rf.9.jpg"]
make_split("train", train_files, {
    "a_png.rf.1.jpg": [(1, [10, 10, 200, 200])],
    # b is the only copy of its source, so it is always kept
    "b_png.rf.9.jpg": [(1, [100, 100, 20, 40]),        # person
                       (1, [630, 600, 30, 60]),        # overflows -> clamped to 10 px wide
                       (1, [50, 50, 1, 30])],          # degenerate -> dropped
})
make_split("valid", ["v1_png.rf.1.jpg", "v1_png.rf.2.jpg"], {   # valid: never subsampled
    "v1_png.rf.1.jpg": [(1, [0, 0, 16, 16]), (2, [300, 300, 40, 40])],   # "1" -> person too
})
make_split("test", ["real.rf.1.jpg"], {"real.rf.1.jpg": [(1, [1, 1, 10, 10])]})

r = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_flood_c2a.py"),
                    "--src", str(src), "--out", str(out), "--copies-per-source", "2"],
                   capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0, "converter crashed"

stats = json.loads((out / "flood_c2a_stats.json").read_text())
tr = sorted(p.name for p in (out / "train" / "images").iterdir())
assert len(tr) == 3 and "b_png.rf.9.jpg" in tr, f"train subsampling wrong: {tr}"
assert stats["train"]["sources"] == 2 and stats["train"]["images_in_export"] == 5
assert len(list((out / "val" / "images").iterdir())) == 2, "val must not be subsampled"
assert len(list((out / "test" / "images").iterdir())) == 1

# every image has a label file, and image/label stems match
for split in ("train", "val", "test"):
    imgs = {p.stem for p in (out / split / "images").iterdir()}
    lbls = {p.stem for p in (out / split / "labels").iterdir()}
    assert imgs == lbls, f"{split}: image/label mismatch"

if "a_png.rf.1.jpg" in tr:
    lines = (out / "train" / "labels" / "a_png.rf.1.txt").read_text().split("\n")
    assert len(lines) == 2, f"expected 2 boxes (degenerate dropped), got {lines}"
    assert all(l.split()[0] == "0" for l in lines)
    clamped = [float(v) for v in lines[1].split()[1:]]
    assert abs(clamped[2] * 640 - 10) < 0.1 and abs(clamped[3] * 640 - 40) < 0.1, clamped
    assert stats["train"]["skipped_degenerate"] == 1

v = (out / "val" / "labels" / "v1_png.rf.1.txt").read_text().split("\n")
assert [l.split()[0] for l in v] == ["0", "0"], "category '1' must map to person"
assert stats["val"]["source_categories"] == {"0": 1, "1": 1}
assert (out / "val" / "labels" / "v1_png.rf.2.txt").read_text() == ""
assert stats["val"]["images_with_no_objects"] == 1

for split in ("train", "val", "test"):
    for lbl in (out / split / "labels").iterdir():
        for l in filter(None, lbl.read_text().split("\n")):
            assert all(0.0 <= float(x) <= 1.0 for x in l.split()[1:]), f"out of range: {l}"

y = (out / "flood_c2a.yaml").read_text()
assert "nc: 3" in y and "test: test/images" in y and "2: animal" in y

# re-running must rebuild, not accumulate
r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_flood_c2a.py"),
                     "--src", str(src), "--out", str(out)], capture_output=True, text=True)
assert r2.returncode == 0 and len(list((out / "train" / "images").iterdir())) == 3

print("OK - prepare_flood_c2a.py passed")
