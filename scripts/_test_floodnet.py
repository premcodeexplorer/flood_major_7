"""Smoke test for the Model-2 pipeline: prep -> 1 training step per run -> compare.

Needs torch, segmentation-models-pytorch, albumentations, opencv (CPU is fine).
Uses a synthetic FloodNet tree with the official folder names, random weights
(no ImageNet download), and tiny frames, so it checks wiring, not accuracy.
    python scripts/_test_floodnet.py            # all 4 runs
    python scripts/_test_floodnet.py unet_r50   # just one
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "scripts"
sys.path.insert(0, str(S))
from floodnet_train import RUNS  # noqa: E402

tmp = Path(tempfile.mkdtemp())
dl = tmp / "download"          # layout of the official Dropbox zip
raw, data, out = dl / "FloodNet-Supervised_v1.0", tmp / "cache", tmp / "drive"
(dl / "ColorMasks-FloodNetv1.0" / "ColorMasks-ValSet").mkdir(parents=True)
cv2.imwrite(str(dl / "ColorMasks-FloodNetv1.0" / "ColorMasks-ValSet" / "val0_lab.png"),
            np.zeros((60, 80, 3), np.uint8))    # RGB preview: must be ignored
rng = np.random.default_rng(0)
for split, n in [("train", 4), ("val", 2), ("test", 2)]:
    (raw / split / f"{split}-org-img").mkdir(parents=True)
    (raw / split / f"{split}-label-img").mkdir(parents=True)
    for i in range(n):
        h, w = (800, 600) if (split, i) == ("train", 0) else (600, 800)   # one portrait image
        cv2.imwrite(str(raw / split / f"{split}-org-img" / f"{split}{i}.jpg"),
                    rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        m = np.zeros((h, w), np.uint8)
        for c in range(10):        # every class present, in bands
            m[:, c * w // 10:(c + 1) * w // 10] = c
        cv2.imwrite(str(raw / split / f"{split}-label-img" / f"{split}{i}_lab.png"), m)


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-2500:])
    assert r.returncode == 0, f"failed: {cmd}"
    return r


run([str(S / "floodnet_prep.py"), "--src", str(dl), "--out", str(data),
     "--width", "256", "--height", "192", "--workers", "2"])
st = json.loads((data / "floodnet_stats.json").read_text())
assert [st["splits"][s]["images"] for s in ("train", "val", "test")] == [4, 2, 2]
assert st["portrait_images_rotated"] == 1
assert st["splits"]["val"]["images_containing"]["pool"] == 2
img = cv2.imread(str(data / "train" / "images" / "train0.jpg"))
mask = cv2.imread(str(data / "train" / "masks" / "train0.png"), cv2.IMREAD_UNCHANGED)
assert img.shape == (192, 256, 3) and mask.shape == (192, 256), (img.shape, mask.shape)
assert set(np.unique(mask)) <= set(range(10)), "nearest resize must keep class ids"

# a broken download must be refused: delete one mask
bad = tmp / "bad"
import shutil  # noqa: E402
shutil.copytree(raw, bad)
(bad / "val" / "val-label-img" / "val0_lab.png").unlink()
r = subprocess.run([sys.executable, str(S / "floodnet_prep.py"), "--src", str(bad),
                    "--out", str(tmp / "bad_out")], capture_output=True, text=True)
assert r.returncode != 0 and "missing mask" in r.stdout, "unpaired image must fail prep"

runs = sys.argv[1:] or sorted(RUNS)
for name in runs:
    run([str(S / "floodnet_train.py"), "--run", name, "--data", str(data), "--out", str(out),
         "--epochs", "2", "--val-every", "1", "--batch", "2", "--crop", "128", "--workers", "0",
         "--max-steps", "1", "--no-pretrained"])
    res = json.loads((out / "results" / f"{name}.json").read_text())
    assert res["best_epoch"] in (1, 2) and len(res["test"]["per_class_IoU"]) == 10
    assert (out / "runs" / name / "test_predictions.jpg").exists()
    # finished runs refuse to retrain; --eval-only re-evaluates
    r = subprocess.run([sys.executable, str(S / "floodnet_train.py"), "--run", name, "--data", str(data),
                        "--out", str(out), "--no-pretrained"], capture_output=True, text=True)
    assert r.returncode != 0 and "finished" in (r.stdout + r.stderr)

# resume: kill-equivalent = stop after epoch 1 of 2, then rerun with 2 epochs
name = runs[0]
(out / "results" / f"{name}.json").unlink()
run([str(S / "floodnet_train.py"), "--run", name, "--data", str(data), "--out", str(out), "--seed", "1",
     "--epochs", "1", "--val-every", "1", "--batch", "2", "--crop", "128", "--workers", "0",
     "--max-steps", "1", "--no-pretrained"])
(out / "results" / f"{name}_seed1.json").unlink()
r = run([str(S / "floodnet_train.py"), "--run", name, "--data", str(data), "--out", str(out), "--seed", "1",
         "--epochs", "2", "--val-every", "1", "--batch", "2", "--crop", "128", "--workers", "0",
         "--max-steps", "1", "--no-pretrained"])
assert "[resume] from epoch 1" in r.stdout, "must resume from last.pt"
hist = json.loads((out / "runs" / f"{name}_seed1" / "history.json").read_text())
assert [h["epoch"] for h in hist] == [0, 1], hist

run([str(S / "floodnet_compare.py"), "--results", str(out / "results")])
md = (out / "comparison.md").read_text(encoding="utf-8")
assert "PSPNet-R101 [FloodNet] *" in md and "79.69" in md
print(f"OK - FloodNet pipeline passed for {len(runs)} run(s)")
