"""
Builds a SIMULATED drone pass over one still aerial photo, for testing the
tracker, camera-motion compensation and memory when no real flood video is
at hand.

A crop window drifts across the photo with slow zoom and slight rotation,
the way a hovering drone wanders. What it can test: IDs stay stable while
the camera moves, still people are recognised as still (time stranded
grows), the flood map stays aligned. What it CANNOT show: people moving or
water rising - nothing in a still photo changes. Label any result from it
as "simulated camera motion over a still image".

Usage:
    python scripts/make_demo_video.py --image some_flood_photo.jpg --out demo.mp4 --seconds 30 --fps 10
"""

import argparse
import math

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], help="output W H")
    ap.add_argument("--view", type=float, default=0.75, help="crop width as a share of the photo")
    ap.add_argument("--rotate", type=float, default=3.0, help="max rotation, degrees")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    H, W = img.shape[:2]
    ow, oh = args.size
    n = int(args.seconds * args.fps)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ow, oh))
    for i in range(n):
        u = i / max(1, n - 1)
        zoom = 1.0 + 0.15 * math.sin(math.pi * u)                    # descend a little, then back
        cw = args.view * W / zoom
        ch = cw * oh / ow
        if ch > H * 0.95:
            ch = H * 0.95
            cw = ch * ow / oh
        # centre wanders on a slow ellipse inside the free margin
        mx, my = (W - cw) / 2, (H - ch) / 2
        cx = W / 2 + 0.8 * mx * math.sin(2 * math.pi * u)
        cy = H / 2 + 0.8 * my * math.sin(math.pi * u)
        ang = args.rotate * math.sin(2 * math.pi * u * 1.5)
        s = ow / cw
        M = cv2.getRotationMatrix2D((cx, cy), ang, s)
        M[:, 2] += np.array([ow / 2 - cx, oh / 2 - cy])
        vw.write(cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    vw.release()
    print(f"[demo] {n} frames @ {args.fps} fps -> {args.out}  (simulated camera motion over a still photo)")


if __name__ == "__main__":
    main()
