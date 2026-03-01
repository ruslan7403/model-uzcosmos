#!/usr/bin/env python3
"""Run detection on archive test images and assert 0 objects captured (traffic_sign_yolo)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Run detection on archive test set; expect 0 detections.")
    p.add_argument("--test-dir", default=str(ROOT / "data" / "archive_test"), help="Directory with negative test images")
    p.add_argument("--embedding-model", required=True, help="Path to best_model.pth")
    p.add_argument("--gallery", required=True, help="Base path for gallery (gallery.json + .npz)")
    p.add_argument("--detection-conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = p.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        print(f"Test dir not found: {test_dir}")
        return 1

    images = sorted(test_dir.glob("negative_*.jpg"))
    if not images:
        print(f"No negative_*.jpg images in {test_dir}")
        return 1

    cmd_base = [
        sys.executable,
        str(ROOT / "scripts" / "detect_and_recognize.py"),
        "--embedding-model", args.embedding_model,
        "--gallery", args.gallery,
        "--detection-conf", str(args.detection_conf),
    ]
    failed = []
    for img in images:
        cmd = cmd_base + ["--image", str(img)]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        # Count "Detected N traffic sign(s)" and require N == 0
        out = r.stdout + r.stderr
        if "Detected 0 traffic sign(s)" not in out:
            failed.append(img.name)
        if r.returncode != 0:
            print(out)
    if failed:
        print(f"FAIL: expected 0 detections on every image, but got detections on: {failed}")
        return 1
    print(f"OK: {len(images)} images, 0 objects captured on each (traffic_sign_yolo).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
