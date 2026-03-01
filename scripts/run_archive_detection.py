#!/usr/bin/env python3
"""Run detect_and_recognize on every test_archive_* image in data/archive_test."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_TEST_DIR = ROOT / "data" / "archive_test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    p = argparse.ArgumentParser(description="Process all test_archive_* images in data/archive_test")
    p.add_argument("--embedding-model", default=None, help="Path to best_model.pth")
    p.add_argument("--gallery", default=None, help="Base path for gallery (gallery.json + .npz)")
    p.add_argument("--detection-conf", type=float, default=0.25)
    args = p.parse_args()

    embedding_model = args.embedding_model
    gallery = args.gallery
    if not embedding_model or not gallery:
        # Default to model folder next to task-uzcosmos (e.g. C:\\Users\\user\\Desktop\\model)
        default_model = ROOT.parent / "best_model.pth"
        default_gallery = ROOT.parent / "gallery"
        if default_model.exists() and Path(f"{default_gallery}.json").exists():
            embedding_model = str(default_model)
            gallery = str(default_gallery)
    if not embedding_model or not gallery:
        print("Provide --embedding-model and --gallery, or place best_model.pth and gallery.json in parent of task-uzcosmos.")
        sys.exit(1)

    if not ARCHIVE_TEST_DIR.is_dir():
        print(f"Directory not found: {ARCHIVE_TEST_DIR}")
        sys.exit(1)

    images = sorted(
        f for f in ARCHIVE_TEST_DIR.iterdir()
        if f.is_file() and f.name.startswith("test_archive_") and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No test_archive_* images in {ARCHIVE_TEST_DIR}")
        sys.exit(1)

    script = ROOT / "scripts" / "detect_and_recognize.py"
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}")
        cmd = [
            sys.executable,
            str(script),
            "--image", str(path),
            "--embedding-model", embedding_model,
            "--gallery", gallery,
            "--detection-conf", str(args.detection_conf),
        ]
        r = subprocess.run(cmd, cwd=str(ROOT), timeout=90)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode})")

    print(f"Done. Processed {len(images)} images. Annotated: *_detected.* in {ARCHIVE_TEST_DIR}")


if __name__ == "__main__":
    main()
