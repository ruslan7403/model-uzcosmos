#!/usr/bin/env python3
"""Prepare test images from archive class dirs (full images, no negative crops).

Samples images from data/mapillary (archive_02, archive_03, ...) and copies
them to data/archive_test for running detection.
"""

import random
import shutil
import sys
from pathlib import Path

# Where archive class dirs live and where to write test images
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "data" / "mapillary"
OUTPUT_DIR = ROOT / "data" / "archive_test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_IMAGES = 10


def collect_archive_images(max_per_class: int = 100) -> list[Path]:
    """Collect image paths from archive class subdirs."""
    if not ARCHIVE_DIR.is_dir():
        print(f"Archive dir not found: {ARCHIVE_DIR}")
        return []
    paths = []
    for class_dir in sorted(ARCHIVE_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            continue
        sample = random.sample(files, min(max_per_class, len(files)))
        paths.extend(sample)
    return paths


def main():
    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = collect_archive_images()
    if len(candidates) < NUM_IMAGES:
        print(f"Not enough archive images (have {len(candidates)}, want {NUM_IMAGES})")
        sys.exit(1)

    chosen = random.sample(candidates, NUM_IMAGES)
    for i, path in enumerate(chosen):
        out_path = OUTPUT_DIR / f"test_archive_{i + 1:02d}{path.suffix.lower()}"
        shutil.copy2(path, out_path)
        print(f"  {out_path.name} (from {path.parent.name}/{path.name})")

    print(f"\nSaved {len(chosen)} test images from archives to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
