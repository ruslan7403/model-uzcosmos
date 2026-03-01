#!/usr/bin/env python3
"""Take 20 images from archive folder, run YOLO detection, save with rectangles.

Collects images from data/archive_test (test_archive_*.jpg, then other images)
up to 20, then runs detection and writes to data/archive_20_detected.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_TEST_DIR = ROOT / "data" / "archive_test"
OUTPUT_DIR = ROOT / "data" / "archive_20_detected"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_IMAGES = 20


def collect_archive_test_images() -> list[Path]:
    """Images from data/archive_test (test_archive_*.jpg preferred). Excludes *_detected*."""
    if not ARCHIVE_TEST_DIR.is_dir():
        return []
    all_jpgs = [
        f for f in ARCHIVE_TEST_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and "_detected" not in f.stem
    ]
    test_archive = sorted(f for f in all_jpgs if f.name.startswith("test_archive_"))
    rest = sorted(f for f in all_jpgs if f not in test_archive)
    return (test_archive + rest)[:NUM_IMAGES]


def main():
    p = argparse.ArgumentParser(description="Detect and draw rectangles on 20 archive images")
    p.add_argument("--yolo-model", required=True, help="Path to traffic_sign_yolo.pt")
    p.add_argument("--output-dir", default=None, help=f"Output folder (default: {OUTPUT_DIR})")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = p.parse_args()

    candidates = collect_archive_test_images()
    if len(candidates) < 1:
        print("No archive images found in data/archive_test.")
        sys.exit(1)
    if len(candidates) < NUM_IMAGES:
        print(f"Only {len(candidates)} archive images found; using all of them (requested {NUM_IMAGES}).")
    else:
        candidates = candidates[:NUM_IMAGES]

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from traffic_sign_recognition.detector import AllObjectDetector

    detector = AllObjectDetector(
        model_path=args.yolo_model,
        confidence_threshold=args.conf,
        device="cpu",
    )

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, f in enumerate(candidates):
        img = Image.open(f).convert("RGB")
        detections = detector.detect(img)
        draw = ImageDraw.Draw(img)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            label = f"{d.detector_class} {d.confidence:.2f}"
            draw.text((x1, max(0, y1 - 16)), label, fill=(0, 255, 0), font=font)
        out_name = f"archive_{i + 1:02d}_detected{f.suffix}"
        out_path = output_dir / out_name
        img.save(out_path)
        print(f"  [{i+1}/{len(candidates)}] {f.name} -> {len(detections)} boxes -> {out_path.name}")

    print(f"Done. Saved {len(candidates)} images to {output_dir}")


if __name__ == "__main__":
    main()
