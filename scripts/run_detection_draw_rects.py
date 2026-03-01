#!/usr/bin/env python3
"""Run YOLO detection on a folder of images and save with rectangles drawn.
No embedding/gallery needed.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Detect and draw rectangles on images")
    p.add_argument("--yolo-model", required=True, help="Path to traffic_sign_yolo.pt")
    p.add_argument("--images-dir", required=True, help="Folder of images (e.g. data/detection_test/stop_sign)")
    p.add_argument("--output-dir", default=None, help="Output folder (default: <images_dir>_detected)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Not a directory: {images_dir}")
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else images_dir.parent / (images_dir.name + "_detected")
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from traffic_sign_recognition.detector import AllObjectDetector, Detection

    detector = AllObjectDetector(
        model_path=args.yolo_model,
        confidence_threshold=args.conf,
        device="cpu",
    )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in exts)
    print(f"Found {len(files)} images in {images_dir}")

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        detections = detector.detect(img)
        draw = ImageDraw.Draw(img)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            label = f"{d.detector_class} {d.confidence:.2f}"
            draw.text((x1, max(0, y1 - 16)), label, fill=(0, 255, 0), font=font)
        out_path = output_dir / f"{f.stem}_detected{f.suffix}"
        img.save(out_path)
        print(f"  [{i+1}/{len(files)}] {f.name} -> {len(detections)} boxes -> {out_path.name}")

    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    main()
