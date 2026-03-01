#!/usr/bin/env python3
"""Run detect_and_recognize on every image in data/detection_test (stop_sign, traffic_light, parking_meter).
Saves annotated images to data/detection_test_results preserving folder structure.
Uses traffic_sign_yolo when found next to the embedding model.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBDIRS = ["stop_sign", "traffic_light", "parking_meter"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    p = argparse.ArgumentParser(description="Run detection on data/detection_test (99 images: 33 each)")
    p.add_argument("--embedding-model", default=None, help="Path to best_model.pth")
    p.add_argument("--gallery", default=None, help="Base path for gallery")
    p.add_argument("--detection-conf", type=float, default=0.25)
    args = p.parse_args()

    root = ROOT / "data" / "detection_test"
    out_root = ROOT / "data" / "detection_test_results"
    embedding_model = args.embedding_model
    gallery = args.gallery
    if not embedding_model or not gallery:
        default_model = ROOT.parent / "best_model.pth"
        default_gallery = ROOT.parent / "gallery"
        if default_model.exists() and (default_gallery.with_suffix(".json")).exists():
            embedding_model = str(default_model)
            gallery = str(default_gallery)
    if not embedding_model or not gallery:
        embedding_model = str(ROOT / "output" / "best_model.pth")
        gallery = str(ROOT / "output" / "gallery")
    if not os.path.isfile(embedding_model):
        print(f"Embedding model not found: {embedding_model}")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for subdir in SUBDIRS:
        src_dir = root / subdir
        dst_dir = out_root / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in EXTENSIONS:
                continue
            out_path = dst_dir / f"{f.stem}_detected{f.suffix}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "detect_and_recognize.py"),
                "--image", str(f),
                "--output", str(out_path),
                "--embedding-model", embedding_model,
                "--gallery", gallery,
                "--detection-conf", str(args.detection_conf),
            ]
            total += 1
            print(f"[{total}] {subdir}/{f.name}")
            r = subprocess.run(cmd, cwd=str(ROOT), timeout=90)
            if r.returncode != 0:
                print("  FAILED")
    print(f"Done. Processed {total} images. Results in data/detection_test_results/")


if __name__ == "__main__":
    main()
