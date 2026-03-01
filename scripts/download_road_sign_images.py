#!/usr/bin/env python3
"""Download N images of road/traffic signs from the internet via Bing image search."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "road_sign_internet"
NUM_IMAGES = 20


def main():
    p = argparse.ArgumentParser(description="Download road sign images from the internet")
    p.add_argument("--num", type=int, default=NUM_IMAGES, help="Number of images to download")
    p.add_argument("--output-dir", default=None, help=f"Output directory (default: {OUTPUT_DIR})")
    args = p.parse_args()

    try:
        from bing_image_downloader import downloader
    except ImportError:
        print("Install bing-image-downloader: pip install bing-image-downloader")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    # Bing downloader creates a subdir named by search term; we'll use a temp name then move
    search_term = "road traffic sign"
    raw_dir = out_dir.parent / "road_sign_internet_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloader.download(
            search_term,
            limit=args.num,
            output_dir=str(raw_dir),
            adult_filter_off=False,
            force_replace=False,
            timeout=15,
            verbose=False,
        )
    except Exception as e:
        print(f"Download error: {e}")
        sys.exit(1)

    # Move images into flat output dir (downloader uses raw_dir/search_term/)
    sub = raw_dir / search_term
    if not sub.is_dir():
        print("No images downloaded.")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in exts)
    for i, f in enumerate(files[: args.num]):
        dest = out_dir / f"road_sign_{i + 1:02d}{f.suffix.lower()}"
        shutil.copy2(f, dest)
        print(f"  [{i + 1}/{len(files[: args.num])}] {dest.name}")
    try:
        shutil.rmtree(raw_dir)
    except Exception:
        pass
    print(f"\nSaved {len(files[: args.num])} images to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
