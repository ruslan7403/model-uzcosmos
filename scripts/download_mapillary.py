#!/usr/bin/env python3
"""Download and organize the Mapillary Traffic Sign Dataset.

Downloads images from the Mapillary Traffic Sign Dataset and organizes
them into a class-per-directory structure suitable for training.

The dataset requires an access token from https://www.mapillary.com/
Set the MAPILLARY_ACCESS_TOKEN environment variable before running.

Usage:
    export MAPILLARY_ACCESS_TOKEN="your_token_here"
    python scripts/download_mapillary.py --output-dir data/mapillary

If the dataset is already partially downloaded, re-running will skip
existing images (resume-friendly).
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def download_file(url: str, dest: str, retries: int = 3) -> bool:
    """Download a file with retries."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except (urllib.error.URLError, OSError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {retries} attempts: {e}")
                return False


def organize_by_class(annotations_file: str, images_dir: str, output_dir: str):
    """Organize downloaded images into class subdirectories.

    Args:
        annotations_file: Path to annotations JSON file.
        images_dir: Directory containing downloaded images.
        output_dir: Output directory with class subdirectories.
    """
    with open(annotations_file) as f:
        annotations = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    organized = 0
    for entry in annotations.get("images", annotations.get("annotations", [])):
        # Support different annotation formats
        if "label" in entry:
            class_name = entry["label"]
        elif "category_id" in entry:
            class_name = str(entry["category_id"])
        elif "class" in entry:
            class_name = entry["class"]
        else:
            continue

        if "file_name" in entry:
            src = os.path.join(images_dir, entry["file_name"])
        elif "image_path" in entry:
            src = os.path.join(images_dir, entry["image_path"])
        else:
            continue

        if not os.path.exists(src):
            continue

        # Sanitize class name for filesystem
        safe_class = class_name.replace("/", "_").replace(" ", "_").strip(".")
        class_dir = os.path.join(output_dir, safe_class)
        os.makedirs(class_dir, exist_ok=True)

        dest = os.path.join(class_dir, os.path.basename(src))
        if not os.path.exists(dest):
            os.link(src, dest) if os.path.samefile(
                os.path.dirname(src), os.path.dirname(dest)
            ) else __import__("shutil").copy2(src, dest)
            organized += 1

    print(f"Organized {organized} images into {output_dir}")


def download_via_vistas_api(access_token: str, output_dir: str,
                            max_images: int = None):
    """Download traffic sign images via Mapillary Vistas API.

    Uses the Mapillary API to fetch traffic sign detections and
    download the corresponding images.

    Args:
        access_token: Mapillary API access token.
        output_dir: Directory to save images organized by class.
        max_images: Maximum number of images to download per class.
    """
    import urllib.parse

    base_url = "https://graph.mapillary.com"
    images_dir = os.path.join(output_dir, "raw_images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Traffic sign object values from Mapillary
    # These are the most common traffic sign types
    sign_types = {
        "regulatory--stop--g1": "stop",
        "regulatory--yield--g1": "yield",
        "regulatory--no-entry--g1": "no_entry",
        "regulatory--speed-limit-30--g1": "speed_limit_30",
        "regulatory--speed-limit-50--g1": "speed_limit_50",
        "regulatory--speed-limit-60--g1": "speed_limit_60",
        "regulatory--speed-limit-80--g1": "speed_limit_80",
        "regulatory--speed-limit-100--g1": "speed_limit_100",
        "regulatory--speed-limit-120--g1": "speed_limit_120",
        "regulatory--no-overtaking--g1": "no_overtaking",
        "regulatory--no-parking--g1": "no_parking",
        "regulatory--one-way-right--g1": "one_way_right",
        "regulatory--one-way-left--g1": "one_way_left",
        "regulatory--turn-right--g1": "turn_right",
        "regulatory--turn-left--g1": "turn_left",
        "regulatory--go-straight--g1": "go_straight",
        "regulatory--pedestrians-only--g1": "pedestrians_only",
        "regulatory--bicycles-only--g1": "bicycles_only",
        "warning--pedestrians-crossing--g1": "pedestrians_crossing",
        "warning--curve-right--g1": "curve_right",
        "warning--curve-left--g1": "curve_left",
        "warning--road-bump--g1": "road_bump",
        "warning--slippery-road-surface--g1": "slippery_road",
        "warning--construction--g1": "construction",
        "warning--traffic-signals--g1": "traffic_signals",
        "information--parking--g1": "parking",
        "information--hospital--g1": "hospital",
        "information--gas-station--g1": "gas_station",
        "complementary--chevron-right--g1": "chevron_right",
        "complementary--chevron-left--g1": "chevron_left",
    }

    print(f"Downloading {len(sign_types)} traffic sign classes...")
    total_downloaded = 0

    for sign_value, class_name in sign_types.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        existing = len([f for f in os.listdir(class_dir)
                        if f.endswith((".jpg", ".png"))])
        if max_images and existing >= max_images:
            print(f"  [{class_name}] Already has {existing} images, skipping")
            continue

        print(f"  [{class_name}] Fetching detections for: {sign_value}")

        # Query map features (traffic sign detections)
        params = urllib.parse.urlencode({
            "access_token": access_token,
            "fields": "id,object_value,geometry",
            "object_values": sign_value,
            "limit": max_images or 500,
        })
        url = f"{base_url}/map_features?{params}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError) as e:
            print(f"    Failed to fetch detections: {e}")
            continue

        features = data.get("data", [])
        print(f"    Found {len(features)} detections")

        class_count = existing
        for feat in features:
            if max_images and class_count >= max_images:
                break

            feat_id = feat.get("id")
            if not feat_id:
                continue

            # Get the image thumbnail for this detection
            thumb_params = urllib.parse.urlencode({
                "access_token": access_token,
                "fields": "id,thumb_2048_url",
            })
            thumb_url = f"{base_url}/{feat_id}/detections?{thumb_params}"

            try:
                req = urllib.request.Request(thumb_url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    det_data = json.loads(resp.read().decode())
            except (urllib.error.URLError, OSError):
                continue

            for detection in det_data.get("data", []):
                image_url = detection.get("thumb_2048_url")
                if not image_url:
                    continue

                img_filename = f"{class_name}_{feat_id}_{detection.get('id', class_count)}.jpg"
                img_path = os.path.join(class_dir, img_filename)

                if os.path.exists(img_path):
                    class_count += 1
                    continue

                if download_file(image_url, img_path):
                    class_count += 1
                    total_downloaded += 1

                    if total_downloaded % 50 == 0:
                        print(f"    Downloaded {total_downloaded} images total...")

                # Be polite to the API
                time.sleep(0.1)

        final_count = len([f for f in os.listdir(class_dir)
                           if f.endswith((".jpg", ".png"))])
        print(f"    [{class_name}] {final_count} total images")

    print(f"\nDownload complete: {total_downloaded} new images")
    print(f"Dataset directory: {output_dir}")

    # Print summary
    total = 0
    for d in sorted(os.listdir(output_dir)):
        dp = os.path.join(output_dir, d)
        if os.path.isdir(dp) and d != "raw_images":
            n = len([f for f in os.listdir(dp) if f.endswith((".jpg", ".png"))])
            if n > 0:
                print(f"  {d}: {n} images")
                total += n
    print(f"  Total: {total} images")


def main():
    parser = argparse.ArgumentParser(
        description="Download the Mapillary Traffic Sign Dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/mapillary",
        help="Directory to save the dataset (default: data/mapillary)",
    )
    parser.add_argument(
        "--max-images-per-class", type=int, default=None,
        help="Maximum images per class (default: all available)",
    )
    parser.add_argument(
        "--annotations", type=str, default=None,
        help="Path to existing annotations JSON (for local dataset)",
    )
    parser.add_argument(
        "--images-dir", type=str, default=None,
        help="Path to existing raw images dir (for local dataset)",
    )

    args = parser.parse_args()

    if args.annotations and args.images_dir:
        # Organize an existing local dataset
        print("Organizing local dataset...")
        organize_by_class(args.annotations, args.images_dir, args.output_dir)
    else:
        # Download via Mapillary API
        access_token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
        if not access_token:
            print("ERROR: Set the MAPILLARY_ACCESS_TOKEN environment variable.")
            print("Get a token at: https://www.mapillary.com/dashboard/developers")
            sys.exit(1)

        download_via_vistas_api(
            access_token=access_token,
            output_dir=args.output_dir,
            max_images=args.max_images_per_class,
        )


if __name__ == "__main__":
    main()
