#!/usr/bin/env python3
"""Download the YOLO-v5-format traffic signs dataset from Kaggle.

Downloads the dataset from:
  https://www.kaggle.com/datasets/valentynsichkar/yolo-v5-format-of-the-traffic-signs-dataset

Credentials are read from environment variables KAGGLE_USERNAME and
KAGGLE_KEY, or from a kaggle.json file (~/.kaggle/kaggle.json).

The dataset is extracted into a YOLO-ready directory structure:

    <output_dir>/
        dataset.yaml          # auto-generated if missing
        train/
            images/
            labels/
        val/   (or valid/)
            images/
            labels/

Usage:
    # Set credentials via env vars (e.g. in Railway config)
    export KAGGLE_USERNAME="your_username"
    export KAGGLE_KEY="your_api_key"

    python scripts/download_kaggle_yolo_dataset.py --output-dir /data/yolo_signs
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


KAGGLE_DATASET = "valentynsichkar/yolo-v5-format-of-the-traffic-signs-dataset"


def ensure_kaggle_installed():
    """Ensure the kaggle package is installed."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: F401
    except ImportError:
        print("Installing kaggle package...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
        )


def download_dataset(output_dir: str) -> bool:
    """Download and extract the Kaggle dataset using the Python API.

    Returns True on success.
    """
    ensure_kaggle_installed()
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # If the dataset already has images, skip download
    for subdir in ("train", "val", "valid"):
        images_dir = output_path / subdir / "images"
        if images_dir.is_dir() and any(images_dir.iterdir()):
            print(f"Dataset already present at {output_dir} (found {images_dir}), skipping download.")
            return True

    with tempfile.TemporaryDirectory(prefix="kaggle_yolo_") as tmp:
        print(f"Downloading {KAGGLE_DATASET} ...")
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                KAGGLE_DATASET,
                path=tmp,
                unzip=True,
            )
        except Exception as exc:
            print(f"Kaggle download failed: {exc}")
            print("Make sure KAGGLE_USERNAME and KAGGLE_KEY are set, or")
            print("~/.kaggle/kaggle.json exists with valid credentials.")
            return False

        # The archive may extract into a subdirectory; find the actual root.
        extracted_root = _find_dataset_root(tmp)
        if extracted_root is None:
            print("ERROR: Could not locate train/images inside extracted data.")
            return False

        print(f"Dataset root found at: {extracted_root}")

        # Move contents into the final output directory
        for item in Path(extracted_root).iterdir():
            dest = output_path / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

    _ensure_dataset_yaml(output_path)
    _print_summary(output_path)
    return True


def _find_dataset_root(base: str) -> str | None:
    """Walk *base* looking for a directory that contains train/images/."""
    base_path = Path(base)

    # Direct check
    if (base_path / "train" / "images").is_dir():
        return str(base_path)

    # One level deep
    for child in base_path.iterdir():
        if child.is_dir() and (child / "train" / "images").is_dir():
            return str(child)

    # Two levels deep
    for child in base_path.iterdir():
        if child.is_dir():
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "train" / "images").is_dir():
                    return str(grandchild)

    return None


def _ensure_dataset_yaml(output_path: Path):
    """Create a dataset.yaml if the dataset doesn't already include one."""
    yaml_candidates = list(output_path.glob("*.yaml")) + list(output_path.glob("*.yml"))
    if yaml_candidates:
        print(f"Using existing dataset config: {yaml_candidates[0].name}")
        return

    # Determine val directory name
    val_name = "val"
    if (output_path / "valid").is_dir():
        val_name = "valid"

    # Count classes from a sample label file
    nc, names = _infer_classes(output_path / "train" / "labels")

    yaml_path = output_path / "dataset.yaml"
    yaml_content = (
        f"path: {output_path.resolve()}\n"
        f"train: train/images\n"
        f"val: {val_name}/images\n"
        f"\n"
        f"nc: {nc}\n"
        f"names:\n"
    )
    for i, name in enumerate(names):
        yaml_content += f"  {i}: {name}\n"

    yaml_path.write_text(yaml_content)
    print(f"Generated dataset.yaml with {nc} classes at {yaml_path}")


def _infer_classes(labels_dir: Path) -> tuple[int, list[str]]:
    """Read label files to discover the set of class IDs."""
    class_ids: set[int] = set()
    if labels_dir.is_dir():
        for lbl in labels_dir.iterdir():
            if lbl.suffix == ".txt":
                for line in lbl.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_ids.add(int(parts[0]))
                        except ValueError:
                            pass
            if len(class_ids) > 100:
                break  # enough to know the range

    if not class_ids:
        return 1, ["traffic_sign"]

    nc = max(class_ids) + 1
    names = [f"class_{i}" for i in range(nc)]
    return nc, names


def _print_summary(output_path: Path):
    """Print a summary of the downloaded dataset."""
    print("\n=== YOLO Dataset Summary ===")
    for split in ("train", "val", "valid", "test"):
        img_dir = output_path / split / "images"
        lbl_dir = output_path / split / "labels"
        if img_dir.is_dir():
            n_imgs = len(list(img_dir.iterdir()))
            n_lbls = len(list(lbl_dir.iterdir())) if lbl_dir.is_dir() else 0
            print(f"  {split}: {n_imgs} images, {n_lbls} labels")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO traffic signs dataset from Kaggle"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/yolo_signs",
        help="Directory to save the YOLO-format dataset (default: data/yolo_signs)",
    )
    args = parser.parse_args()

    ok = download_dataset(args.output_dir)
    if not ok:
        sys.exit(1)
    print("Done!")


if __name__ == "__main__":
    main()
