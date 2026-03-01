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
import re
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

        # Always print what Kaggle actually extracted (so we can adapt the code to the real structure).
        print("=== Kaggle extracted structure ===")
        _print_extracted_structure(tmp)
        print("=== End structure ===")

        # The archive may extract into a subdirectory; find the actual root.
        extracted_root, layout = _find_dataset_root(tmp)
        if extracted_root is None:
            print("ERROR: Could not locate train/images or images/train inside extracted data.")
            return False

        print(f"Dataset root found at: {extracted_root} (layout: {layout})")

        root_path = Path(extracted_root)
        nc_names_from_source: tuple[int, list[str]] | None = None
        if layout == "images_train":
            # Dataset uses images/train, images/validation, images/test -> normalize to train/images, val/images, test/images
            _normalize_images_train_layout(root_path, output_path)
            # Preserve class names from source dataset.yaml (ts43classes: '0'..'42', ts4classes: prohibitory/danger/mandatory/other)
            nc_names_from_source = _parse_source_dataset_yaml(root_path)
        else:
            # Standard layout: train/images, valid/images -> move as-is
            for item in root_path.iterdir():
                dest = output_path / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

    _ensure_dataset_yaml(output_path, nc_names_override=nc_names_from_source)
    _print_summary(output_path)
    return True


def _find_dataset_root(base: str) -> tuple[str | None, str]:
    """Walk *base* for a dir with train/images/ (standard) or images/train/ (alt).
    For this Kaggle dataset, prefer ts43classes over ts4classes when both exist.
    Returns (root_path, "standard"|"images_train") or (None, "").
    """
    base_path = Path(base)
    max_depth = 5
    candidates: list[tuple[str, str]] = []  # (path, layout)

    def search(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        if (path / "train" / "images").is_dir():
            candidates.append((str(path), "standard"))
            return
        if (path / "images" / "train").is_dir():
            candidates.append((str(path), "images_train"))
            return
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir():
                    search(child, depth + 1)

    search(base_path, 0)
    if not candidates:
        return None, ""
    # Prefer ts43classes (43 classes) over ts4classes when both exist
    for path, layout in candidates:
        if "ts43classes" in path:
            return path, layout
    return candidates[0][0], candidates[0][1]


def _normalize_images_train_layout(src_root: Path, dest_root: Path) -> None:
    """Copy images/train -> train/images, images/validation -> val/images, images/test -> test/images (and labels)."""
    # Matches dataset.yaml: train, val=validation, test=test
    splits = [
        ("train", "train"),
        ("validation", "val"),
        ("valid", "val"),
        ("test", "test"),
    ]
    for src_split, dest_split in splits:
        img_src = src_root / "images" / src_split
        lbl_src = src_root / "labels" / src_split
        if not img_src.is_dir():
            continue
        img_dest = dest_root / dest_split / "images"
        lbl_dest = dest_root / dest_split / "labels"
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        for f in img_src.iterdir():
            if f.is_file():
                shutil.copy2(f, img_dest / f.name)
        if lbl_src.is_dir():
            for f in lbl_src.iterdir():
                if f.is_file():
                    shutil.copy2(f, lbl_dest / f.name)
    # Don't copy source dataset.yaml (paths would be wrong); _ensure_dataset_yaml will generate one.


def _print_extracted_structure(base: str, max_lines: int = 120, max_depth: int = 8, sample_files: int = 5) -> None:
    """Print the full extracted directory structure so we can adapt code to the real layout."""
    base_path = Path(base)
    lines: list[str] = []

    def walk(p: Path, prefix: str, depth: int) -> None:
        if depth > max_depth or len(lines) >= max_lines:
            return
        try:
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        for entry in dirs:
            if len(lines) >= max_lines:
                return
            lines.append(f"{prefix}{entry.name}/")
            walk(entry, prefix + "  ", depth + 1)
        # Show files: first sample_files, then count if more
        if files:
            for f in files[:sample_files]:
                lines.append(f"{prefix}{f.name}")
            if len(files) > sample_files:
                lines.append(f"{prefix}... ({len(files)} files total)")
        if len(lines) >= max_lines:
            lines.append(prefix + "... (truncated)")

    walk(base_path, "", 0)
    print("\n".join(lines))


def _parse_source_dataset_yaml(src_root: Path) -> tuple[int, list[str]] | None:
    """Read dataset.yaml in src_root; return (nc, names) or None if missing/unparseable."""
    for name in ("dataset.yaml", "data.yaml", "dataset.yml"):
        p = src_root / name
        if not p.is_file():
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        nc = None
        names: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("nc:") and nc is None:
                try:
                    nc = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            if "names:" in line and "[" in line:
                # names: [ '0', '1', ... ] or names: [ 'prohibitory', 'danger', ... ]
                match = re.search(r"\[(.*)\]", line)
                if match:
                    inner = match.group(1)
                    names = [s.strip().strip("'\"").strip() for s in inner.split(",")]
                break
        if nc is not None and len(names) >= nc:
            return nc, names[:nc]
        if names:
            return len(names), names
    return None


def _ensure_dataset_yaml(output_path: Path, nc_names_override: tuple[int, list[str]] | None = None):
    """Create dataset.yaml. Use nc_names_override (from source dataset.yaml) when provided to preserve class names."""
    yaml_candidates = list(output_path.glob("*.yaml")) + list(output_path.glob("*.yml"))
    if yaml_candidates:
        print(f"Using existing dataset config: {yaml_candidates[0].name}")
        return

    # Determine val directory name
    val_name = "val"
    if (output_path / "valid").is_dir():
        val_name = "valid"

    if nc_names_override is not None:
        nc, names = nc_names_override
        print(f"Using class names from source dataset.yaml (nc={nc})")
    else:
        nc, names = _infer_classes(output_path / "train" / "labels")

    yaml_path = output_path / "dataset.yaml"
    yaml_content = (
        f"path: {output_path.resolve()}\n"
        f"train: train/images\n"
        f"val: {val_name}/images\n"
    )
    if (output_path / "test" / "images").is_dir():
        yaml_content += f"test: test/images\n"
    yaml_content += (
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
