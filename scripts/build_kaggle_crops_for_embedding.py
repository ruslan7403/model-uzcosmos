#!/usr/bin/env python3
"""Build class-per-folder crop dataset from Kaggle YOLO dataset for embedding training.

Reads train/images + train/labels (and optionally val, test), crops each bbox,
saves to output_dir/class_name/ so train.py (triplet/embedding) can use it.
Single source of truth is Kaggle YOLO dataset.

Usage:
    python scripts/build_kaggle_crops_for_embedding.py \
        --yolo-dir /data/yolo_signs \
        --output-dir /data/kaggle_crops
"""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_dataset_yaml(yolo_dir: Path) -> tuple[int, list[str]]:
    """Read dataset.yaml for nc and names. Fallback to class_0, class_1, ..."""
    for name in ("dataset.yaml", "data.yaml"):
        p = yolo_dir / name
        if not p.is_file():
            continue
        text = p.read_text()
        nc, names = None, []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("nc:"):
                try:
                    nc = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            if "names:" in line and "[" in line:
                m = re.search(r"\[(.*)\]", line)
                if m:
                    names = [s.strip().strip("'\"").strip() for s in m.group(1).split(",")]
        if nc is not None and len(names) >= nc:
            return nc, names[:nc]
        if names:
            return len(names), names
    return 0, []


def yolo_line_to_bbox(line: str, img_w: int, img_h: int, padding: float = 0.1):
    """Convert YOLO line (class_id x_center y_center w h normalized) to pixel crop box with padding."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    try:
        cid = int(parts[0])
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h
    except (ValueError, IndexError):
        return None, None
    # box in center format
    x1 = max(0, xc - w / 2 - padding * w)
    y1 = max(0, yc - h / 2 - padding * h)
    x2 = min(img_w, xc + w / 2 + padding * w)
    y2 = min(img_h, yc + h / 2 + padding * h)
    return cid, (int(x1), int(y1), int(x2), int(y2))


def build_crops(yolo_dir: Path, output_dir: Path, splits: tuple[str, ...] = ("train", "val"), min_side: int = 32):
    """Extract crops from YOLO images/labels into output_dir/class_name/."""
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nc, names = parse_dataset_yaml(yolo_dir)
    if not names:
        # Infer from labels
        class_ids = set()
        for split in splits:
            lbl_dir = yolo_dir / split / "labels"
            if not lbl_dir.is_dir():
                continue
            for f in lbl_dir.glob("*.txt"):
                for line in f.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_ids.add(int(parts[0]))
                        except ValueError:
                            pass
        nc = max(class_ids) + 1 if class_ids else 1
        names = [str(i) for i in range(nc)]
    print(f"Classes: nc={nc}, names={names[:5]}{'...' if len(names) > 5 else ''}")

    # sanitize dir names (no slashes)
    class_dirs = {}
    for i, n in enumerate(names):
        safe = re.sub(r"[^\w\-]", "_", str(n)) or f"class_{i}"
        class_dirs[i] = output_dir / safe
        class_dirs[i].mkdir(parents=True, exist_ok=True)

    count_per_class = {i: 0 for i in range(len(names))}
    for split in splits:
        img_dir = yolo_dir / split / "images"
        lbl_dir = yolo_dir / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.is_file():
                continue
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            h, w = img.shape[:2]
            for idx, line in enumerate(lbl_path.read_text().strip().splitlines()):
                cid, box = yolo_line_to_bbox(line, w, h)
                if cid is None or cid >= len(names):
                    continue
                x1, y1, x2, y2 = box
                if x2 - x1 < min_side or y2 - y1 < min_side:
                    continue
                crop = img[y1:y2, x1:x2]
                out_name = f"{img_path.stem}_{idx}.jpg"
                out_path = class_dirs[cid] / out_name
                try:
                    Image.fromarray(crop).save(out_path, quality=95)
                except Exception:
                    continue
                count_per_class[cid] = count_per_class.get(cid, 0) + 1

    total = sum(count_per_class.values())
    print(f"Wrote {total} crops to {output_dir}")
    for i in range(len(names)):
        print(f"  {names[i]}: {count_per_class.get(i, 0)} crops")
    return total


def main():
    p = argparse.ArgumentParser(description="Build class-per-folder crops from Kaggle YOLO for embedding training")
    p.add_argument("--yolo-dir", type=str, default="data/yolo_signs", help="Kaggle YOLO dataset root (train/images, train/labels, ...)")
    p.add_argument("--output-dir", type=str, default="data/kaggle_crops", help="Output directory (class subdirs with crop images)")
    p.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Which splits to use (train val test)")
    args = p.parse_args()
    build_crops(Path(args.yolo_dir), Path(args.output_dir), tuple(args.splits))


if __name__ == "__main__":
    main()
