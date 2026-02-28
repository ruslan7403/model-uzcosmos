#!/usr/bin/env python3
"""Train a YOLOv8 model to detect traffic signs as a single class.

Generates synthetic detection training data by pasting cropped traffic sign
images onto random backgrounds, then fine-tunes YOLOv8n to detect a single
"traffic_sign" class. This replaces the generic COCO model that also detects
people, cars, and other non-sign objects.

Usage:
    python scripts/train_yolo_detector.py \
        --sign-dir data/mapillary \
        --output-dir output \
        --epochs 50
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_sign_images(sign_dir: str) -> list[Path]:
    """Collect all sign images from class subdirectories."""
    images = []
    for class_dir in Path(sign_dir).iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(img_path)
    return images


def generate_random_background(width: int, height: int) -> Image.Image:
    """Generate a random background image that looks like a road scene."""
    bg_type = random.choice(["gradient", "noise", "solid", "sky_road"])

    if bg_type == "gradient":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky-to-road gradient
        for y in range(height):
            t = y / height
            r = int(135 * (1 - t) + 80 * t + random.randint(-20, 20))
            g = int(206 * (1 - t) + 80 * t + random.randint(-20, 20))
            b = int(235 * (1 - t) + 80 * t + random.randint(-20, 20))
            arr[y, :] = [max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))]
        img = Image.fromarray(arr)

    elif bg_type == "noise":
        arr = np.random.randint(60, 200, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=3))

    elif bg_type == "sky_road":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        horizon = height // 2 + random.randint(-height // 6, height // 6)
        # Sky
        sky_color = [
            random.randint(100, 200),
            random.randint(150, 220),
            random.randint(200, 255),
        ]
        arr[:horizon] = sky_color
        # Road/ground
        ground_color = [
            random.randint(50, 120),
            random.randint(50, 120),
            random.randint(50, 120),
        ]
        arr[horizon:] = ground_color
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))

    else:  # solid
        color = tuple(random.randint(40, 200) for _ in range(3))
        img = Image.new("RGB", (width, height), color)

    return img


def paste_sign_on_background(
    bg: Image.Image,
    sign: Image.Image,
    min_scale: float = 0.05,
    max_scale: float = 0.4,
) -> tuple[float, float, float, float] | None:
    """Paste a sign image onto a background at a random position and scale.

    Returns YOLO-format bbox (x_center, y_center, width, height) normalized
    to [0, 1], or None if the sign can't fit.
    """
    bg_w, bg_h = bg.size

    # Random scale relative to background
    scale = random.uniform(min_scale, max_scale)
    sign_w = int(bg_w * scale)
    sign_h = int(sign.height * (sign_w / sign.width))

    if sign_w < 10 or sign_h < 10:
        return None
    if sign_w >= bg_w or sign_h >= bg_h:
        return None

    sign_resized = sign.resize((sign_w, sign_h), Image.LANCZOS)

    # Random position
    x = random.randint(0, bg_w - sign_w)
    y = random.randint(0, bg_h - sign_h)

    # Slight random rotation
    angle = random.uniform(-15, 15)
    sign_rotated = sign_resized.rotate(angle, expand=True, fillcolor=(0, 0, 0))

    # Paste (handle RGBA transparency)
    if sign_rotated.mode == "RGBA":
        bg.paste(sign_rotated, (x, y), sign_rotated)
    else:
        bg.paste(sign_rotated, (x, y))

    # Calculate YOLO bbox (normalized)
    rw, rh = sign_rotated.size
    cx = (x + rw / 2) / bg_w
    cy = (y + rh / 2) / bg_h
    nw = rw / bg_w
    nh = rh / bg_h

    # Clamp to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = min(nw, 1.0)
    nh = min(nh, 1.0)

    return (cx, cy, nw, nh)


def generate_dataset(
    sign_images: list[Path],
    output_dir: str,
    num_train: int = 1000,
    num_val: int = 200,
    img_size: int = 640,
    max_signs_per_image: int = 5,
):
    """Generate a YOLO-format synthetic detection dataset.

    Structure:
        output_dir/
            dataset.yaml
            train/
                images/
                labels/
            val/
                images/
                labels/
    """
    for split, count in [("train", num_train), ("val", num_val)]:
        img_dir = Path(output_dir) / split / "images"
        lbl_dir = Path(output_dir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {count} {split} images...")
        for i in range(count):
            # Random background
            bg = generate_random_background(img_size, img_size)

            # Paste 1-N signs
            n_signs = random.randint(1, max_signs_per_image)
            labels = []

            for _ in range(n_signs):
                sign_path = random.choice(sign_images)
                try:
                    sign = Image.open(sign_path).convert("RGB")
                except Exception:
                    continue

                bbox = paste_sign_on_background(bg, sign)
                if bbox is not None:
                    # Class 0 = traffic_sign
                    labels.append(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

            # Also generate some images with no signs (negative samples, ~10%)
            if random.random() < 0.1:
                labels = []
                bg = generate_random_background(img_size, img_size)

            # Save image and label
            img_name = f"synthetic_{i:05d}.jpg"
            bg.save(str(img_dir / img_name), quality=90)

            lbl_name = f"synthetic_{i:05d}.txt"
            with open(str(lbl_dir / lbl_name), "w") as f:
                f.write("\n".join(labels))

    # Write dataset YAML
    yaml_path = Path(output_dir) / "dataset.yaml"
    yaml_content = (
        f"path: {os.path.abspath(output_dir)}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: traffic_sign\n"
    )
    with open(str(yaml_path), "w") as f:
        f.write(yaml_content)

    print(f"Dataset YAML: {yaml_path}")


def train_yolo(
    dataset_yaml: str,
    output_dir: str,
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    base_model: str = "yolov8n.pt",
):
    """Fine-tune YOLOv8 on the synthetic traffic sign dataset."""
    from ultralytics import YOLO

    model = YOLO(base_model)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir,
        name="traffic_sign_detector",
        exist_ok=True,
        verbose=True,
    )

    # Copy best model to output dir
    best_src = Path(output_dir) / "traffic_sign_detector" / "weights" / "best.pt"
    best_dst = Path(output_dir) / "traffic_sign_yolo.pt"
    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dst))
        print(f"\nBest YOLO model saved to: {best_dst}")
    else:
        # Fallback to last.pt
        last_src = Path(output_dir) / "traffic_sign_detector" / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy2(str(last_src), str(best_dst))
            print(f"\nYOLO model (last) saved to: {best_dst}")

    return str(best_dst)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 to detect traffic signs"
    )
    parser.add_argument(
        "--sign-dir", type=str, required=True,
        help="Directory with class subdirectories of cropped sign images",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Output directory for model and synthetic data (default: output)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="YOLO training epochs (default: 50)",
    )
    parser.add_argument(
        "--num-train", type=int, default=2000,
        help="Number of synthetic training images (default: 2000)",
    )
    parser.add_argument(
        "--num-val", type=int, default=400,
        help="Number of synthetic validation images (default: 400)",
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="YOLO input image size (default: 640)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="YOLO training batch size (default: 16)",
    )
    parser.add_argument(
        "--base-model", type=str, default="yolov8n.pt",
        help="Base YOLO model to fine-tune (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--max-signs", type=int, default=5,
        help="Max signs per synthetic image (default: 5)",
    )

    args = parser.parse_args()

    # Collect sign images
    print(f"Collecting sign images from: {args.sign_dir}")
    sign_images = collect_sign_images(args.sign_dir)
    print(f"Found {len(sign_images)} sign images")

    if len(sign_images) < 10:
        print("ERROR: Need at least 10 sign images to generate training data.")
        return

    # Generate synthetic dataset
    synth_dir = os.path.join(args.output_dir, "yolo_dataset")
    generate_dataset(
        sign_images=sign_images,
        output_dir=synth_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        img_size=args.img_size,
        max_signs_per_image=args.max_signs,
    )

    # Train YOLO
    dataset_yaml = os.path.join(synth_dir, "dataset.yaml")
    print(f"\nTraining YOLOv8 detector...")
    model_path = train_yolo(
        dataset_yaml=dataset_yaml,
        output_dir=args.output_dir,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        base_model=args.base_model,
    )

    # Clean up synthetic data to save disk space
    print(f"\nCleaning up synthetic dataset...")
    shutil.rmtree(synth_dir, ignore_errors=True)

    print(f"\nDone! Use the trained detector:")
    print(f"  python scripts/detect_and_recognize.py \\")
    print(f"    --image scene.jpg \\")
    print(f"    --yolo-model {model_path} \\")
    print(f"    --embedding-model output/best_model.pth \\")
    print(f"    --gallery output/gallery \\")
    print(f"    --detect-all")


if __name__ == "__main__":
    main()
