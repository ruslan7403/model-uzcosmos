#!/usr/bin/env python3
"""Demo script showing the full traffic sign recognition workflow.

Generates synthetic traffic sign data, trains the embedding model,
builds a gallery, performs recognition, and demonstrates incremental
enrollment of new classes.
"""

import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw

from traffic_sign_recognition.gallery import SignGallery
from traffic_sign_recognition.model import EmbeddingNet
from traffic_sign_recognition.recognizer import TrafficSignRecognizer
from traffic_sign_recognition.trainer import build_gallery, train


def create_synthetic_sign(shape: str, color: tuple, bg_color: tuple = (255, 255, 255),
                          size: int = 96, variation: float = 0.0) -> Image.Image:
    """Create a synthetic traffic sign image.

    Args:
        shape: One of "circle", "triangle", "square", "diamond".
        color: RGB fill color.
        bg_color: RGB background color.
        size: Image size in pixels.
        variation: Amount of random variation (0.0-1.0).
    """
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    margin = int(size * 0.15)
    noise = int(variation * size * 0.05)

    def jitter():
        return np.random.randint(-noise, noise + 1) if noise > 0 else 0

    if shape == "circle":
        draw.ellipse(
            [margin + jitter(), margin + jitter(),
             size - margin + jitter(), size - margin + jitter()],
            fill=color, outline="black", width=2,
        )
    elif shape == "triangle":
        points = [
            (size // 2 + jitter(), margin + jitter()),
            (margin + jitter(), size - margin + jitter()),
            (size - margin + jitter(), size - margin + jitter()),
        ]
        draw.polygon(points, fill=color, outline="black", width=2)
    elif shape == "square":
        draw.rectangle(
            [margin + jitter(), margin + jitter(),
             size - margin + jitter(), size - margin + jitter()],
            fill=color, outline="black", width=2,
        )
    elif shape == "diamond":
        cx, cy = size // 2, size // 2
        r = size // 2 - margin
        points = [
            (cx + jitter(), cy - r + jitter()),
            (cx + r + jitter(), cy + jitter()),
            (cx + jitter(), cy + r + jitter()),
            (cx - r + jitter(), cy + jitter()),
        ]
        draw.polygon(points, fill=color, outline="black", width=2)

    return img


def create_demo_dataset(root: str):
    """Create a synthetic traffic sign dataset for demo purposes."""
    sign_specs = {
        "stop_sign": ("circle", (220, 30, 30)),
        "warning_triangle": ("triangle", (255, 200, 0)),
        "speed_limit": ("circle", (240, 240, 240)),
        "info_square": ("square", (30, 100, 220)),
        "priority_diamond": ("diamond", (255, 200, 0)),
    }

    for class_name, (shape, color) in sign_specs.items():
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(15):
            variation = 0.2 + (i * 0.05)
            bg_brightness = np.random.randint(200, 255)
            bg = (bg_brightness,) * 3
            img = create_synthetic_sign(shape, color, bg, variation=min(variation, 1.0))
            img.save(os.path.join(class_dir, f"sample_{i:03d}.png"))


def main():
    print("=" * 60)
    print("FaceID-like Traffic Sign Recognition System - Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "train_data")
        output_dir = os.path.join(tmpdir, "output")

        # Step 1: Generate synthetic dataset
        print("\n[Step 1] Generating synthetic traffic sign dataset...")
        create_demo_dataset(data_dir)
        classes = sorted(os.listdir(data_dir))
        print(f"  Created {len(classes)} classes: {classes}")
        for c in classes:
            n = len(os.listdir(os.path.join(data_dir, c)))
            print(f"    {c}: {n} images")

        # Step 2: Train embedding model
        print("\n[Step 2] Training embedding model...")
        model, gallery = train(
            data_dir=data_dir,
            output_dir=output_dir,
            embedding_dim=128,
            epochs=5,
            batch_size=16,
            triplets_per_epoch=200,
            image_size=224,
            gallery_threshold=0.5,
        )
        print(f"  Gallery: {gallery.num_classes} classes, "
              f"{gallery.total_prototypes()} prototypes")

        # Step 3: Test recognition
        print("\n[Step 3] Testing recognition on known signs...")
        recognizer = TrafficSignRecognizer(
            model=model, gallery=gallery, device="cpu"
        )

        for class_name in classes[:3]:
            test_img_path = os.path.join(data_dir, class_name, "sample_000.png")
            pred, score, _ = recognizer.recognize_file(test_img_path)
            status = "CORRECT" if pred == class_name else f"WRONG (got {pred})"
            print(f"  {class_name}: predicted={pred}, score={score:.4f} [{status}]")

        # Step 4: Test unknown sign detection
        print("\n[Step 4] Testing unknown sign detection...")
        recognizer.gallery.similarity_threshold = 0.85
        unknown_img = create_synthetic_sign("circle", (128, 0, 128))
        pred, score, _ = recognizer.recognize(unknown_img)
        print(f"  Unknown purple circle: predicted={pred}, score={score:.4f}")
        if pred is None:
            print("  Successfully detected as UNKNOWN")
        else:
            print(f"  Matched to {pred} (threshold may need tuning)")

        # Step 5: Incremental learning - add new class
        print("\n[Step 5] Incremental learning - adding new sign class...")
        new_class_dir = os.path.join(tmpdir, "new_class")
        os.makedirs(new_class_dir)
        for i in range(3):
            img = create_synthetic_sign("diamond", (0, 200, 0), variation=0.1 * i)
            img.save(os.path.join(new_class_dir, f"sample_{i}.png"))

        count = recognizer.enroll_from_directory("green_diamond", new_class_dir)
        print(f"  Enrolled 'green_diamond' with {count} prototypes")
        print(f"  Gallery now has {recognizer.gallery.num_classes} classes, "
              f"{recognizer.gallery.total_prototypes()} prototypes")
        print("  No model retraining was needed!")

        # Test recognition of the new class
        recognizer.gallery.similarity_threshold = 0.5
        test_new = create_synthetic_sign("diamond", (0, 200, 0), variation=0.15)
        pred, score, _ = recognizer.recognize(test_new)
        print(f"  New green diamond: predicted={pred}, score={score:.4f}")

        # Step 6: Save and reload
        print("\n[Step 6] Saving and reloading system...")
        model_path = os.path.join(output_dir, "demo_model.pth")
        gallery_path = os.path.join(output_dir, "demo_gallery")
        recognizer.save(model_path, gallery_path)

        loaded = TrafficSignRecognizer.load(
            model_path=model_path,
            gallery_path=gallery_path,
            embedding_dim=128,
        )
        print(f"  Loaded gallery: {loaded.gallery.num_classes} classes, "
              f"{loaded.gallery.total_prototypes()} prototypes")
        assert "green_diamond" in loaded.gallery.class_names
        print("  Incrementally added class survived save/load!")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
