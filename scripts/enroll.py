#!/usr/bin/env python3
"""Enrollment script to add new traffic sign classes to the gallery.

Enables incremental learning by adding new classes using only a few images,
without retraining the model.

Usage:
    python scripts/enroll.py --model output/best_model.pth \
        --gallery output/gallery \
        --class-name "new_sign" \
        --images img1.jpg img2.jpg img3.jpg
"""

import argparse

from traffic_sign_recognition.recognizer import TrafficSignRecognizer


def main():
    parser = argparse.ArgumentParser(
        description="Enroll new traffic sign classes into the gallery"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model weights (.pth)",
    )
    parser.add_argument(
        "--gallery", type=str, required=True,
        help="Base path for gallery files (without extension)",
    )
    parser.add_argument(
        "--class-name", type=str, required=True,
        help="Name for the new traffic sign class",
    )
    parser.add_argument(
        "--images", type=str, nargs="+",
        help="Image files to use as prototypes",
    )
    parser.add_argument(
        "--image-dir", type=str,
        help="Directory containing prototype images (alternative to --images)",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    recognizer = TrafficSignRecognizer.load(
        model_path=args.model,
        gallery_path=args.gallery,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )

    print(f"Gallery before: {recognizer.gallery.num_classes} classes, "
          f"{recognizer.gallery.total_prototypes()} prototypes")

    if args.images:
        count = recognizer.enroll_class(args.class_name, args.images)
    elif args.image_dir:
        count = recognizer.enroll_from_directory(args.class_name, args.image_dir)
    else:
        parser.error("Must provide either --images or --image-dir")
        return

    # Save updated gallery
    recognizer.gallery.save(args.gallery)

    print(f"\nEnrolled class '{args.class_name}' with {count} prototype(s)")
    print(f"Gallery after: {recognizer.gallery.num_classes} classes, "
          f"{recognizer.gallery.total_prototypes()} prototypes")
    print(f"Gallery saved to: {args.gallery}.json + {args.gallery}.npz")


if __name__ == "__main__":
    main()
