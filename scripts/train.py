#!/usr/bin/env python3
"""Training script for the traffic sign embedding model.

Supports checkpointing and time-limited training for CI/CD workflows
(e.g. GitHub Actions 6-hour limit).

Usage:
    # Normal training
    python scripts/train.py --data-dir data/train --output-dir output

    # Time-limited with checkpoint resume (for CI/CD)
    python scripts/train.py --data-dir data/train --output-dir output \
        --time-limit 19800 --checkpoint output/checkpoint.pt
"""

import argparse
import sys

from traffic_sign_recognition.trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Train the traffic sign embedding model"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing class subdirectories of training images",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory to save model and gallery (default: output)",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--triplets-per-epoch", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gallery-threshold", type=float, default=0.6)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Maximum training time in seconds (e.g. 19800 for 5.5 hours)",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1,
        help="Save checkpoint every N epochs (default: 1)",
    )

    args = parser.parse_args()

    model, gallery = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        triplets_per_epoch=args.triplets_per_epoch,
        image_size=args.image_size,
        device=args.device,
        gallery_threshold=args.gallery_threshold,
        checkpoint_path=args.checkpoint,
        time_limit_seconds=args.time_limit,
        checkpoint_interval=args.checkpoint_interval,
    )

    print(f"\nTraining complete!")
    print(f"Model saved to: {args.output_dir}/best_model.pth")
    print(f"Gallery saved to: {args.output_dir}/gallery.json + gallery.npz")
    print(f"Gallery contains {gallery.num_classes} classes with "
          f"{gallery.total_prototypes()} prototypes")


if __name__ == "__main__":
    main()
