#!/usr/bin/env python3
"""Build gallery from a trained model and dataset. Use before recognition if you have no gallery.

Usage:
    python scripts/build_gallery.py --model output/best_model.pth \
        --data-dir data/mapillary --output output/gallery
"""
import argparse
import torch
from traffic_sign_recognition.model import EmbeddingNet
from traffic_sign_recognition.trainer import build_gallery


def main():
    p = argparse.ArgumentParser(description="Build gallery from model + dataset")
    p.add_argument("--model", required=True, help="Path to best_model.pth")
    p.add_argument("--data-dir", default="data/mapillary", help="Dataset with class subdirs")
    p.add_argument("--output", default="output/gallery", help="Base path for gallery.json + .npz")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Use at most N images (for quick testing)")
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingNet(embedding_dim=args.embedding_dim)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    gallery = build_gallery(
        model, args.data_dir, device=device,
        image_size=args.image_size, similarity_threshold=args.threshold,
        max_samples=args.limit,
    )
    gallery.save(args.output)
    print(f"Saved gallery: {args.output}.json + .npz")
    print(f"Classes: {gallery.num_classes}, Prototypes: {gallery.total_prototypes()}")


if __name__ == "__main__":
    main()
