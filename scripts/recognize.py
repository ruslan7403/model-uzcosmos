#!/usr/bin/env python3
"""Recognition script for identifying traffic signs using a trained model.

Usage:
    python scripts/recognize.py --model output/best_model.pth \
        --gallery output/gallery --image test_image.jpg
"""

import argparse

from traffic_sign_recognition.recognizer import TrafficSignRecognizer


def main():
    parser = argparse.ArgumentParser(
        description="Recognize traffic signs using a trained model"
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
        "--image", type=str, required=True,
        help="Path to the image to recognize",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Show top-K most similar classes")

    args = parser.parse_args()

    recognizer = TrafficSignRecognizer.load(
        model_path=args.model,
        gallery_path=args.gallery,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )

    predicted_class, score, all_scores = recognizer.recognize_file(args.image)

    print(f"\nImage: {args.image}")
    print(f"Gallery: {recognizer.gallery.num_classes} classes, "
          f"{recognizer.gallery.total_prototypes()} prototypes")
    print("-" * 50)

    if predicted_class is None:
        print(f"Result: UNKNOWN (best score: {score:.4f})")
        print(f"Threshold: {recognizer.gallery.similarity_threshold}")
    else:
        print(f"Result: {predicted_class}")
        print(f"Confidence: {score:.4f}")

    # Show top-K
    if all_scores:
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop-{min(args.top_k, len(sorted_scores))} matches:")
        for cls, s in sorted_scores[:args.top_k]:
            marker = " <--" if cls == predicted_class else ""
            print(f"  {cls}: {s:.4f}{marker}")


if __name__ == "__main__":
    main()
