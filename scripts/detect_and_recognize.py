#!/usr/bin/env python3
"""Detect traffic signs in a scene image, recognize each one, and visualize.

This is the main end-user script. It takes a scene photo, detects all traffic
signs with YOLOv8, recognizes each using the embedding gallery, and outputs
an annotated image with bounding boxes, class names, and similarity scores.

Usage:
    python scripts/detect_and_recognize.py \
        --image scene.jpg \
        --embedding-model output/best_model.pth \
        --gallery output/gallery \
        --output result.jpg

    # With a custom YOLO model trained on traffic signs:
    python scripts/detect_and_recognize.py \
        --image scene.jpg \
        --yolo-model traffic_signs_yolo.pt \
        --embedding-model output/best_model.pth \
        --gallery output/gallery \
        --detect-all
"""

import argparse
import os
import sys

from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Detect and recognize traffic signs in scene images"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input scene image",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save annotated output image (default: <input>_detected.<ext>)",
    )
    parser.add_argument(
        "--yolo-model", type=str, default="yolov8n.pt",
        help="Path to YOLO model weights (default: yolov8n.pt, auto-downloads)",
    )
    parser.add_argument(
        "--embedding-model", type=str, required=True,
        help="Path to trained EmbeddingNet weights (.pth)",
    )
    parser.add_argument(
        "--gallery", type=str, required=True,
        help="Base path for gallery files (without extension)",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128,
        help="Embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--detection-conf", type=float, default=0.25,
        help="YOLO detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--detect-all", action="store_true",
        help="Detect all object classes (use with custom-trained YOLO models)",
    )
    parser.add_argument(
        "--target-classes", type=str, nargs="+", default=None,
        help="COCO class names to detect (e.g., 'stop sign' 'traffic light')",
    )
    parser.add_argument(
        "--with-panel", action="store_true", default=True,
        help="Add side panel with detailed similarity scores (default: True)",
    )
    parser.add_argument(
        "--no-panel", action="store_true",
        help="Disable side panel",
    )
    parser.add_argument(
        "--line-width", type=int, default=3,
        help="Bounding box line width (default: 3)",
    )
    parser.add_argument(
        "--font-size", type=int, default=16,
        help="Label font size (default: 16)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--crop-padding", type=float, default=0.1,
        help="Padding around detected crops as a fraction (default: 0.1)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    with_panel = args.with_panel and not args.no_panel

    # Determine output path
    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.image)
        output_path = f"{base}_detected{ext or '.jpg'}"

    # Build the pipeline
    if args.detect_all:
        from traffic_sign_recognition.detector import AllObjectDetector
        detector = AllObjectDetector(
            model_path=args.yolo_model,
            confidence_threshold=args.detection_conf,
            device=args.device,
        )
    else:
        from traffic_sign_recognition.detector import TrafficSignDetector
        target = args.target_classes
        if target is None:
            target = list(TrafficSignDetector.TRAFFIC_COCO_CLASSES)
        detector = TrafficSignDetector(
            model_path=args.yolo_model,
            confidence_threshold=args.detection_conf,
            target_classes=target,
            device=args.device,
        )

    from traffic_sign_recognition.recognizer import TrafficSignRecognizer
    recognizer = TrafficSignRecognizer.load(
        model_path=args.embedding_model,
        gallery_path=args.gallery,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )

    from traffic_sign_recognition.pipeline import DetectionRecognitionPipeline
    pipeline = DetectionRecognitionPipeline(
        detector=detector,
        recognizer=recognizer,
        crop_padding=args.crop_padding,
    )

    # Process image
    print(f"Processing: {args.image}")
    image = Image.open(args.image).convert("RGB")

    annotated_img, detections = pipeline.process_and_visualize(
        image,
        output_path=output_path,
        with_panel=with_panel,
        line_width=args.line_width,
        font_size=args.font_size,
    )

    # Print results
    print(f"\nDetected {len(detections)} traffic sign(s):")
    print("-" * 60)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        cls_name = det.recognized_class or "UNKNOWN"
        print(f"  [{i + 1}] {cls_name}")
        print(f"      Bounding box: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"      Detection confidence: {det.detection_confidence:.4f}")
        print(f"      Similarity score: {det.similarity_score:.4f}")
        if det.all_scores:
            top3 = sorted(det.all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"      Top matches: {', '.join(f'{n}={s:.3f}' for n, s in top3)}")
        print()

    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
