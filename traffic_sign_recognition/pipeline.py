"""Full detection + recognition pipeline.

Combines YOLOv8 object detection with embedding-based recognition:
1. Detect traffic signs in a scene image (bounding boxes)
2. Crop each detection
3. Embed each crop and match against the gallery
4. Return annotated detections with class names and similarity scores
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from traffic_sign_recognition.detector import Detection, TrafficSignDetector
from traffic_sign_recognition.recognizer import TrafficSignRecognizer
from traffic_sign_recognition.visualize import AnnotatedDetection, draw_detections, save_annotated


class DetectionRecognitionPipeline:
    """End-to-end pipeline: detect signs in a scene, then recognize each one.

    Stage 1 (Detection): A YOLOv8 model finds traffic sign bounding boxes.
    Stage 2 (Recognition): Each cropped sign is embedded and matched against
        the gallery to produce a class label + similarity score.

    Args:
        detector: TrafficSignDetector instance.
        recognizer: TrafficSignRecognizer instance.
        crop_padding: Extra padding (fraction) around each detection crop,
            to include some context and handle tight bounding boxes.
    """

    def __init__(
        self,
        detector: TrafficSignDetector,
        recognizer: TrafficSignRecognizer,
        crop_padding: float = 0.1,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.crop_padding = crop_padding

    @classmethod
    def load(
        cls,
        yolo_model_path: str = "yolov8n.pt",
        embedding_model_path: str = "output/best_model.pth",
        gallery_path: str = "output/gallery",
        embedding_dim: int = 128,
        detection_confidence: float = 0.25,
        target_classes: Optional[list[str]] = None,
        device: str = "cpu",
        crop_padding: float = 0.1,
    ) -> "DetectionRecognitionPipeline":
        """Load a full pipeline from saved models and gallery.

        Args:
            yolo_model_path: Path to YOLO model weights.
            embedding_model_path: Path to EmbeddingNet weights (.pth).
            gallery_path: Base path for gallery files (without extension).
            embedding_dim: Embedding dimension.
            detection_confidence: Minimum YOLO detection confidence.
            target_classes: YOLO class names to detect (None = all).
            device: Torch device.
            crop_padding: Padding fraction for detection crops.

        Returns:
            Loaded DetectionRecognitionPipeline instance.
        """
        # Auto-detect custom YOLO model trained specifically for traffic signs
        use_all_objects = False
        if yolo_model_path == "yolov8n.pt":
            custom_yolo = os.path.join(
                os.path.dirname(embedding_model_path), "traffic_sign_yolo.pt"
            )
            if os.path.exists(custom_yolo):
                yolo_model_path = custom_yolo
                use_all_objects = True  # custom model only detects signs

        if use_all_objects:
            from traffic_sign_recognition.detector import AllObjectDetector
            detector = AllObjectDetector(
                model_path=yolo_model_path,
                confidence_threshold=detection_confidence,
                device=device,
            )
        else:
            detector = TrafficSignDetector(
                model_path=yolo_model_path,
                confidence_threshold=detection_confidence,
                target_classes=target_classes,
                device=device,
            )
        recognizer = TrafficSignRecognizer.load(
            model_path=embedding_model_path,
            gallery_path=gallery_path,
            embedding_dim=embedding_dim,
            device=device,
        )
        return cls(detector=detector, recognizer=recognizer, crop_padding=crop_padding)

    def _crop_detection(
        self, image: Image.Image, detection: Detection
    ) -> Image.Image:
        """Crop a detection from the image with optional padding.

        Args:
            image: Full scene image.
            detection: Detection with bounding box.

        Returns:
            Cropped PIL Image of the detected sign.
        """
        x1, y1, x2, y2 = detection.bbox
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)

        img_w, img_h = image.size
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(img_w, x2 + pad_x)
        cy2 = min(img_h, y2 + pad_y)

        return image.crop((cx1, cy1, cx2, cy2))

    def process_image(
        self, image: Image.Image
    ) -> list[AnnotatedDetection]:
        """Run the full pipeline on a single image.

        1. Detect traffic signs (bounding boxes)
        2. Crop each detection
        3. Embed and recognize each crop
        4. Return annotated detections

        Args:
            image: PIL Image of a scene containing traffic signs.

        Returns:
            List of AnnotatedDetection with bbox, class, and score.
        """
        # Stage 1: Detection
        detections = self.detector.detect(image)

        # Stage 2: Recognition
        annotated = []
        for det in detections:
            crop = self._crop_detection(image, det)
            pred_class, sim_score, all_scores = self.recognizer.recognize(crop)

            annotated.append(AnnotatedDetection(
                bbox=det.bbox,
                detection_confidence=det.confidence,
                recognized_class=pred_class,
                similarity_score=sim_score,
                all_scores=all_scores,
            ))

        return annotated

    def process_image_file(self, image_path: str) -> list[AnnotatedDetection]:
        """Run the full pipeline on an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            List of AnnotatedDetection results.
        """
        image = Image.open(image_path).convert("RGB")
        return self.process_image(image)

    def process_and_visualize(
        self,
        image: Image.Image,
        output_path: Optional[str] = None,
        with_panel: bool = True,
        line_width: int = 3,
        font_size: int = 16,
    ) -> tuple[Image.Image, list[AnnotatedDetection]]:
        """Run pipeline and produce an annotated visualization.

        Args:
            image: Input scene image.
            output_path: If provided, save the annotated image here.
            with_panel: Add a side panel with detailed similarity scores.
            line_width: Bounding box line width.
            font_size: Label text size.

        Returns:
            Tuple of (annotated_image, detections).
        """
        detections = self.process_image(image)

        if with_panel:
            from traffic_sign_recognition.visualize import draw_detections_with_panel
            annotated_img = draw_detections_with_panel(
                image, detections,
                line_width=line_width, font_size=font_size,
            )
        else:
            annotated_img = draw_detections(
                image, detections,
                line_width=line_width, font_size=font_size,
            )

        if output_path:
            annotated_img.save(output_path)

        return annotated_img, detections

    def process_and_visualize_file(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        with_panel: bool = True,
    ) -> tuple[Image.Image, list[AnnotatedDetection]]:
        """Run pipeline on an image file and visualize.

        Args:
            image_path: Path to input image.
            output_path: Path to save annotated image. If None,
                auto-generates a name from the input path.

        Returns:
            Tuple of (annotated_image, detections).
        """
        image = Image.open(image_path).convert("RGB")

        if output_path is None:
            p = Path(image_path)
            output_path = str(p.parent / f"{p.stem}_detected{p.suffix}")

        return self.process_and_visualize(
            image, output_path=output_path, with_panel=with_panel,
        )
