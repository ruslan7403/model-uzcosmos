"""Traffic sign detector using YOLOv8.

Wraps a YOLO model to detect traffic signs in full scene images and return
bounding boxes. Supports both pre-trained COCO models (for stop signs) and
custom-trained traffic sign models.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class Detection:
    """A single detected traffic sign bounding box.

    Attributes:
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        confidence: Detection confidence score (0-1).
        detector_class: Class name from the detector (e.g., "stop sign").
    """
    bbox: tuple[int, int, int, int]
    confidence: float
    detector_class: str = ""


class TrafficSignDetector:
    """YOLOv8-based traffic sign detector.

    Uses ultralytics YOLOv8 to detect traffic signs in scene images.
    Can use either the COCO-pretrained model (detects stop signs, traffic
    lights) or a custom model trained on traffic sign data.

    Args:
        model_path: Path to YOLO model weights, or a model name like
            "yolov8n.pt" to auto-download. Defaults to "yolov8n.pt".
        confidence_threshold: Minimum detection confidence.
        target_classes: List of COCO class names to detect. If None,
            defaults to traffic-related classes.
        device: Torch device for inference.
    """

    # COCO classes relevant to traffic signs
    TRAFFIC_COCO_CLASSES = {"stop sign", "traffic light", "parking meter"}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        target_classes: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for detection. "
                "Install it with: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = set(target_classes) if target_classes else None

        # Get class name mapping from the model
        self._class_names = self.model.names  # {id: name}

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect traffic signs in an image.

        Args:
            image: PIL Image (RGB) of a scene.

        Returns:
            List of Detection objects for each found sign.
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self._class_names.get(cls_id, str(cls_id))
                conf = float(boxes.conf[i].item())

                # Filter by target classes if specified
                if self.target_classes and cls_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    detector_class=cls_name,
                ))

        return detections

    def detect_file(self, image_path: str) -> list[Detection]:
        """Detect traffic signs in an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            List of Detection objects.
        """
        image = Image.open(image_path).convert("RGB")
        return self.detect(image)


class AllObjectDetector:
    """Detects all objects in an image (no class filtering).

    Useful when using a custom-trained model where every detection is
    a traffic sign, so no filtering is needed.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for detection. "
                "Install it with: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._class_names = self.model.names

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect all objects in an image.

        Args:
            image: PIL Image (RGB).

        Returns:
            List of Detection objects.
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self._class_names.get(cls_id, str(cls_id))
                conf = float(boxes.conf[i].item())

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    detector_class=cls_name,
                ))

        return detections

    def detect_file(self, image_path: str) -> list[Detection]:
        """Detect all objects in an image file."""
        image = Image.open(image_path).convert("RGB")
        return self.detect(image)
