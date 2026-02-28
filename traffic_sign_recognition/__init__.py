"""FaceID-like Traffic Sign Recognition System.

A similarity-based traffic sign recognition system that uses learned embeddings
to identify signs. Supports open-set recognition and incremental learning
without retraining. Includes YOLOv8-based object detection for locating signs
in full scene images.
"""

from traffic_sign_recognition.model import EmbeddingNet
from traffic_sign_recognition.gallery import SignGallery
from traffic_sign_recognition.recognizer import TrafficSignRecognizer
from traffic_sign_recognition.visualize import AnnotatedDetection, draw_detections

__all__ = [
    "EmbeddingNet",
    "SignGallery",
    "TrafficSignRecognizer",
    "AnnotatedDetection",
    "draw_detections",
]
