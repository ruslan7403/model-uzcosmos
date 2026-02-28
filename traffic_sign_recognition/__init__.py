"""FaceID-like Traffic Sign Recognition System.

A similarity-based traffic sign recognition system that uses learned embeddings
to identify signs. Supports open-set recognition and incremental learning
without retraining.
"""

from traffic_sign_recognition.model import EmbeddingNet
from traffic_sign_recognition.gallery import SignGallery
from traffic_sign_recognition.recognizer import TrafficSignRecognizer

__all__ = ["EmbeddingNet", "SignGallery", "TrafficSignRecognizer"]
