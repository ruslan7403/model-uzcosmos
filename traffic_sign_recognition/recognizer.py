"""High-level traffic sign recognizer combining model and gallery.

Provides a unified interface for embedding images, enrolling new classes,
and recognizing traffic signs.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from traffic_sign_recognition.dataset import get_eval_transforms
from traffic_sign_recognition.gallery import SignGallery
from traffic_sign_recognition.model import EmbeddingNet


class TrafficSignRecognizer:
    """End-to-end traffic sign recognition system.

    Combines the embedding model with a gallery of known sign prototypes
    to perform recognition, enrollment, and incremental learning.

    Args:
        model: Trained EmbeddingNet model.
        gallery: SignGallery with enrolled prototypes.
        device: Torch device for inference.
        image_size: Expected input image size.
    """

    def __init__(
        self,
        model: EmbeddingNet,
        gallery: Optional[SignGallery] = None,
        device: str = "cpu",
        image_size: int = 224,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.gallery = gallery or SignGallery()
        self.device = device
        self.transform = get_eval_transforms(image_size)

    @classmethod
    def load(
        cls,
        model_path: str,
        gallery_path: Optional[str] = None,
        embedding_dim: int = 128,
        device: str = "cpu",
    ) -> "TrafficSignRecognizer":
        """Load a recognizer from saved model and gallery files.

        Args:
            model_path: Path to saved model weights (.pth).
            gallery_path: Base path for gallery files (without extension).
            embedding_dim: Embedding dimension of the model.
            device: Torch device.

        Returns:
            Loaded TrafficSignRecognizer instance.
        """
        model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        gallery = None
        if gallery_path and Path(f"{gallery_path}.json").exists():
            gallery = SignGallery.load(gallery_path)

        return cls(model=model, gallery=gallery, device=device)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Compute embedding for a single PIL image.

        Args:
            image: PIL Image (RGB).

        Returns:
            1-D numpy embedding array.
        """
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_embedding(tensor)
        return embedding.cpu().numpy().flatten()

    def embed_image_file(self, image_path: str) -> np.ndarray:
        """Compute embedding for an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            1-D numpy embedding array.
        """
        image = Image.open(image_path).convert("RGB")
        return self.embed_image(image)

    def recognize(
        self, image: Image.Image
    ) -> tuple[Optional[str], float, dict[str, float]]:
        """Recognize a traffic sign in the given image.

        Args:
            image: PIL Image containing a traffic sign.

        Returns:
            Tuple of (predicted_class, confidence, all_scores).
            predicted_class is None if the sign is unknown.
        """
        embedding = self.embed_image(image)
        return self.gallery.query(embedding)

    def recognize_file(
        self, image_path: str
    ) -> tuple[Optional[str], float, dict[str, float]]:
        """Recognize a traffic sign from an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (predicted_class, confidence, all_scores).
        """
        image = Image.open(image_path).convert("RGB")
        return self.recognize(image)

    def enroll_class(
        self, class_name: str, image_paths: list[str]
    ) -> int:
        """Enroll a new traffic sign class using reference images.

        This is the core of incremental learning: new classes are added
        by simply computing and storing their embeddings. No retraining
        of the model is required.

        Args:
            class_name: Name for the new traffic sign class.
            image_paths: Paths to reference images of this sign.

        Returns:
            Number of embeddings added to the gallery.
        """
        count = 0
        for path in image_paths:
            embedding = self.embed_image_file(path)
            self.gallery.add_embedding(class_name, embedding)
            count += 1
        return count

    def enroll_from_directory(self, class_name: str, directory: str) -> int:
        """Enroll a new class from all images in a directory.

        Args:
            class_name: Name for the new traffic sign class.
            directory: Directory containing reference images.

        Returns:
            Number of embeddings added.
        """
        dir_path = Path(directory)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            str(p) for p in dir_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        return self.enroll_class(class_name, image_paths)

    def save(self, model_path: str, gallery_path: str) -> None:
        """Save the model and gallery to disk.

        Args:
            model_path: Path for model weights (.pth).
            gallery_path: Base path for gallery files (without extension).
        """
        torch.save(self.model.state_dict(), model_path)
        self.gallery.save(gallery_path)
