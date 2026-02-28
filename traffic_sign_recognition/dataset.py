"""Dataset and data loading utilities for traffic sign embedding training.

Provides a triplet dataset that generates (anchor, positive, negative) tuples
from a directory of traffic sign images organized by class, along with
standard image transforms.
"""

import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with augmentation.

    Args:
        image_size: Target image size (square).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Evaluation transforms (deterministic, no augmentation).

    Args:
        image_size: Target image size (square).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TripletTrafficSignDataset(Dataset):
    """Dataset that generates triplets (anchor, positive, negative) for training.

    Expects a directory structure where each subdirectory is a class:
        data_dir/
            stop_sign/
                img1.jpg
                img2.jpg
            speed_limit_50/
                img1.jpg
                ...

    Each __getitem__ call returns a triplet where anchor and positive belong
    to the same class, and negative belongs to a different class.

    Args:
        data_dir: Root directory containing class subdirectories.
        transform: Image transforms to apply.
        triplets_per_epoch: Number of triplets to generate per epoch.
    """

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose = None,
        triplets_per_epoch: int = 1000,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform or get_train_transforms()
        self.triplets_per_epoch = triplets_per_epoch

        # Build class-to-images mapping
        self.class_to_images: dict[str, list[Path]] = {}
        self.classes: list[str] = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            images = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            if len(images) >= 2:  # Need at least 2 images to form a positive pair
                self.class_to_images[class_dir.name] = images
                self.classes.append(class_dir.name)

        if len(self.classes) < 2:
            raise ValueError(
                f"Need at least 2 classes with >= 2 images each. "
                f"Found {len(self.classes)} in {data_dir}"
            )

    def __len__(self) -> int:
        return self.triplets_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pick anchor class and a different negative class
        anchor_class = random.choice(self.classes)
        negative_class = random.choice([c for c in self.classes if c != anchor_class])

        # Pick anchor and positive from the same class
        anchor_img_path, positive_img_path = random.sample(
            self.class_to_images[anchor_class], 2
        )
        # Pick negative from a different class
        negative_img_path = random.choice(self.class_to_images[negative_class])

        anchor = self._load_image(anchor_img_path)
        positive = self._load_image(positive_img_path)
        negative = self._load_image(negative_img_path)

        return anchor, positive, negative

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class ClassImageDataset(Dataset):
    """Simple dataset that loads images with class labels.

    Used for gallery enrollment and evaluation.

    Args:
        data_dir: Root directory containing class subdirectories.
        transform: Image transforms to apply.
    """

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform or get_eval_transforms()

        self.samples: list[tuple[Path, str]] = []
        self.classes: list[str] = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            self.classes.append(class_dir.name)
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, class_dir.name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path, class_name = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_name
