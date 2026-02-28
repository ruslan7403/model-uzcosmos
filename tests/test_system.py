"""Tests for the traffic sign recognition system.

Uses synthetic images to validate the full pipeline: model, gallery,
dataset, trainer, and recognizer.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from traffic_sign_recognition.gallery import SignGallery
from traffic_sign_recognition.model import EmbeddingNet, TripletLoss
from traffic_sign_recognition.recognizer import TrafficSignRecognizer


def make_synthetic_image(color: tuple[int, int, int], size: int = 64) -> Image.Image:
    """Create a solid-color image for testing."""
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def create_test_dataset(root: str, num_classes: int = 4, images_per_class: int = 5):
    """Create a directory of synthetic images organized by class."""
    colors = [
        (255, 0, 0),    # red
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (0, 255, 0),    # green
        (255, 128, 0),  # orange
        (128, 0, 255),  # purple
    ]
    class_names = []
    for i in range(num_classes):
        class_name = f"sign_class_{i}"
        class_names.append(class_name)
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        base_color = colors[i % len(colors)]
        for j in range(images_per_class):
            # Add slight variation per image
            noise = np.random.randint(-20, 20, size=3)
            color = tuple(np.clip(np.array(base_color) + noise, 0, 255).astype(int))
            img = make_synthetic_image(color)
            img.save(os.path.join(class_dir, f"img_{j}.png"))
    return class_names


class TestEmbeddingNet:
    def test_output_shape(self):
        model = EmbeddingNet(embedding_dim=128, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 128)

    def test_output_normalized(self):
        model = EmbeddingNet(embedding_dim=64, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_get_embedding_same_as_forward(self):
        model = EmbeddingNet(embedding_dim=128, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model.get_embedding(x)
        assert torch.allclose(out1, out2)


class TestTripletLoss:
    def test_zero_loss_when_positive_closer(self):
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[0.9, 0.1]])
        negative = torch.tensor([[-1.0, 0.0]])
        loss_fn = TripletLoss(margin=0.3)
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() == 0.0

    def test_nonzero_loss_when_negative_closer(self):
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[-1.0, 0.0]])
        negative = torch.tensor([[0.9, 0.1]])
        loss_fn = TripletLoss(margin=0.3)
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() > 0.0


class TestSignGallery:
    def test_add_and_query(self):
        gallery = SignGallery(similarity_threshold=0.5)
        gallery.add_embedding("stop", np.array([1.0, 0.0, 0.0]))
        gallery.add_embedding("yield", np.array([0.0, 1.0, 0.0]))

        pred, score, _ = gallery.query(np.array([0.95, 0.05, 0.0]))
        assert pred == "stop"
        assert score > 0.9

    def test_unknown_detection(self):
        gallery = SignGallery(similarity_threshold=0.9)
        gallery.add_embedding("stop", np.array([1.0, 0.0, 0.0]))

        pred, score, _ = gallery.query(np.array([0.0, 1.0, 0.0]))
        assert pred is None

    def test_multiple_prototypes(self):
        gallery = SignGallery(similarity_threshold=0.5)
        gallery.add_embedding("stop", np.array([1.0, 0.0, 0.0]))
        gallery.add_embedding("stop", np.array([0.9, 0.1, 0.0]))
        assert gallery.num_prototypes("stop") == 2

    def test_remove_class(self):
        gallery = SignGallery()
        gallery.add_embedding("stop", np.array([1.0, 0.0]))
        assert gallery.remove_class("stop") is True
        assert gallery.num_classes == 0
        assert gallery.remove_class("stop") is False

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gallery = SignGallery(similarity_threshold=0.7)
            gallery.add_embedding("stop", np.array([1.0, 0.0, 0.0]))
            gallery.add_embedding("yield", np.array([0.0, 1.0, 0.0]))
            gallery.add_embedding("stop", np.array([0.9, 0.1, 0.0]))

            path = os.path.join(tmpdir, "test_gallery")
            gallery.save(path)

            loaded = SignGallery.load(path)
            assert loaded.num_classes == 2
            assert loaded.num_prototypes("stop") == 2
            assert loaded.similarity_threshold == 0.7

            pred, score, _ = loaded.query(np.array([0.95, 0.05, 0.0]))
            assert pred == "stop"

    def test_empty_gallery_returns_none(self):
        gallery = SignGallery()
        pred, score, all_scores = gallery.query(np.array([1.0, 0.0]))
        assert pred is None
        assert score == 0.0
        assert all_scores == {}


class TestTrafficSignRecognizer:
    def test_embed_image(self):
        model = EmbeddingNet(embedding_dim=64, pretrained=False)
        recognizer = TrafficSignRecognizer(model=model, device="cpu")
        img = make_synthetic_image((255, 0, 0))
        emb = recognizer.embed_image(img)
        assert emb.shape == (64,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_recognize_returns_none_with_empty_gallery(self):
        model = EmbeddingNet(embedding_dim=64, pretrained=False)
        recognizer = TrafficSignRecognizer(model=model, device="cpu")
        img = make_synthetic_image((255, 0, 0))
        pred, score, _ = recognizer.recognize(img)
        assert pred is None

    def test_enroll_and_recognize(self):
        model = EmbeddingNet(embedding_dim=64, pretrained=False)
        recognizer = TrafficSignRecognizer(
            model=model,
            gallery=SignGallery(similarity_threshold=0.5),
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save test images
            for i in range(3):
                img = make_synthetic_image((255, 0, 0))
                img.save(os.path.join(tmpdir, f"red_{i}.png"))

            count = recognizer.enroll_class(
                "red_sign",
                [os.path.join(tmpdir, f"red_{i}.png") for i in range(3)],
            )
            assert count == 3
            assert recognizer.gallery.num_classes == 1

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = EmbeddingNet(embedding_dim=64, pretrained=False)
            gallery = SignGallery(similarity_threshold=0.6)
            gallery.add_embedding("test", np.random.randn(64).astype(np.float32))

            recognizer = TrafficSignRecognizer(
                model=model, gallery=gallery, device="cpu"
            )
            model_path = os.path.join(tmpdir, "model.pth")
            gallery_path = os.path.join(tmpdir, "gallery")
            recognizer.save(model_path, gallery_path)

            loaded = TrafficSignRecognizer.load(
                model_path=model_path,
                gallery_path=gallery_path,
                embedding_dim=64,
                device="cpu",
            )
            assert loaded.gallery.num_classes == 1


class TestDataset:
    def test_triplet_dataset(self):
        from traffic_sign_recognition.dataset import TripletTrafficSignDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_classes=3, images_per_class=4)
            dataset = TripletTrafficSignDataset(
                data_dir=tmpdir, triplets_per_epoch=10
            )
            assert len(dataset) == 10
            anchor, positive, negative = dataset[0]
            assert anchor.shape == (3, 224, 224)
            assert positive.shape == (3, 224, 224)
            assert negative.shape == (3, 224, 224)

    def test_class_image_dataset(self):
        from traffic_sign_recognition.dataset import ClassImageDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_classes=2, images_per_class=3)
            dataset = ClassImageDataset(data_dir=tmpdir)
            assert len(dataset) == 6
            img, cls = dataset[0]
            assert img.shape == (3, 224, 224)
            assert isinstance(cls, str)


class TestIncrementalLearning:
    """Tests verifying incremental learning works without retraining."""

    def test_add_new_class_after_initial_gallery(self):
        gallery = SignGallery(similarity_threshold=0.5)
        gallery.add_embedding("stop", np.array([1.0, 0.0, 0.0]))
        gallery.add_embedding("yield", np.array([0.0, 1.0, 0.0]))
        assert gallery.num_classes == 2

        # Add a new class (simulating incremental enrollment)
        gallery.add_embedding("speed_limit", np.array([0.0, 0.0, 1.0]))
        assert gallery.num_classes == 3

        # The new class should be recognizable
        pred, score, _ = gallery.query(np.array([0.05, 0.05, 0.95]))
        assert pred == "speed_limit"
        assert score > 0.8

    def test_incremental_enrollment_through_recognizer(self):
        model = EmbeddingNet(embedding_dim=64, pretrained=False)
        gallery = SignGallery(similarity_threshold=0.3)
        recognizer = TrafficSignRecognizer(
            model=model, gallery=gallery, device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Enroll first class
            for i in range(2):
                img = make_synthetic_image((255, 0, 0))
                img.save(os.path.join(tmpdir, f"red_{i}.png"))
            recognizer.enroll_class(
                "red_sign",
                [os.path.join(tmpdir, f"red_{i}.png") for i in range(2)],
            )

            # Enroll second class (incremental, no retraining)
            for i in range(2):
                img = make_synthetic_image((0, 0, 255))
                img.save(os.path.join(tmpdir, f"blue_{i}.png"))
            recognizer.enroll_class(
                "blue_sign",
                [os.path.join(tmpdir, f"blue_{i}.png") for i in range(2)],
            )

            assert recognizer.gallery.num_classes == 2
            assert recognizer.gallery.num_prototypes("red_sign") == 2
            assert recognizer.gallery.num_prototypes("blue_sign") == 2

    def test_gallery_persistence_across_save_load(self):
        """New classes survive save/load cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gallery = SignGallery(similarity_threshold=0.5)
            gallery.add_embedding("stop", np.array([1.0, 0.0, 0.0]))
            gallery.add_embedding("new_sign", np.array([0.0, 0.0, 1.0]))

            path = os.path.join(tmpdir, "gallery")
            gallery.save(path)
            loaded = SignGallery.load(path)

            assert loaded.num_classes == 2
            assert "new_sign" in loaded.class_names
            pred, score, _ = loaded.query(np.array([0.0, 0.1, 0.95]))
            assert pred == "new_sign"
