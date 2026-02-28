"""Tests for the traffic sign recognition system.

Uses synthetic images to validate the full pipeline: model, gallery,
dataset, trainer, recognizer, detector, visualizer, and pipeline.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from traffic_sign_recognition.gallery import SignGallery
from traffic_sign_recognition.model import EmbeddingNet, TripletLoss
from traffic_sign_recognition.recognizer import TrafficSignRecognizer
from traffic_sign_recognition.visualize import AnnotatedDetection, draw_detections, draw_detections_with_panel
from traffic_sign_recognition.detector import Detection


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


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_detection_creation(self):
        det = Detection(bbox=(10, 20, 100, 150), confidence=0.85, detector_class="stop sign")
        assert det.bbox == (10, 20, 100, 150)
        assert det.confidence == 0.85
        assert det.detector_class == "stop sign"

    def test_detection_default_class(self):
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.5)
        assert det.detector_class == ""


class TestAnnotatedDetection:
    """Tests for AnnotatedDetection dataclass."""

    def test_annotated_detection_creation(self):
        det = AnnotatedDetection(
            bbox=(10, 20, 100, 150),
            detection_confidence=0.9,
            recognized_class="stop_sign",
            similarity_score=0.87,
            all_scores={"stop_sign": 0.87, "yield": 0.3},
        )
        assert det.recognized_class == "stop_sign"
        assert det.similarity_score == 0.87
        assert len(det.all_scores) == 2

    def test_annotated_detection_unknown(self):
        det = AnnotatedDetection(
            bbox=(0, 0, 50, 50),
            detection_confidence=0.5,
            recognized_class=None,
            similarity_score=0.2,
            all_scores={"stop_sign": 0.2},
        )
        assert det.recognized_class is None


class TestVisualization:
    """Tests for the visualization module."""

    def _make_scene_image(self, width=640, height=480) -> Image.Image:
        """Create a synthetic scene image."""
        img = Image.new("RGB", (width, height), (100, 150, 200))
        draw = ImageDraw.Draw(img)
        # Draw a red circle (fake stop sign)
        draw.ellipse([50, 50, 150, 150], fill=(220, 30, 30), outline="black")
        # Draw a yellow triangle (fake warning)
        draw.polygon([(300, 80), (250, 160), (350, 160)], fill=(255, 200, 0))
        return img

    def test_draw_detections_returns_image(self):
        img = self._make_scene_image()
        detections = [
            AnnotatedDetection(
                bbox=(50, 50, 150, 150),
                detection_confidence=0.95,
                recognized_class="stop_sign",
                similarity_score=0.88,
                all_scores={"stop_sign": 0.88, "yield": 0.2},
            ),
        ]
        result = draw_detections(img, detections)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_draw_detections_multiple(self):
        img = self._make_scene_image()
        detections = [
            AnnotatedDetection(
                bbox=(50, 50, 150, 150),
                detection_confidence=0.95,
                recognized_class="stop_sign",
                similarity_score=0.88,
                all_scores={"stop_sign": 0.88},
            ),
            AnnotatedDetection(
                bbox=(250, 80, 350, 160),
                detection_confidence=0.80,
                recognized_class="warning",
                similarity_score=0.75,
                all_scores={"warning": 0.75},
            ),
        ]
        result = draw_detections(img, detections)
        assert isinstance(result, Image.Image)

    def test_draw_detections_unknown(self):
        img = self._make_scene_image()
        detections = [
            AnnotatedDetection(
                bbox=(50, 50, 150, 150),
                detection_confidence=0.7,
                recognized_class=None,
                similarity_score=0.3,
                all_scores={"stop_sign": 0.3},
            ),
        ]
        result = draw_detections(img, detections, show_unknown=True)
        assert isinstance(result, Image.Image)

    def test_draw_detections_hide_unknown(self):
        img = self._make_scene_image()
        detections = [
            AnnotatedDetection(
                bbox=(50, 50, 150, 150),
                detection_confidence=0.7,
                recognized_class=None,
                similarity_score=0.3,
                all_scores={},
            ),
        ]
        result = draw_detections(img, detections, show_unknown=False)
        assert isinstance(result, Image.Image)

    def test_draw_detections_empty_list(self):
        img = self._make_scene_image()
        result = draw_detections(img, [])
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_draw_detections_with_panel(self):
        img = self._make_scene_image()
        detections = [
            AnnotatedDetection(
                bbox=(50, 50, 150, 150),
                detection_confidence=0.95,
                recognized_class="stop_sign",
                similarity_score=0.88,
                all_scores={"stop_sign": 0.88, "yield": 0.3, "speed_limit": 0.1},
            ),
        ]
        result = draw_detections_with_panel(img, detections, panel_width=250)
        assert isinstance(result, Image.Image)
        # Panel should make the image wider
        assert result.size[0] == img.size[0] + 250

    def test_save_annotated(self):
        from traffic_sign_recognition.visualize import save_annotated

        img = self._make_scene_image()
        detections = [
            AnnotatedDetection(
                bbox=(50, 50, 150, 150),
                detection_confidence=0.9,
                recognized_class="stop_sign",
                similarity_score=0.85,
                all_scores={"stop_sign": 0.85},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "annotated.png")
            result_path = save_annotated(img, detections, out_path)
            assert os.path.exists(result_path)
            saved = Image.open(result_path)
            assert saved.size == img.size

    def test_save_annotated_with_panel(self):
        from traffic_sign_recognition.visualize import save_annotated

        img = self._make_scene_image()
        detections = [
            AnnotatedDetection(
                bbox=(50, 50, 150, 150),
                detection_confidence=0.9,
                recognized_class="stop_sign",
                similarity_score=0.85,
                all_scores={"stop_sign": 0.85},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "annotated_panel.png")
            result_path = save_annotated(img, detections, out_path, with_panel=True)
            assert os.path.exists(result_path)
            saved = Image.open(result_path)
            # Should be wider than original due to panel
            assert saved.size[0] > img.size[0]


class TestPipelineCropLogic:
    """Tests for the detection-recognition pipeline crop logic.

    Tests the pipeline internals without requiring ultralytics installed,
    by mocking the detector.
    """

    def test_crop_detection_basic(self):
        from traffic_sign_recognition.pipeline import DetectionRecognitionPipeline

        model = EmbeddingNet(embedding_dim=64, pretrained=False)
        recognizer = TrafficSignRecognizer(
            model=model,
            gallery=SignGallery(similarity_threshold=0.5),
            device="cpu",
        )
        # Create pipeline with a dummy detector (won't use it for this test)
        pipeline = DetectionRecognitionPipeline.__new__(DetectionRecognitionPipeline)
        pipeline.recognizer = recognizer
        pipeline.crop_padding = 0.1

        img = Image.new("RGB", (640, 480), (100, 100, 100))
        det = Detection(bbox=(100, 100, 200, 200), confidence=0.9)
        crop = pipeline._crop_detection(img, det)

        assert isinstance(crop, Image.Image)
        # Crop should be roughly 100x100 + padding
        assert crop.size[0] > 0
        assert crop.size[1] > 0

    def test_crop_detection_respects_image_bounds(self):
        from traffic_sign_recognition.pipeline import DetectionRecognitionPipeline

        pipeline = DetectionRecognitionPipeline.__new__(DetectionRecognitionPipeline)
        pipeline.crop_padding = 0.5  # large padding

        img = Image.new("RGB", (200, 200), (100, 100, 100))
        # Detection near the edge
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.9)
        crop = pipeline._crop_detection(img, det)

        assert isinstance(crop, Image.Image)
        # Should not exceed image bounds
        assert crop.size[0] <= 200
        assert crop.size[1] <= 200
