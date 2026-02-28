"""Training pipeline for the traffic sign embedding model.

Trains the EmbeddingNet using triplet loss on a directory of traffic sign
images organized by class.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from traffic_sign_recognition.dataset import (
    ClassImageDataset,
    TripletTrafficSignDataset,
    get_eval_transforms,
    get_train_transforms,
)
from traffic_sign_recognition.gallery import SignGallery
from traffic_sign_recognition.model import EmbeddingNet, TripletLoss


def train(
    data_dir: str,
    output_dir: str = "output",
    embedding_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    margin: float = 0.3,
    triplets_per_epoch: int = 2000,
    image_size: int = 224,
    device: str = None,
    gallery_threshold: float = 0.6,
) -> tuple[EmbeddingNet, SignGallery]:
    """Train the embedding model and build a gallery from training data.

    Args:
        data_dir: Directory containing class subdirectories of images.
        output_dir: Directory to save model and gallery.
        embedding_dim: Embedding vector dimension.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        margin: Triplet loss margin.
        triplets_per_epoch: Number of triplets per epoch.
        image_size: Input image size.
        device: Torch device (auto-detected if None).
        gallery_threshold: Similarity threshold for gallery matching.

    Returns:
        Tuple of (trained model, populated gallery).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model, loss, optimizer
    model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=True).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create training dataset
    train_dataset = TripletTrafficSignDataset(
        data_dir=data_dir,
        transform=get_train_transforms(image_size),
        triplets_per_epoch=triplets_per_epoch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
    print(f"Training for {epochs} epochs with {triplets_per_epoch} triplets/epoch")

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True
        )
        for anchor, positive, negative in progress:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    # Build gallery from training data
    print("\nBuilding gallery from training data...")
    gallery = build_gallery(model, data_dir, device, image_size, gallery_threshold)
    gallery.save(os.path.join(output_dir, "gallery"))

    print(f"Gallery built with {gallery.num_classes} classes, "
          f"{gallery.total_prototypes()} total prototypes")

    return model, gallery


def build_gallery(
    model: EmbeddingNet,
    data_dir: str,
    device: str = "cpu",
    image_size: int = 224,
    similarity_threshold: float = 0.6,
) -> SignGallery:
    """Build a gallery by embedding all images in a data directory.

    Args:
        model: Trained EmbeddingNet.
        data_dir: Directory with class subdirectories of images.
        device: Torch device.
        image_size: Input image size.
        similarity_threshold: Gallery similarity threshold.

    Returns:
        Populated SignGallery.
    """
    model.eval()
    gallery = SignGallery(similarity_threshold=similarity_threshold)

    dataset = ClassImageDataset(data_dir=data_dir, transform=get_eval_transforms(image_size))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    with torch.no_grad():
        for images, class_names in tqdm(loader, desc="Building gallery"):
            images = images.to(device)
            embeddings = model.get_embedding(images).cpu().numpy()
            for emb, cls_name in zip(embeddings, class_names):
                gallery.add_embedding(cls_name, emb)

    return gallery


def evaluate(
    model: EmbeddingNet,
    gallery: SignGallery,
    test_dir: str,
    device: str = "cpu",
    image_size: int = 224,
) -> dict:
    """Evaluate recognition accuracy on a test set.

    Args:
        model: Trained EmbeddingNet.
        gallery: Populated SignGallery.
        test_dir: Directory with class subdirectories of test images.
        device: Torch device.
        image_size: Input image size.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    dataset = ClassImageDataset(data_dir=test_dir, transform=get_eval_transforms(image_size))

    correct = 0
    total = 0
    unknown_count = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            image, true_class = dataset[i]
            image = image.unsqueeze(0).to(device)
            embedding = model.get_embedding(image).cpu().numpy().flatten()

            predicted_class, score, _ = gallery.query(embedding)
            total += 1
            if predicted_class is None:
                unknown_count += 1
            elif predicted_class == true_class:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_count / total if total > 0 else 0.0,
    }
