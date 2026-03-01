"""Training pipeline for the traffic sign embedding model.

Trains the EmbeddingNet using triplet loss on a directory of traffic sign
images organized by class. Supports checkpointing and time-limited training
for use in CI/CD workflows with run-time limits.
"""

import json
import os
import time
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


def save_checkpoint(
    path: str,
    model: EmbeddingNet,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    total_epochs: int,
    embedding_dim: int,
    learning_rate: float,
    margin: float,
) -> None:
    """Save a full training checkpoint for resuming later.

    Args:
        path: File path to save the checkpoint (.pt).
        model: Current model state.
        optimizer: Current optimizer state.
        scheduler: Current scheduler state.
        epoch: Last completed epoch (0-indexed).
        best_loss: Best loss seen so far.
        total_epochs: Target number of epochs.
        embedding_dim: Model embedding dimension.
        learning_rate: Initial learning rate.
        margin: Triplet loss margin.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "total_epochs": total_epochs,
        "embedding_dim": embedding_dim,
        "learning_rate": learning_rate,
        "margin": margin,
    }, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file (.pt).
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary with all saved state.
    """
    return torch.load(path, map_location=device, weights_only=False)


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
    checkpoint_path: str = None,
    time_limit_seconds: float = None,
    checkpoint_interval: int = 1,
) -> tuple[EmbeddingNet, SignGallery]:
    """Train the embedding model and build a gallery from training data.

    Supports resuming from a checkpoint and stopping after a time limit,
    making it suitable for CI/CD environments with run-time caps.

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
        checkpoint_path: Path to a checkpoint file to resume from.
        time_limit_seconds: Maximum training time in seconds. Training
            stops gracefully after this limit. None means no limit.
        checkpoint_interval: Save a checkpoint every N epochs.

    Returns:
        Tuple of (trained model, populated gallery).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    start_epoch = 0
    best_loss = float("inf")

    # Initialize model, loss, optimizer
    model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=True).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Resume from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        # Keep current run's epochs (do not override with checkpoint's total_epochs)
        print(f"  Resumed at epoch {start_epoch + 1}/{epochs}, "
              f"best_loss={best_loss:.4f}")

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
    print(f"Training epochs {start_epoch + 1} to {epochs} "
          f"with {triplets_per_epoch} triplets/epoch")
    if time_limit_seconds:
        print(f"Time limit: {time_limit_seconds:.0f}s "
              f"({time_limit_seconds / 3600:.1f}h)")

    # Training loop
    train_start_time = time.time()
    stopped_early = False
    last_completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        # Check time limit before starting a new epoch
        if time_limit_seconds:
            elapsed = time.time() - train_start_time
            if elapsed >= time_limit_seconds:
                print(f"\nTime limit reached ({elapsed:.0f}s). "
                      f"Stopping at epoch {epoch}/{epochs}.")
                stopped_early = True
                break

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

            # Check time limit mid-epoch for very long epochs
            if time_limit_seconds:
                elapsed = time.time() - train_start_time
                if elapsed >= time_limit_seconds:
                    print(f"\nTime limit reached mid-epoch ({elapsed:.0f}s).")
                    stopped_early = True
                    break

        if stopped_early:
            # Save checkpoint even on early stop
            save_checkpoint(
                os.path.join(output_dir, "checkpoint.pt"),
                model, optimizer, scheduler,
                last_completed_epoch,
                best_loss, epochs, embedding_dim, learning_rate, margin,
            )
            break

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        last_completed_epoch = epoch
        elapsed = time.time() - train_start_time
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Avg Loss: {avg_loss:.4f} - "
              f"Elapsed: {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pth"))

        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                os.path.join(output_dir, "checkpoint.pt"),
                model, optimizer, scheduler,
                epoch, best_loss, epochs,
                embedding_dim, learning_rate, margin,
            )

    # Save training status
    completed = not stopped_early and (last_completed_epoch + 1 >= epochs)
    status = {
        "completed": completed,
        "epochs_done": last_completed_epoch + 1,
        "total_epochs": epochs,
        "best_loss": float(best_loss),
        "elapsed_seconds": time.time() - train_start_time,
    }
    with open(os.path.join(output_dir, "training_status.json"), "w") as f:
        json.dump(status, f, indent=2)

    print(f"\nTraining status: {'COMPLETED' if completed else 'PAUSED'}")
    print(f"  Epochs done: {status['epochs_done']}/{epochs}")
    print(f"  Best loss: {best_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    # Build gallery from training data (skip if already present, e.g. after redeploy)
    gallery_path = os.path.join(output_dir, "gallery")
    if os.path.exists(f"{gallery_path}.json") and os.path.exists(f"{gallery_path}.npz"):
        print("\nGallery already exists at output, skipping build.")
        gallery = SignGallery.load(gallery_path)
    else:
        print("\nBuilding gallery from training data...")
        gallery = build_gallery(model, data_dir, device, image_size, gallery_threshold)
        gallery.save(gallery_path)

    print(f"Gallery: {gallery.num_classes} classes, "
          f"{gallery.total_prototypes()} total prototypes")

    return model, gallery


def build_gallery(
    model: EmbeddingNet,
    data_dir: str,
    device: str = "cpu",
    image_size: int = 224,
    similarity_threshold: float = 0.6,
    max_samples: int | None = None,
) -> SignGallery:
    """Build a gallery by embedding all images in a data directory.

    Args:
        model: Trained EmbeddingNet.
        data_dir: Directory with class subdirectories of images.
        device: Torch device.
        image_size: Input image size.
        similarity_threshold: Gallery similarity threshold.
        max_samples: If set, use at most this many images (for quick testing).

    Returns:
        Populated SignGallery.
    """
    model.eval()
    gallery = SignGallery(similarity_threshold=similarity_threshold)

    dataset = ClassImageDataset(data_dir=data_dir, transform=get_eval_transforms(image_size))
    if max_samples is not None and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
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
