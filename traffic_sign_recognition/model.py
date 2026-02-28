"""Embedding network for traffic sign recognition.

Uses a CNN backbone to map traffic sign images into a compact embedding space
where similar signs are close together and different signs are far apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmbeddingNet(nn.Module):
    """CNN-based embedding network for traffic sign images.

    Maps input images to a normalized embedding vector. Uses a ResNet-18
    backbone with a projection head that outputs L2-normalized embeddings.

    Args:
        embedding_dim: Dimension of the output embedding vector.
        pretrained: Whether to use ImageNet-pretrained backbone weights.
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the final classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        backbone_out_dim = 512

        # Projection head: maps backbone features to embedding space
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embedding from input image.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            L2-normalized embedding of shape (B, embedding_dim).
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embedding = self.projector(features)
        return F.normalize(embedding, p=2, dim=1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward, used during inference."""
        return self.forward(x)


class TripletLoss(nn.Module):
    """Triplet margin loss for metric learning.

    Encourages the distance between anchor-positive pairs to be smaller
    than anchor-negative pairs by at least `margin`.

    Args:
        margin: Minimum margin between positive and negative distances.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
