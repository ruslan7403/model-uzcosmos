"""Gallery system for storing and matching traffic sign embeddings.

The gallery stores reference embeddings (prototypes) for known traffic sign
classes. During recognition, a query embedding is compared against all stored
prototypes to find the closest match. This is analogous to the enrolled face
templates in a FaceID system.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class SignGallery:
    """Stores reference embeddings for known traffic sign classes.

    Each class can have multiple prototype embeddings (e.g. from different
    viewpoints or lighting conditions). Recognition is performed by finding
    the nearest prototype using cosine similarity.

    Args:
        similarity_threshold: Minimum cosine similarity for a match.
            Queries below this threshold are reported as unknown.
    """

    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        # {class_name: list of embedding numpy arrays}
        self._gallery: dict[str, list[np.ndarray]] = {}

    @property
    def class_names(self) -> list[str]:
        """Return list of all registered class names."""
        return list(self._gallery.keys())

    @property
    def num_classes(self) -> int:
        """Return number of registered classes."""
        return len(self._gallery)

    def num_prototypes(self, class_name: str) -> int:
        """Return number of prototypes for a given class."""
        return len(self._gallery.get(class_name, []))

    def total_prototypes(self) -> int:
        """Return total number of prototypes across all classes."""
        return sum(len(v) for v in self._gallery.values())

    def add_embedding(self, class_name: str, embedding: np.ndarray) -> None:
        """Add a prototype embedding for a traffic sign class.

        Args:
            class_name: Name/label of the traffic sign class.
            embedding: 1-D numpy array representing the embedding.
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if class_name not in self._gallery:
            self._gallery[class_name] = []
        self._gallery[class_name].append(embedding)

    def add_embeddings(
        self, class_name: str, embeddings: list[np.ndarray]
    ) -> None:
        """Add multiple prototype embeddings for a class.

        Args:
            class_name: Name/label of the traffic sign class.
            embeddings: List of embedding arrays.
        """
        for emb in embeddings:
            self.add_embedding(class_name, emb)

    def remove_class(self, class_name: str) -> bool:
        """Remove a class and all its prototypes from the gallery.

        Returns:
            True if the class existed and was removed, False otherwise.
        """
        if class_name in self._gallery:
            del self._gallery[class_name]
            return True
        return False

    def query(
        self, embedding: np.ndarray
    ) -> tuple[Optional[str], float, dict[str, float]]:
        """Find the most similar traffic sign class for a query embedding.

        Computes cosine similarity between the query and all stored prototypes.
        For each class, takes the maximum similarity across its prototypes.

        Args:
            embedding: 1-D query embedding array.

        Returns:
            Tuple of (predicted_class, similarity_score, all_scores).
            If no class exceeds the threshold, predicted_class is None.
        """
        if not self._gallery:
            return None, 0.0, {}

        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        class_scores: dict[str, float] = {}

        for class_name, prototypes in self._gallery.items():
            max_sim = -1.0
            for proto in prototypes:
                proto_norm = np.linalg.norm(proto)
                if proto_norm > 0:
                    proto_normalized = proto / proto_norm
                else:
                    proto_normalized = proto
                sim = float(np.dot(embedding, proto_normalized))
                max_sim = max(max_sim, sim)
            class_scores[class_name] = max_sim

        best_class = max(class_scores, key=class_scores.get)
        best_score = class_scores[best_class]

        if best_score < self.similarity_threshold:
            return None, best_score, class_scores

        return best_class, best_score, class_scores

    def save(self, path: str) -> None:
        """Save the gallery to disk.

        Saves embeddings as a .npz file and metadata as a .json file.

        Args:
            path: Base path (without extension) for the saved files.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        metadata = {
            "similarity_threshold": self.similarity_threshold,
            "classes": {},
        }
        all_embeddings = []
        idx = 0

        for class_name, prototypes in self._gallery.items():
            start_idx = idx
            for proto in prototypes:
                all_embeddings.append(proto)
                idx += 1
            metadata["classes"][class_name] = {
                "start_idx": start_idx,
                "count": len(prototypes),
            }

        if all_embeddings:
            np.savez(f"{path}.npz", embeddings=np.stack(all_embeddings))
        else:
            np.savez(f"{path}.npz", embeddings=np.array([]))

        with open(f"{path}.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SignGallery":
        """Load a gallery from disk.

        Args:
            path: Base path (without extension) used when saving.

        Returns:
            Loaded SignGallery instance.
        """
        with open(f"{path}.json", "r") as f:
            metadata = json.load(f)

        data = np.load(f"{path}.npz")
        all_embeddings = data["embeddings"]

        gallery = cls(similarity_threshold=metadata["similarity_threshold"])

        for class_name, info in metadata["classes"].items():
            start = info["start_idx"]
            count = info["count"]
            for i in range(count):
                gallery.add_embedding(class_name, all_embeddings[start + i])

        return gallery
