from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class VQCodebook:
    centroids: np.ndarray  # [N, d_e], float32
    faiss_index: Optional[faiss.IndexFlatL2] = None

    @property
    def num_tokens(self) -> int:
        return self.centroids.shape[0]

    @property
    def dim(self) -> int:
        return self.centroids.shape[1]

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        num_tokens: int = 1024,
        batch_size: int = 4096,
        use_faiss: bool = True,
    ) -> "VQCodebook":
        """
        Build a VQ codebook via MiniBatchKMeans over a set of embeddings.

        Args:
            embeddings: [N, d_e] float32 array.
        """
        assert embeddings.ndim == 2

        kmeans = MiniBatchKMeans(
            n_clusters=num_tokens,
            batch_size=batch_size,
            compute_labels=False,
            verbose=1,
        )
        kmeans.fit(embeddings)
        centroids = kmeans.cluster_centers_.astype(np.float32)

        faiss_index = None
        if use_faiss:
            faiss_index = faiss.IndexFlatL2(centroids.shape[1])
            faiss_index.add(centroids)

        return cls(centroids=centroids, faiss_index=faiss_index)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Map continuous embeddings to nearest centroid indices.

        Args:
            x: [B, d_e] float32 array.
        Returns:
            ids: [B] int64 array of token indices.
        """
        assert x.ndim == 2
        if self.faiss_index is not None:
            _, idx = self.faiss_index.search(x.astype(np.float32), 1)
            return idx[:, 0].astype(np.int64)

        # Fallback to brute-force
        diff = x[:, None, :] - self.centroids[None, :, :]
        dist = (diff**2).sum(axis=-1)  # [B, N]
        return dist.argmin(axis=1).astype(np.int64)

