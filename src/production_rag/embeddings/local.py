"""Deterministic local dense embedding provider for development and testing."""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from production_rag.embeddings.base import EmbeddingProvider


class LocalDeterministicEmbeddingProvider(EmbeddingProvider):
    """Local, deterministic dense embedding provider based on signed subword feature hashing.

    This provider maps text tokens and character n-grams into a fixed-dimensional continuous
    vector space using signed hash projections followed by L2-normalization.

    Key Benefits for this Learning Laboratory:
    1. Zero external dependencies / network calls: Runs 100% locally and offline.
    2. Deterministic: The exact same text always produces the exact same numerical vector.
    3. Mathematical compatibility: Produces unit-norm dense vectors (|v| = 1.0) ready for
       cosine similarity, Euclidean distance, and dot product search.
    4. Fast and lightweight: Embeds thousands of text chunks in milliseconds.
    """

    def __init__(
        self, dimension: int = 128, model_name: str = "local-hash-dense-v1"
    ) -> None:
        """Initialize provider with target embedding dimensionality.

        Args:
            dimension: Dimensionality of the dense vector (default 128).
            model_name: Descriptive name for the model configuration.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a normalized dense vector."""
        valid_text = self._validate_text(text)
        vector = self._vectorize(valid_text)
        return vector.tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple text strings into normalized dense vectors efficiently."""
        if not texts:
            return []

        # Validate all texts first
        validated = [self._validate_text(t) for t in texts]

        # Stack into numpy matrix
        matrix = np.vstack([self._vectorize(t) for t in validated])
        return matrix.tolist()

    def _vectorize(self, text: str) -> np.ndarray:
        """Project text into dense vector using signed subword token hashing."""
        vec = np.zeros(self._dimension, dtype=np.float64)

        # Normalize text and extract word tokens
        cleaned = text.lower()
        words = re.findall(r"\b\w+\b", cleaned)

        # Extract multi-scale features: words + character 3-grams/4-grams
        features: list[str] = list(words)
        for word in words:
            if len(word) >= 3:
                for n in (3, 4):
                    for i in range(len(word) - n + 1):
                        features.append(word[i : i + n])

        if not features:
            # Fallback for strings without word characters
            features = [cleaned]

        for feat in features:
            h = hashlib.sha256(feat.encode("utf-8")).digest()
            # Derive bucket index from first 4 bytes
            idx = int.from_bytes(h[:4], "big") % self._dimension
            # Derive sign (+1 or -1) from 5th byte to minimize hash collision bias
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += sign

        # L2-normalization to produce unit vector (|v| = 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            # Degenerate case fallback
            vec[0] = 1.0

        return vec
