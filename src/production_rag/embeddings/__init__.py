"""Embedding generation and vector representation package."""

from production_rag.embeddings.base import EmbeddingProvider
from production_rag.embeddings.local import LocalDeterministicEmbeddingProvider
from production_rag.embeddings.models import EmbeddedChunk
from production_rag.embeddings.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
)

__all__ = [
    "EmbeddedChunk",
    "EmbeddingProvider",
    "LocalDeterministicEmbeddingProvider",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
]
