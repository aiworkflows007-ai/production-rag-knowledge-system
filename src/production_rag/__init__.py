"""Production RAG Knowledge System package."""

from production_rag.embeddings import (
    EmbeddedChunk,
    EmbeddingProvider,
    LocalDeterministicEmbeddingProvider,
    cosine_similarity,
    dot_product,
    euclidean_distance,
)
from production_rag.indexing import BaseVectorStore, InMemoryVectorStore
from production_rag.ingestion import Chunk, Document, LocalFileLoader, TextChunker
from production_rag.pipeline import IndexingPipeline

__version__ = "0.1.0"

__all__ = [
    "BaseVectorStore",
    "Chunk",
    "Document",
    "EmbeddedChunk",
    "EmbeddingProvider",
    "InMemoryVectorStore",
    "IndexingPipeline",
    "LocalDeterministicEmbeddingProvider",
    "LocalFileLoader",
    "TextChunker",
    "__version__",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
]
