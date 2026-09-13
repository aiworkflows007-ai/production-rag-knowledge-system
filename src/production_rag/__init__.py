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
from production_rag.retrieval import BaseRetriever, RetrievalResult, VectorRetriever

__version__ = "0.1.0"

__all__ = [
    "BaseRetriever",
    "BaseVectorStore",
    "Chunk",
    "Document",
    "EmbeddedChunk",
    "EmbeddingProvider",
    "InMemoryVectorStore",
    "IndexingPipeline",
    "LocalDeterministicEmbeddingProvider",
    "LocalFileLoader",
    "RetrievalResult",
    "TextChunker",
    "VectorRetriever",
    "__version__",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
]
