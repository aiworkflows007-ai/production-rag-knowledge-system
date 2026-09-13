"""Abstract base class for vector storage and indexing."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from production_rag.embeddings.models import EmbeddedChunk


class BaseVectorStore(ABC):
    """Abstract interface for storing, indexing, and accessing embedded chunks.

    Decouples higher-level RAG workflows from specific vector store backends
    (e.g., in-memory index, Qdrant, pgvector, Chroma, Milvus).
    """

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Return the vector dimensionality supported by this store, or None if not yet configured."""

    @abstractmethod
    def add(self, chunk: EmbeddedChunk) -> None:
        """Add a single EmbeddedChunk to the vector store.

        Args:
            chunk: EmbeddedChunk containing chunk identity, text, metadata, and vector.

        Raises:
            ValueError: If vector dimension does not match store dimension.
            TypeError: If chunk is not an EmbeddedChunk instance.
        """

    @abstractmethod
    def add_many(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Add multiple EmbeddedChunks to the vector store.

        Args:
            chunks: Sequence of EmbeddedChunk instances.
        """

    @abstractmethod
    def get(self, chunk_id: str) -> EmbeddedChunk | None:
        """Retrieve an EmbeddedChunk by its unique chunk_id.

        Args:
            chunk_id: The identifier of the chunk to look up.

        Returns:
            The stored EmbeddedChunk, or None if not found.
        """

    @abstractmethod
    def contains(self, chunk_id: str) -> bool:
        """Check whether a chunk_id exists in the store."""

    def __contains__(self, chunk_id: str) -> bool:
        return self.contains(chunk_id)

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored vectors/chunks."""

    def __len__(self) -> int:
        return self.count()

    @abstractmethod
    def all_chunks(self) -> list[EmbeddedChunk]:
        """Return all stored EmbeddedChunks."""

    @abstractmethod
    def all_chunk_ids(self) -> list[str]:
        """Return all stored chunk IDs in insertion order."""
