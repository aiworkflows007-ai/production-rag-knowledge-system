"""In-memory vector store and index implementation."""

from collections.abc import Sequence

from production_rag.embeddings.models import EmbeddedChunk
from production_rag.indexing.base import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """Lightweight, in-memory vector storage and index.

    Stores EmbeddedChunks in memory with O(1) key lookups and strict dimensionality checks.
    Provides the storage foundation for vector similarity operations without external database dependencies.
    """

    def __init__(self, dimension: int | None = None) -> None:
        """Initialize in-memory store.

        Args:
            dimension: Expected vector dimension. If None, inferred from the first added chunk.
        """
        if dimension is not None and dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self._storage: dict[str, EmbeddedChunk] = {}

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def add(self, chunk: EmbeddedChunk) -> None:
        """Add a single EmbeddedChunk to the store."""
        if not isinstance(chunk, EmbeddedChunk):
            raise TypeError(
                f"Expected EmbeddedChunk instance, got {type(chunk).__name__}"
            )

        # Enforce dimensionality consistency
        chunk_dim = chunk.dimension
        if self._dimension is None:
            self._dimension = chunk_dim
        elif chunk_dim != self._dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self._dimension}, got {chunk_dim} for chunk {chunk.chunk_id}"
            )

        self._storage[chunk.chunk_id] = chunk

    def add_many(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Add multiple EmbeddedChunks in sequence."""
        for chunk in chunks:
            self.add(chunk)

    def get(self, chunk_id: str) -> EmbeddedChunk | None:
        """Retrieve an EmbeddedChunk by chunk_id."""
        return self._storage.get(chunk_id)

    def get_vector(self, chunk_id: str) -> list[float] | None:
        """Retrieve only the embedding vector for a given chunk_id."""
        record = self._storage.get(chunk_id)
        return record.embedding if record else None

    def contains(self, chunk_id: str) -> bool:
        """Check if chunk_id is present in the store."""
        return chunk_id in self._storage

    def count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._storage)

    def all_chunks(self) -> list[EmbeddedChunk]:
        """Return a list of all stored EmbeddedChunks."""
        return list(self._storage.values())

    def all_chunk_ids(self) -> list[str]:
        """Return a list of all chunk IDs in insertion order."""
        return list(self._storage.keys())

    def clear(self) -> None:
        """Remove all records from storage."""
        self._storage.clear()
