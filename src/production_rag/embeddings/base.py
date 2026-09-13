"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from production_rag.embeddings.models import EmbeddedChunk
from production_rag.ingestion.models import Chunk


class EmbeddingProvider(ABC):
    """Abstract interface defining the contract for embedding providers.

    Converts textual content into dense numerical vectors while keeping the rest
    of the RAG pipeline decoupled from specific model implementations.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the fixed dimensionality of vectors produced by this provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name or identifier."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a numerical vector.

        Args:
            text: Non-empty string to embed.

        Returns:
            List of floats representing the dense embedding vector.

        Raises:
            ValueError: If text is empty or contains only whitespace.
            TypeError: If input is not a string.
        """

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of text strings into numerical vectors.

        Default implementation processes sequentially. Concrete providers can override
        with batched matrix operations for efficiency.

        Args:
            texts: Sequence of strings to embed.

        Returns:
            List of embedding vectors.
        """
        return [self.embed_text(t) for t in texts]

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Embed a single Chunk and return an EmbeddedChunk.

        Args:
            chunk: Input Chunk from the ingestion layer.

        Returns:
            EmbeddedChunk containing original chunk text, metadata, and numerical vector.
        """
        vector = self.embed_text(chunk.content)
        return EmbeddedChunk.from_chunk(chunk, vector)

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed multiple Chunk objects efficiently.

        Args:
            chunks: Sequence of Chunks to embed.

        Returns:
            List of EmbeddedChunk objects preserving order and metadata.
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        vectors = self.embed_texts(texts)

        return [
            EmbeddedChunk.from_chunk(chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def _validate_text(self, text: str) -> str:
        """Validate that input is a non-empty string."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be str, got {type(text).__name__}")
        stripped = text.strip()
        if not stripped:
            raise ValueError("Cannot embed empty or whitespace-only text")
        return stripped
