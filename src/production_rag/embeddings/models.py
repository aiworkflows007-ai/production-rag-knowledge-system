"""Data models for embedded chunks."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from production_rag.ingestion.models import Chunk


class EmbeddedChunk(BaseModel):
    """Structured representation of a document chunk augmented with a dense embedding vector.

    The embedding is a numerical vector representation of the text; the original text
    and all metadata are strictly preserved.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique deterministic identifier for the chunk")
    document_id: str = Field(description="Parent document identifier")
    text: str = Field(description="Original raw chunk text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved and enriched chunk metadata",
    )
    embedding: list[float] = Field(
        description="Dense numerical vector representation of the chunk text"
    )

    # Contextual lineage attributes inherited from parent Chunk
    chunk_index: int = Field(
        default=0,
        description="0-indexed sequence position within parent document",
    )
    source: str = Field(
        default="",
        description="Origin source path or URI inherited from parent document",
    )
    title: str = Field(
        default="",
        description="Title inherited from parent document",
    )

    @property
    def content(self) -> str:
        """Alias for text ensuring backward compatibility with Chunk."""
        return self.text

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vector."""
        return len(self.embedding)

    @classmethod
    def from_chunk(cls, chunk: Chunk, embedding: list[float]) -> "EmbeddedChunk":
        """Factory constructor creating an EmbeddedChunk from an existing Chunk and its vector."""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.content,
            metadata=dict(chunk.metadata),
            embedding=list(embedding),
            chunk_index=chunk.chunk_index,
            source=chunk.source,
            title=chunk.title,
        )
