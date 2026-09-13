"""Data models for document ingestion and chunking."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Structured representation of an ingested document."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(
        description="Unique deterministic identifier for the document"
    )
    source: str = Field(description="Origin source path or URI of the document")
    title: str = Field(description="Title or descriptive label of the document")
    content: str = Field(description="Raw text content of the document")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata key-values"
    )

    @property
    def is_empty(self) -> bool:
        """Return True if content is empty or contains only whitespace."""
        return not bool(self.content and self.content.strip())

    @property
    def character_count(self) -> int:
        """Return total character length of the document content."""
        return len(self.content)


class Chunk(BaseModel):
    """Structured representation of a document chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique deterministic identifier for the chunk")
    document_id: str = Field(description="Parent document identifier")
    chunk_index: int = Field(
        description="0-indexed sequence position within parent document"
    )
    content: str = Field(description="Text segment contained in this chunk")
    source: str = Field(
        description="Origin source path or URI inherited from parent document"
    )
    title: str = Field(description="Title inherited from parent document")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata preserved and enriched for this chunk",
    )

    @property
    def character_count(self) -> int:
        """Return total character length of the chunk content."""
        return len(self.content)
