"""Data models for retrieval results."""

from pydantic import BaseModel, ConfigDict, Field

from production_rag.embeddings.models import EmbeddedChunk


class RetrievalResult(BaseModel):
    """Structured representation of a single retrieved chunk with its query-time similarity score.

    Note: The similarity score is computed dynamically at query time relative to a specific
    query vector. It is never permanently persisted onto the underlying chunk.
    """

    model_config = ConfigDict(frozen=True)

    chunk: EmbeddedChunk = Field(
        description="The retrieved EmbeddedChunk containing original text, vector, and metadata"
    )
    score: float = Field(
        description="Query-time similarity score (e.g. cosine similarity in range [-1.0, 1.0])"
    )
    metric: str = Field(
        default="cosine_similarity",
        description="Distance/similarity metric used to compute the score",
    )

    @property
    def similarity_score(self) -> float:
        """Convenience alias for score."""
        return self.score

    @property
    def text(self) -> str:
        """Direct access to the retrieved chunk's raw text payload."""
        return self.chunk.text

    @property
    def chunk_id(self) -> str:
        """Direct access to the retrieved chunk's identifier."""
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        """Direct access to the parent document identifier."""
        return self.chunk.document_id

    @property
    def metadata(self) -> dict:
        """Direct access to chunk metadata."""
        return self.chunk.metadata
