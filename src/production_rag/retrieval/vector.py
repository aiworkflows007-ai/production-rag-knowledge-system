"""Vector-based similarity retriever."""

from collections.abc import Callable, Sequence

from production_rag.embeddings.base import EmbeddingProvider
from production_rag.embeddings.similarity import cosine_similarity
from production_rag.indexing.base import BaseVectorStore
from production_rag.retrieval.base import BaseRetriever
from production_rag.retrieval.models import RetrievalResult


class VectorRetriever(BaseRetriever):
    """Retrieves relevant chunks by computing vector similarity between queries and stored chunks.

    This retriever uses dependency injection for its embedding model and vector store,
    ensuring it is completely decoupled from any single concrete embedding or storage implementation.

    Architecture Note:
    Currently, this retriever performs an exact linear scan (brute-force comparison) across
    stored chunks in the InMemoryVectorStore. This is appropriate for learning and development,
    and will later be upgraded to an Approximate Nearest Neighbor (ANN) index for large-scale corpora.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: BaseVectorStore,
        similarity_func: Callable[
            [Sequence[float], Sequence[float]], float
        ] = cosine_similarity,
    ) -> None:
        """Initialize the vector retriever.

        Args:
            embedder: EmbeddingProvider instance used to embed user queries.
            vector_store: BaseVectorStore instance holding indexed EmbeddedChunk objects.
            similarity_func: Callable calculating similarity between two vectors (defaults to cosine_similarity).
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_func = similarity_func

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and rank Top-K relevant chunks for a user query.

        Args:
            query: Plain text search query.
            top_k: Maximum number of relevant chunks to return (must be > 0).
            similarity_threshold: Optional minimum similarity score threshold. Chunks with
                similarity strictly below this value are filtered out.
                Note: Threshold values are embedding-model and corpus dependent; no universal
                default is imposed.

        Returns:
            List of RetrievalResult objects ordered from highest to lowest similarity.
        """
        self._validate_inputs(query, top_k)

        # Handle empty vector store cleanly
        if self.vector_store.count() == 0:
            return []

        # 1. Generate query embedding vector using existing EmbeddingProvider
        query_vector = self.embedder.embed_text(query)

        # 2. Compute similarity against each stored chunk (exact linear comparison)
        candidates: list[RetrievalResult] = []
        for chunk in self.vector_store.all_chunks():
            score = self.similarity_func(query_vector, chunk.embedding)

            # 3. Apply optional similarity threshold filtering
            if similarity_threshold is not None and score < similarity_threshold:
                continue

            candidates.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    metric="cosine_similarity",
                )
            )

        # 4. Rank descending by similarity score (with deterministic tie-breaking on chunk_id)
        candidates.sort(key=lambda r: (-r.score, r.chunk.chunk_id))

        # 5. Return top-K candidates (or all available if count < top_k)
        return candidates[:top_k]

    def _validate_inputs(self, query: str, top_k: int) -> None:
        """Validate search query and top_k parameter."""
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
