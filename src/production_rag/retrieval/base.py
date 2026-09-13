"""Abstract base class for retrieval components."""

from abc import ABC, abstractmethod

from production_rag.retrieval.models import RetrievalResult


class BaseRetriever(ABC):
    """Abstract interface defining the contract for query-time retrieval.

    Decouples retrieval orchestration from specific index implementations,
    embedding providers, or scoring algorithms.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-K relevant chunks matching the user query.

        Args:
            query: Plain text user search query.
            top_k: Maximum number of relevant chunks to return (must be > 0).
            similarity_threshold: Optional cutoff threshold; results with similarity
                below this value will be excluded.

        Returns:
            List of RetrievalResult objects ordered from highest to lowest similarity.

        Raises:
            ValueError: If query is empty/whitespace or top_k <= 0.
            TypeError: If query is not a string.
        """
