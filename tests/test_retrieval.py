"""Unit tests for the vector retrieval layer."""

from pathlib import Path

import pytest

from production_rag.embeddings.local import LocalDeterministicEmbeddingProvider
from production_rag.embeddings.models import EmbeddedChunk
from production_rag.indexing.memory import InMemoryVectorStore
from production_rag.ingestion.chunker import TextChunker
from production_rag.ingestion.loader import LocalFileLoader
from production_rag.pipeline import IndexingPipeline
from production_rag.retrieval.models import RetrievalResult
from production_rag.retrieval.vector import VectorRetriever


@pytest.fixture
def embedder() -> LocalDeterministicEmbeddingProvider:
    """Fixture providing a deterministic embedding provider."""
    return LocalDeterministicEmbeddingProvider(dimension=128)


@pytest.fixture
def populated_vector_store(
    embedder: LocalDeterministicEmbeddingProvider,
) -> InMemoryVectorStore:
    """Fixture providing a vector store populated with distinct topic chunks."""
    store = InMemoryVectorStore(dimension=128)

    topics = [
        (
            "chk_rag_concept",
            "doc_1",
            "Retrieval-Augmented Generation enhances language models with relevant external documents.",
            "concepts.md",
            "RAG Concepts",
            {"category": "ai", "importance": "high"},
        ),
        (
            "chk_rag_chunking",
            "doc_1",
            "Chunking splits raw texts into manageable pieces while preserving semantic boundary tokens.",
            "concepts.md",
            "RAG Concepts",
            {"category": "ai", "importance": "medium"},
        ),
        (
            "chk_db_indexes",
            "doc_2",
            "Relational database B-trees and hash indexes enable fast deterministic row lookups.",
            "database.md",
            "Database Systems",
            {"category": "databases", "importance": "medium"},
        ),
        (
            "chk_culinary_recipe",
            "doc_3",
            "Traditional sourdough bread requires active starter, bread flour, water, and sea salt.",
            "recipes.md",
            "Baking Guide",
            {"category": "cooking", "importance": "low"},
        ),
    ]

    for idx, (cid, doc_id, text, source, title, meta) in enumerate(topics):
        vec = embedder.embed_text(text)
        chunk = EmbeddedChunk(
            chunk_id=cid,
            document_id=doc_id,
            text=text,
            embedding=vec,
            chunk_index=idx,
            source=source,
            title=title,
            metadata=meta,
        )
        store.add(chunk)

    return store


# ==============================================================================
# 1. Query Embedding & Basic Retrieval
# ==============================================================================


def test_query_embedding_invoked(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """1. Verify query is embedded using the injected EmbeddingProvider."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)
    query = "How does Retrieval-Augmented Generation work?"

    results = retriever.retrieve(query, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(isinstance(r.score, float) for r in results)


def test_cosine_similarity_ranking(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """2. Verify results are strictly ordered descending by cosine similarity score."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)
    query = "Retrieval-Augmented Generation language models external documents"

    results = retriever.retrieve(query, top_k=4)

    assert len(results) == 4
    # Scores must be strictly non-increasing (highest similarity first)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

    # Top result should be the direct RAG concept chunk with high similarity
    assert results[0].chunk.chunk_id == "chk_rag_concept"
    assert results[0].score > 0.80
    assert results[0].score > results[1].score


# ==============================================================================
# 2. Top-K Behavior
# ==============================================================================


def test_top_k_behavior(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """3. Verify top_k limits returned results and handles fewer chunks than top_k."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)
    query = "database index lookups"

    # Request top_k = 1
    results_1 = retriever.retrieve(query, top_k=1)
    assert len(results_1) == 1
    assert results_1[0].chunk.chunk_id == "chk_db_indexes"

    # Request top_k = 10 when only 4 chunks exist in store
    results_many = retriever.retrieve(query, top_k=10)
    assert len(results_many) == 4  # Returns all available chunks without failing


# ==============================================================================
# 3. Similarity Threshold Filtering
# ==============================================================================


def test_similarity_threshold_filtering(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """4. Verify results strictly below the similarity_threshold are excluded."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)
    query = "sourdough bread starter flour"

    # Without threshold, all 4 are returned
    all_results = retriever.retrieve(query, top_k=4)
    assert len(all_results) == 4

    top_score = all_results[0].score

    # Set threshold just below top score: only top match should pass
    threshold = top_score - 0.05
    filtered_results = retriever.retrieve(
        query, top_k=4, similarity_threshold=threshold
    )

    assert len(filtered_results) >= 1
    assert all(r.score >= threshold for r in filtered_results)

    # Set impossible threshold (1.1): should return empty list
    strict_results = retriever.retrieve(query, top_k=4, similarity_threshold=1.1)
    assert strict_results == []


# ==============================================================================
# 4. Empty Store & Edge Cases
# ==============================================================================


def test_empty_vector_store_handled_cleanly(
    embedder: LocalDeterministicEmbeddingProvider,
):
    """5. Verify empty vector store returns an empty list without error."""
    empty_store = InMemoryVectorStore(dimension=128)
    retriever = VectorRetriever(embedder=embedder, vector_store=empty_store)

    results = retriever.retrieve("test query", top_k=5)
    assert results == []


def test_invalid_top_k_raises(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """6. Verify top_k <= 0 or invalid types raise ValueError."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("valid query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("valid query", top_k=-3)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("valid query", top_k="five")  # type: ignore


def test_invalid_and_empty_queries_raise(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """7. Verify empty/whitespace or non-string query inputs raise appropriate errors."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)

    # Empty string
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("")

    # Whitespace only
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("   \n\t  ")

    # Non-string type
    with pytest.raises(TypeError, match="Query must be a string"):
        retriever.retrieve(None)  # type: ignore

    with pytest.raises(TypeError, match="Query must be a string"):
        retriever.retrieve(12345)  # type: ignore


# ==============================================================================
# 5. Determinism & Metadata Preservation
# ==============================================================================


def test_deterministic_retrieval_behavior(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """8. Verify identical query against identical store produces identical ranking and scores."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)
    query = "database index lookups"

    run_1 = retriever.retrieve(query, top_k=3)
    run_2 = retriever.retrieve(query, top_k=3)

    assert len(run_1) == len(run_2) == 3
    for r1, r2 in zip(run_1, run_2, strict=True):
        assert r1.chunk.chunk_id == r2.chunk.chunk_id
        assert r1.score == r2.score
        assert r1.chunk.text == r2.chunk.text


def test_correct_preservation_of_chunk_metadata(
    embedder: LocalDeterministicEmbeddingProvider,
    populated_vector_store: InMemoryVectorStore,
):
    """9. Verify retrieved results preserve all original chunk text, identifiers, and metadata."""
    retriever = VectorRetriever(embedder=embedder, vector_store=populated_vector_store)
    query = "database"

    results = retriever.retrieve(query, top_k=1)
    assert len(results) == 1
    result = results[0]

    # Model and property shortcuts
    assert result.chunk_id == "chk_db_indexes"
    assert result.document_id == "doc_2"
    assert "Relational database B-trees" in result.text
    assert result.chunk.source == "database.md"
    assert result.chunk.title == "Database Systems"
    assert result.chunk.chunk_index == 2
    assert result.metadata["category"] == "databases"
    assert result.metadata["importance"] == "medium"
    assert result.metric == "cosine_similarity"
    assert -1.0 <= result.score <= 1.0


# ==============================================================================
# 6. Full Pipeline Integration (Sample Document)
# ==============================================================================


def test_retriever_with_sample_document_pipeline():
    """Verify retriever retrieves relevant chunks from a real indexed document."""
    loader = LocalFileLoader()
    chunker = TextChunker(chunk_size=350, chunk_overlap=50)
    embedder = LocalDeterministicEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore()

    pipeline = IndexingPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
    )

    pipeline.index_file(Path("data/raw/rag_principles.md"))
    assert store.count() > 0

    retriever = VectorRetriever(embedder=embedder, vector_store=store)
    results = retriever.retrieve("Why Ingestion Quality Matters in RAG", top_k=2)

    assert len(results) == 2
    # Verify the retrieved chunk contains the relevant section
    combined_text = " ".join(r.text for r in results)
    assert "Why Ingestion Quality Matters" in combined_text
