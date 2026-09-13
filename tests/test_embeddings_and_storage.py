"""Unit tests for embedding generation, vector storage, and similarity foundations."""

import math

import pytest

from production_rag.embeddings.base import EmbeddingProvider
from production_rag.embeddings.local import LocalDeterministicEmbeddingProvider
from production_rag.embeddings.models import EmbeddedChunk
from production_rag.embeddings.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
)
from production_rag.indexing.memory import InMemoryVectorStore
from production_rag.ingestion.models import Chunk


@pytest.fixture
def embedder() -> EmbeddingProvider:
    """Fixture providing a deterministic local embedding provider with dimension=128."""
    return LocalDeterministicEmbeddingProvider(dimension=128)


@pytest.fixture
def sample_chunk() -> Chunk:
    """Fixture providing a standard Chunk object."""
    return Chunk(
        chunk_id="doc_sample#chunk_0000",
        document_id="doc_sample",
        chunk_index=0,
        content="Retrieval-Augmented Generation bridges parametric memory with dynamic retrieved context.",
        source="sample.md",
        title="RAG Architecture",
        metadata={"author": "Ashok", "start_char": 0, "end_char": 89},
    )


# ==============================================================================
# 1. Chunk Embedding & Return Types
# ==============================================================================


def test_chunk_can_be_embedded(embedder: EmbeddingProvider, sample_chunk: Chunk):
    """1. A chunk can be embedded into an EmbeddedChunk."""
    embedded = embedder.embed_chunk(sample_chunk)

    assert isinstance(embedded, EmbeddedChunk)
    assert embedded.chunk_id == sample_chunk.chunk_id
    assert embedded.text == sample_chunk.content
    assert len(embedded.embedding) == embedder.dimension


def test_embedding_returns_numerical_vector(embedder: EmbeddingProvider):
    """2. Embedding returns a numerical vector of floats."""
    vector = embedder.embed_text("Production RAG pipeline engineering.")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)
    assert all(not math.isnan(v) and not math.isinf(v) for v in vector)


def test_vector_dimensionality_is_consistent(embedder: EmbeddingProvider):
    """3. Vector dimensionality is consistent across different texts."""
    texts = [
        "Short",
        "A medium length sentence discussing vector databases and cosine distance.",
        (
            "A much longer paragraph providing extensive architectural considerations "
            "for scalable vector retrieval and chunking strategies in production AI."
        ),
    ]
    for text in texts:
        vec = embedder.embed_text(text)
        assert len(vec) == embedder.dimension == 128


def test_multiple_chunks_can_be_embedded(embedder: EmbeddingProvider):
    """4. Multiple chunks can be embedded sequentially or in batch."""
    chunks = [
        Chunk(
            chunk_id=f"doc_multi#chunk_{i:04d}",
            document_id="doc_multi",
            chunk_index=i,
            content=f"Content chunk number {i} with specific technical details.",
            source="multi.md",
            title="Multi Test",
            metadata={"index": i},
        )
        for i in range(4)
    ]

    embedded_chunks = embedder.embed_chunks(chunks)

    assert len(embedded_chunks) == 4
    for i, ec in enumerate(embedded_chunks):
        assert isinstance(ec, EmbeddedChunk)
        assert ec.chunk_id == f"doc_multi#chunk_{i:04d}"
        assert ec.chunk_index == i
        assert len(ec.embedding) == embedder.dimension


# ==============================================================================
# 2. Input Validation & Edge Cases
# ==============================================================================


def test_empty_and_invalid_input_handling(embedder: EmbeddingProvider):
    """5. Empty and invalid inputs are rejected with clear exceptions."""
    # Empty string
    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_text("")

    # Whitespace only
    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_text("   \n\t  ")

    # Invalid non-string types
    with pytest.raises(TypeError, match="Expected text to be str"):
        embedder.embed_text(12345)  # type: ignore

    # Batch with empty string inside
    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_texts(["Valid text", "   ", "Another valid text"])


# ==============================================================================
# 3. Model Integrity & Lineage Preservation
# ==============================================================================


def test_embedded_chunk_preserves_all_attributes(
    embedder: EmbeddingProvider, sample_chunk: Chunk
):
    """6. EmbeddedChunk preserves chunk_id, document_id, text, metadata, and embedding."""
    embedded = embedder.embed_chunk(sample_chunk)

    # Core required fields
    assert embedded.chunk_id == "doc_sample#chunk_0000"
    assert embedded.document_id == "doc_sample"
    assert embedded.text == sample_chunk.content
    assert embedded.content == sample_chunk.content  # alias compatibility
    assert embedded.metadata["author"] == "Ashok"
    assert embedded.metadata["start_char"] == 0
    assert embedded.metadata["end_char"] == 89
    assert isinstance(embedded.embedding, list)
    assert len(embedded.embedding) == 128

    # Lineage fields
    assert embedded.chunk_index == 0
    assert embedded.source == "sample.md"
    assert embedded.title == "RAG Architecture"


# ==============================================================================
# 4. Vector Storage Tests
# ==============================================================================


def test_vector_storage_can_store_embedded_chunks(
    embedder: EmbeddingProvider, sample_chunk: Chunk
):
    """7. Vector storage can store embedded chunks."""
    store = InMemoryVectorStore(dimension=128)
    embedded = embedder.embed_chunk(sample_chunk)

    store.add(embedded)
    assert store.count() == 1
    assert sample_chunk.chunk_id in store


def test_stored_chunk_can_be_retrieved_by_id(
    embedder: EmbeddingProvider, sample_chunk: Chunk
):
    """8. Stored chunk can be retrieved by its chunk ID."""
    store = InMemoryVectorStore(dimension=128)
    embedded = embedder.embed_chunk(sample_chunk)
    store.add(embedded)

    retrieved = store.get(sample_chunk.chunk_id)
    assert retrieved is not None
    assert retrieved.chunk_id == sample_chunk.chunk_id
    assert retrieved.text == sample_chunk.content
    assert retrieved.embedding == embedded.embedding
    assert retrieved.metadata == embedded.metadata

    # Nonexistent ID returns None
    assert store.get("nonexistent_id") is None


def test_number_of_stored_vectors_is_correct(embedder: EmbeddingProvider):
    """9. Number of stored vectors is accurate."""
    store = InMemoryVectorStore()
    assert store.count() == 0
    assert len(store) == 0

    chunks = [
        EmbeddedChunk(
            chunk_id=f"chk_{i}",
            document_id="doc_test",
            text=f"Sample text chunk {i}",
            embedding=embedder.embed_text(f"Sample text chunk {i}"),
            chunk_index=i,
        )
        for i in range(5)
    ]

    store.add_many(chunks)
    assert store.count() == 5
    assert len(store) == 5
    assert store.all_chunk_ids() == [f"chk_{i}" for i in range(5)]


def test_vector_store_dimension_mismatch_raises(embedder: EmbeddingProvider):
    """Vector store enforces dimensionality consistency."""
    store = InMemoryVectorStore(dimension=128)

    invalid_chunk = EmbeddedChunk(
        chunk_id="invalid_dim_chk",
        document_id="doc_test",
        text="Text with mismatching dim",
        embedding=[0.1, 0.2, 0.3],  # dimension = 3 instead of 128
    )

    with pytest.raises(ValueError, match="Dimension mismatch"):
        store.add(invalid_chunk)


# ==============================================================================
# 5. Similarity Calculations Foundation
# ==============================================================================


def test_embeddings_used_for_similarity_calculations(embedder: EmbeddingProvider):
    """10. Embeddings can be used for cosine similarity, dot product, and distance calculations."""
    text_a = "Retrieval-Augmented Generation combines search with language models."
    text_b = (
        "RAG architectures integrate document search and language model generation."
    )
    text_c = "The Italian restaurant prepares traditional wood-fired pizza and pasta."

    vec_a = embedder.embed_text(text_a)
    vec_b = embedder.embed_text(text_b)
    vec_c = embedder.embed_text(text_c)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    # Cosine similarity must be a bounded float in [-1.0, 1.0]
    assert -1.0 <= sim_ab <= 1.0
    assert -1.0 <= sim_ac <= 1.0

    # Self-similarity of unit vector should be ~1.0
    sim_self = cosine_similarity(vec_a, vec_a)
    assert pytest.approx(sim_self, abs=1e-5) == 1.0

    # Semantically/lexically related texts should have higher similarity than unrelated text
    assert sim_ab > sim_ac

    # Dot product should equal cosine similarity for normalized vectors
    dot_ab = dot_product(vec_a, vec_b)
    assert pytest.approx(dot_ab, abs=1e-5) == sim_ab

    # Euclidean distance should be smaller for related texts
    dist_ab = euclidean_distance(vec_a, vec_b)
    dist_ac = euclidean_distance(vec_a, vec_c)
    assert dist_ab < dist_ac
