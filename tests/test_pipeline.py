"""Integration tests for the complete indexing pipeline flow."""

from pathlib import Path

from production_rag.embeddings.local import LocalDeterministicEmbeddingProvider
from production_rag.indexing.memory import InMemoryVectorStore
from production_rag.ingestion.chunker import TextChunker
from production_rag.ingestion.loader import LocalFileLoader
from production_rag.pipeline import IndexingPipeline


def test_full_indexing_pipeline_with_sample_document():
    """Verify raw document -> loader -> chunker -> embedder -> vector store pipeline."""
    sample_doc_path = Path("data/raw/rag_principles.md")
    assert sample_doc_path.exists(), (
        "Sample document data/raw/rag_principles.md must exist"
    )

    loader = LocalFileLoader()
    chunker = TextChunker(chunk_size=300, chunk_overlap=40)
    embedder = LocalDeterministicEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore()

    pipeline = IndexingPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
    )

    indexed_chunks = pipeline.index_file(sample_doc_path)

    # Verify pipeline results
    assert len(indexed_chunks) > 0
    assert store.count() == len(indexed_chunks)
    assert store.dimension == 128

    # Verify each indexed chunk in the vector store
    for i, chunk in enumerate(indexed_chunks):
        assert chunk.chunk_index == i
        assert len(chunk.embedding) == 128
        assert chunk.document_id.startswith("doc_")
        assert chunk.title == "Production RAG Engineering Principles"
        assert len(chunk.text) > 0

        # Verify retrieval from store matches
        retrieved = store.get(chunk.chunk_id)
        assert retrieved is not None
        assert retrieved.chunk_id == chunk.chunk_id
        assert retrieved.text == chunk.text
        assert retrieved.embedding == chunk.embedding
