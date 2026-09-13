"""Unit tests for document ingestion and chunking components."""

from pathlib import Path

import pytest

from production_rag.ingestion.chunker import TextChunker
from production_rag.ingestion.loader import LocalFileLoader
from production_rag.ingestion.models import Chunk, Document


@pytest.fixture
def temp_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with various document fixtures."""
    docs_dir = tmp_path / "data_raw"
    docs_dir.mkdir()

    # 1. Standard markdown document with header
    doc1 = docs_dir / "01_intro.md"
    doc1.write_text(
        "# Introduction to RAG\n\n"
        "Retrieval-Augmented Generation improves LLM response fidelity.\n"
        "It connects language models to external, verifiable knowledge repositories.\n\n"
        "This ensures that answers stay grounded, truthful, and up-to-date.",
        encoding="utf-8",
    )

    # 2. Text file without markdown header
    doc2 = docs_dir / "02_plain_notes.txt"
    doc2.write_text(
        "First line of plain notes.\n"
        "Second line with additional details on chunking and indexing.\n"
        "Third line covering system evaluation and observability.",
        encoding="utf-8",
    )

    # 3. Empty document (0 bytes)
    empty_doc = docs_dir / "empty_doc.md"
    empty_doc.write_text("", encoding="utf-8")

    # 4. Whitespace only document
    whitespace_doc = docs_dir / "whitespace_only.txt"
    whitespace_doc.write_text("   \n\t\n   ", encoding="utf-8")

    return docs_dir


# ==============================================================================
# 1. Document Loading Tests
# ==============================================================================


def test_load_document_markdown(temp_docs_dir: Path):
    """Verify loading a markdown file extracts title, metadata, and generates doc_id."""
    loader = LocalFileLoader(base_dir=temp_docs_dir)
    doc = loader.load_file("01_intro.md")

    assert isinstance(doc, Document)
    assert doc.title == "Introduction to RAG"
    assert doc.source == "01_intro.md"
    assert "Retrieval-Augmented Generation improves" in doc.content
    assert doc.document_id.startswith("doc_")
    assert doc.metadata["file_name"] == "01_intro.md"
    assert doc.metadata["extension"] == ".md"
    assert doc.metadata["is_empty"] is False
    assert doc.metadata["file_size_bytes"] > 0


def test_load_document_plain_text_fallback_title(temp_docs_dir: Path):
    """Verify plain text files without markdown header fallback to stem title."""
    loader = LocalFileLoader(base_dir=temp_docs_dir)
    doc = loader.load_file("02_plain_notes.txt")

    assert doc.title == "02 Plain Notes"
    assert doc.source == "02_plain_notes.txt"
    assert "First line of plain notes" in doc.content


def test_load_directory_deterministic_order(temp_docs_dir: Path):
    """Verify loading a directory loads all matching files in alphabetical path order."""
    loader = LocalFileLoader(base_dir=temp_docs_dir)
    docs = loader.load_directory()

    # Sorted by filename: 01_intro.md, 02_plain_notes.txt, empty_doc.md, whitespace_only.txt
    sources = [d.source for d in docs]
    assert sources == [
        "01_intro.md",
        "02_plain_notes.txt",
        "empty_doc.md",
        "whitespace_only.txt",
    ]


def test_load_nonexistent_file_raises(temp_docs_dir: Path):
    """Verify attempting to load a missing file raises FileNotFoundError."""
    loader = LocalFileLoader(base_dir=temp_docs_dir)
    with pytest.raises(FileNotFoundError):
        loader.load_file("nonexistent.md")


def test_load_unsupported_extension_raises(temp_docs_dir: Path):
    """Verify attempting to load unsupported file types raises ValueError."""
    unsupported = temp_docs_dir / "data.csv"
    unsupported.write_text("a,b,c\n1,2,3", encoding="utf-8")

    loader = LocalFileLoader(base_dir=temp_docs_dir)
    with pytest.raises(ValueError, match="Unsupported file extension"):
        loader.load_file(unsupported)


# ==============================================================================
# 2. Empty Document Handling Tests
# ==============================================================================


def test_load_empty_document_allowed(temp_docs_dir: Path):
    """Verify empty document is loaded with empty content when allow_empty=True."""
    loader = LocalFileLoader(base_dir=temp_docs_dir, allow_empty=True)
    doc = loader.load_file("empty_doc.md")

    assert doc.content == ""
    assert doc.is_empty is True
    assert doc.metadata["is_empty"] is True


def test_load_empty_document_disallowed(temp_docs_dir: Path):
    """Verify empty document raises ValueError when allow_empty=False."""
    loader = LocalFileLoader(base_dir=temp_docs_dir, allow_empty=False)
    with pytest.raises(ValueError, match="File is empty"):
        loader.load_file("empty_doc.md")


def test_chunker_handles_empty_document():
    """Verify chunker returns empty list when given an empty or whitespace document."""
    empty_doc = Document(
        document_id="doc_test_empty",
        source="empty.md",
        title="Empty Doc",
        content="   \n\n  ",
        metadata={"file_name": "empty.md"},
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(empty_doc)

    assert chunks == []


# ==============================================================================
# 3. Chunk Creation Tests
# ==============================================================================


def test_chunk_creation_single_chunk():
    """Verify short documents fit into a single chunk."""
    doc = Document(
        document_id="doc_short_01",
        source="short.md",
        title="Short Document",
        content="This is a very short text that easily fits in one chunk.",
        metadata={"author": "Ashok"},
    )
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].chunk_id == "doc_short_01#chunk_0000"
    assert chunks[0].content == doc.content
    assert chunks[0].character_count == len(doc.content)


def test_chunk_creation_multiple_chunks():
    """Verify long documents are split into multiple distinct chunks."""
    long_content = (
        "Paragraph One introduces the retrieval architecture.\n\n"
        "Paragraph Two discusses chunking trade-offs and sentence preservation.\n\n"
        "Paragraph Three details re-ranking and reciprocal rank fusion.\n\n"
        "Paragraph Four summarizes evaluation metrics including faithfulness."
    )
    doc = Document(
        document_id="doc_multi_01",
        source="guide.md",
        title="Guide",
        content=long_content,
        metadata={"category": "architecture"},
    )
    chunker = TextChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert len(chunk.content) > 0


# ==============================================================================
# 4. Chunk Ordering Tests
# ==============================================================================


def test_chunk_ordering():
    """Verify chunk indices are sequential and strictly monotonically increasing."""
    paragraphs = [
        f"Section {i}: Ingestion and indexing pipeline step {i} details."
        for i in range(10)
    ]
    doc = Document(
        document_id="doc_order_01",
        source="ordering.md",
        title="Ordering Test",
        content="\n\n".join(paragraphs),
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=15)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 4
    for expected_index, chunk in enumerate(chunks):
        assert chunk.chunk_index == expected_index
        assert chunk.chunk_id == f"doc_order_01#chunk_{expected_index:04d}"

    # Verify start_char position increases monotonically
    start_positions = [c.metadata["start_char"] for c in chunks]
    assert start_positions == sorted(start_positions)
    assert len(set(start_positions)) == len(start_positions)


# ==============================================================================
# 5. Metadata Preservation Tests
# ==============================================================================


def test_metadata_preservation():
    """Verify parent document_id, source, title, and metadata are preserved in each chunk."""
    custom_metadata = {
        "repo": "production-rag-knowledge-system",
        "domain": "ai",
        "priority": 1,
    }
    doc = Document(
        document_id="doc_meta_1234",
        source="specs/spec.md",
        title="Technical Specification",
        content="First paragraph describing the system components.\n\nSecond paragraph explaining chunking.",
        metadata=custom_metadata,
    )
    chunker = TextChunker(chunk_size=70, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.document_id == "doc_meta_1234"
        assert chunk.source == "specs/spec.md"
        assert chunk.title == "Technical Specification"
        # Preserved custom metadata
        assert chunk.metadata["repo"] == "production-rag-knowledge-system"
        assert chunk.metadata["domain"] == "ai"
        assert chunk.metadata["priority"] == 1
        # Chunk-level metadata extensions
        assert "start_char" in chunk.metadata
        assert "end_char" in chunk.metadata
        assert "character_count" in chunk.metadata
        assert chunk.metadata["character_count"] == len(chunk.content)


# ==============================================================================
# 6. Deterministic Output Tests
# ==============================================================================


def test_deterministic_document_id():
    """Verify loading the identical file content generates the exact same document_id."""
    content = "# Consistent Doc\nSame content everywhere."
    id1 = LocalFileLoader._generate_document_id("docs/same.md", content)
    id2 = LocalFileLoader._generate_document_id("docs/same.md", content)
    assert id1 == id2
    assert id1.startswith("doc_")


def test_deterministic_chunking_output():
    """Verify chunking the exact same document multiple times yields identical results."""
    doc = Document(
        document_id="doc_determ_99",
        source="knowledge/rag.md",
        title="RAG Determinism",
        content=(
            "Chunking must be completely deterministic across executions.\n\n"
            "If document chunking produced varying outputs, vector IDs would shift,\n"
            "invalidating persistent vector database indices and cache keys."
        ),
        metadata={"version": "1.0"},
    )
    chunker = TextChunker(chunk_size=90, chunk_overlap=15)

    run_1 = chunker.chunk_document(doc)
    run_2 = chunker.chunk_document(doc)

    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.chunk_index == c2.chunk_index
        assert c1.content == c2.content
        assert c1.metadata == c2.metadata


def test_invalid_chunker_parameters():
    """Verify invalid chunker sizing raises clear validation exceptions."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        TextChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="strictly smaller than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)
