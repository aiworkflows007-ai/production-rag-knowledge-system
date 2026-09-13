# Production RAG Knowledge System

A systematic, production-grade Retrieval-Augmented Generation (RAG) learning project built to understand, architect, and implement high-performance knowledge retrieval systems.

---

## Project Vision & Scope

Most RAG tutorials demonstrate toy examples (simple chunking + naive vector search + single prompt). In contrast, this project is dedicated to engineering a **production-ready RAG pipeline** handling real-world challenges:

- **Data Ingestion & Preprocessing**: Clean parsing, document structure awareness, dynamic and semantic chunking.
- **Hybrid Retrieval**: Combining dense vector embeddings (semantic similarity) with sparse retrieval (BM25 lexical search) via Reciprocal Rank Fusion (RRF).
- **Post-Retrieval Processing & Re-ranking**: Cross-encoder rerankers, contextual compression, and relevance filtering to combat the "lost in the middle" phenomenon.
- **Grounded Generation**: Prompt engineering with strict context bounding, source attribution, and citation enforcement.
- **Evaluation & Observability**: Quantitative evaluation (faithfulness, answer relevance, context recall/precision) and token/latency tracing.

> **Current Milestone**: **Document Ingestion & Basic Chunking**.
> No heavy RAG frameworks (LangChain, LlamaIndex, vector databases, or LLMs) are used. The ingestion layer is implemented from clean fundamentals using typed models and deterministic algorithms.

---

## What Ingestion Means in a RAG System

In a production RAG system, **ingestion** is the foundational stage responsible for converting raw, unstructured knowledge into structured, digestible, and traceable units:

1. **Source Loading**: Reading documents (Markdown, text, PDFs) from storage while enforcing valid character encoding (UTF-8) and recording provenance metadata (file paths, file size, titles).
2. **Deterministic Identity Generation**: Assigning deterministic, content- and source-derived identifiers (`document_id`) so that re-ingesting identical content does not pollute or duplicate index state.
3. **Semantic Chunking**: Splitting large texts into bounded chunks that fit embedding model context windows. Instead of blindly slicing characters mid-word or mid-sentence, production chunking respects natural structural boundaries (paragraphs, newlines, sentences, spaces) and applies sliding-window overlap so context is not sheared at chunk edges.
4. **Metadata Inheritance & Lineage**: Every emitted chunk preserves its parent `document_id`, `source`, `title`, and positional offsets (`start_char`, `end_char`), ensuring any retrieved chunk can be cited with exact provenance back to the source document.

---

## Current Ingestion Pipeline

```
┌──────────────────────────────────────────────┐
│  Raw Documents (data/raw/*.md, *.txt)        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  LocalFileLoader                             │
│  - UTF-8 decoding & validation               │
│  - Heading & title extraction                │
│  - Deterministic document_id generation      │
│  - Provenance metadata enrichment            │
└──────────────────────┬───────────────────────┘
                       │ Document (Pydantic Model)
                       ▼
┌──────────────────────────────────────────────┐
│  TextChunker                                 │
│  - Sliding window (chunk_size, chunk_overlap)│
│  - Boundary snapping (paragraphs/sentences)   │
│  - Word boundary preservation (no cut words) │
│  - Monotonic chunk_index (0, 1, 2, ...)      │
│  - Deterministic chunk_id generation         │
└──────────────────────┬───────────────────────┘
                       │ List[Chunk] (Pydantic Models)
                       ▼
┌──────────────────────────────────────────────┐
│  Structured Chunks Ready for Future Indexing │
└──────────────────────────────────────────────┘
```

---

## Example: Input Document → Structured Chunks

Using the sample document located at [`data/raw/rag_principles.md`](file:///home/ashok/Projects/production-rag-knowledge-system/data/raw/rag_principles.md):

```python
from pathlib import Path
from production_rag.ingestion import LocalFileLoader, TextChunker

# 1. Load source document
loader = LocalFileLoader(base_dir=Path("data/raw"))
doc = loader.load_file("rag_principles.md")

print(f"Document ID: {doc.document_id}")
print(f"Title:       {doc.title}")
print(f"Source:      {doc.source}")

# 2. Chunk document with sliding window
chunker = TextChunker(chunk_size=350, chunk_overlap=50)
chunks = chunker.chunk_document(doc)
print(f"Total Chunks: {len(chunks)}\n")

for chunk in chunks[:2]:
    print(f"[{chunk.chunk_id}] (index={chunk.chunk_index}, chars={chunk.character_count})")
    print(chunk.content)
    print("-" * 50)
```

**Output:**
```text
Document ID: doc_33658e4f813af600
Title:       Production RAG Engineering Principles
Source:      rag_principles.md
Total Chunks: 5

[doc_33658e4f813af600#chunk_0000] (index=0, chars=322)
# Production RAG Engineering Principles

Retrieval-Augmented Generation (RAG) is an architectural pattern that supplements large language model prompts with dynamic external context. Rather than relying solely on parametric weights learned during pre-training, RAG grounds generations in verifiable, proprietary documents.
--------------------------------------------------
[doc_33658e4f813af600#chunk_0001] (index=1, chars=335)
generations in verifiable, proprietary documents.

## Why Ingestion Quality Matters

The retrieval layer can never be better than the ingestion layer feeding it. If source documents are parsed with garbled text, broken formatting, or chopped-off sentences, downstream vector embeddings will represent noise rather than semantic intent.
--------------------------------------------------
```

---

## Directory Structure

```text
production-rag-knowledge-system/
├── .env.example               # Template for environment variables and secrets
├── .gitignore                 # Python, data artifact, and cache ignore rules
├── pyproject.toml             # Project metadata, build settings, and dependencies
├── requirements.txt           # Minimal pinned base dependencies
├── README.md                  # Project overview and pipeline documentation
├── data/                      # Knowledge assets and document store
│   ├── raw/                   # Unprocessed source documents
│   │   ├── .gitkeep
│   │   └── rag_principles.md  # Sample reference document for testing
│   ├── processed/             # Parsed chunks, indices, and local caches
│   │   └── .gitkeep
│   └── README.md              # Documentation for data lifecycle and storage
├── docs/                      # Architectural specs, RFCs, and learning notes
│   └── architecture.md        # Comprehensive pipeline architectural overview
├── src/                       # Core application source code
│   └── production_rag/        # Primary Python package
│       ├── __init__.py        # Package exports
│       ├── config.py          # Environment and application settings
│       └── ingestion/         # Ingestion layer components
│           ├── __init__.py    # Ingestion exports
│           ├── models.py      # Pydantic Document and Chunk models
│           ├── loader.py      # Local UTF-8 file and directory loader
│           └── chunker.py     # Deterministic boundary-snapping chunker
└── tests/                     # Comprehensive test suites
    ├── __init__.py
    ├── test_smoke.py          # Smoke tests verifying package metadata
    └── test_ingestion.py      # Unit tests for loading, chunking, and metadata
```

---

## Running the Tests

To run the full unit test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run pytest across all test suites
pytest -v

# Or run tests specifically for the ingestion layer
pytest tests/test_ingestion.py -v
```

---

## Phased Roadmap

1. [x] **Phase 0: Clean Baseline Project Setup**
2. [x] **Phase 1: Ingestion & Smart Chunking Strategies**
   - [x] Structured `Document` and `Chunk` schemas with metadata
   - [x] Local text and Markdown file/directory loader
   - [x] Deterministic sliding-window chunker with boundary snapping
   - [x] Unit test suite covering loading, empty files, chunking, ordering, and determinism
3. [ ] **Phase 2: Embedding Generation & Vector Indexing**
4. [ ] **Phase 3: Hybrid Retrieval (Dense + BM25) & Re-ranking**
5. [ ] **Phase 4: Synthesis, Grounding & Citations**
6. [ ] **Phase 5: Evaluation, Guardrails & Benchmarking**
