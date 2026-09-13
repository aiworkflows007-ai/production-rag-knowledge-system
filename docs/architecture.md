# Production RAG Knowledge System Architecture

## Overview
This document outlines the architectural specification and roadmap for transitioning from a naive RAG setup to a robust, scalable, observable, and production-grade **Retrieval-Augmented Generation (RAG)** pipeline.

---

## Architectural Separation: Indexing Time vs. Query Time

A fundamental architectural principle of production RAG systems is the strict lifecycle separation between **Indexing Time** (offline/asynchronous knowledge preparation) and **Query Time** (online/real-time question answering).

```
========================================================================================
INDEXING TIME (Offline / Ingestion Pipeline — CURRENT MILESTONE)
========================================================================================

  Raw Document (UTF-8 Markdown / Text)
        │
        ▼
  LocalFileLoader (provenance metadata, title, deterministic doc_id)
        │
        ▼  [Document]
  TextChunker (sliding window, boundary snapping, chunk_index)
        │
        ▼  [List of Chunks]
  EmbeddingProvider (signed subword projection, L2-normalization, D=128)
        │
        ▼  [List of EmbeddedChunks: text + vector + metadata]
  VectorStore / Index (InMemoryVectorStore: O(1) lookup, dimension validation)


========================================================================================
QUERY TIME (Online / Retrieval & Generation — FUTURE MILESTONES)
========================================================================================

  User Query: "Why does ingestion quality matter in RAG?"
        │
        ▼
  Query Processor (intent routing, hypothetical document expansion / HyDE)
        │
        ▼
  EmbeddingProvider.embed_text(query)
        │
        ▼  [Query Vector: 1 x D]
  Vector Index / Database (Approximate / Exact Nearest Neighbor Search)
        │
        ▼  [Top-K Candidate Chunks]
  Re-ranking Layer (Cross-encoder scoring, context compression, deduplication)
        │
        ▼  [Top-N Ranked Chunks with Source Metadata]
  Prompt Assembly & Generation Layer (Grounded prompt context bounding)
        │
        ▼
  LLM Generation & Evaluation (Response synthesis + citation attribution + faithfulness check)
```

> **Note on Current Scope**:
> In this milestone, only the **Indexing Time** pipeline is implemented:
> `Document` $\rightarrow$ `Chunk` $\rightarrow$ `Embedding` $\rightarrow$ `Vector Store/Index`.
> The query-time retrieval, re-ranking, and LLM generation layers are explicitly deferred to future milestones.

---

## System Components & Layer Breakdown

### 1. Ingestion Layer (`src/production_rag/ingestion/`)
- **`models.py`**:
  - `Document`: Immutable Pydantic model capturing `document_id`, `source`, `title`, `content`, and `metadata`.
  - `Chunk`: Immutable model representing an atomic chunk slice with `chunk_id`, `document_id`, `chunk_index`, `content`, and positional offsets.
- **`loader.py`**:
  - `LocalFileLoader`: Enforces UTF-8 decoding, parses Markdown `#` headers, falls back to humanized stems, and generates deterministic SHA-256 document IDs.
- **`chunker.py`**:
  - `TextChunker`: Sliding window with configurable `chunk_size` and `chunk_overlap`. Snaps boundaries to paragraph breaks (`\n\n`), newlines (`\n`), sentences (`. `), and spaces (` `) to prevent broken tokens or mid-word cuts.

### 2. Embedding Layer (`src/production_rag/embeddings/`)
- **`base.py`**:
  - `EmbeddingProvider` (ABC): Defines the contract for embedding models: `dimension`, `embed_text()`, and `embed_texts()`. Keeps the RAG pipeline decoupled from specific model providers.
- **`models.py`**:
  - `EmbeddedChunk`: Extends the chunk representation to include the dense vector (`embedding: list[float]`). **Strictly preserves the original raw text and metadata**—the embedding is a numerical search coordinate, not a replacement for text payload.
- **`local.py`**:
  - `LocalDeterministicEmbeddingProvider`: Zero-external-dependency, offline embedding provider utilizing signed subword feature projections and L2-normalization.
- **`similarity.py`**:
  - Standalone mathematical functions (`cosine_similarity`, `dot_product`, `euclidean_distance`) preparing the foundation for future similarity search without coupling to a full retrieval engine.

### 3. Vector Storage / Index Layer (`src/production_rag/indexing/`)
- **`base.py`**:
  - `BaseVectorStore` (ABC): Storage abstraction providing `add()`, `add_many()`, `get()`, `count()`, and `contains()`. Allows swapping in-memory storage for persistent backends (Qdrant, pgvector) without downstream code modification.
- **`memory.py`**:
  - `InMemoryVectorStore`: Fast in-memory hash store with strict dimensionality checks, preserving chunk records, raw text, and vector coordinates.

### 4. Indexing Pipeline Orchestration (`src/production_rag/pipeline.py`)
- **`IndexingPipeline`**:
  - Coordinates `LocalFileLoader` $\rightarrow$ `TextChunker` $\rightarrow$ `EmbeddingProvider` $\rightarrow$ `BaseVectorStore`.
  - Provides `index_file()` and `index_directory()` entry points.

---

## Directory Structure Overview

```text
src/production_rag/
├── __init__.py                # Package-level exports
├── config.py                  # Base configuration & environment settings
├── pipeline.py                # IndexingPipeline orchestration
├── ingestion/                 # Ingestion & chunking
│   ├── __init__.py
│   ├── models.py              # Document, Chunk
│   ├── loader.py              # LocalFileLoader
│   └── chunker.py             # TextChunker
├── embeddings/                # Vector representations
│   ├── __init__.py
│   ├── base.py                # EmbeddingProvider (ABC)
│   ├── models.py              # EmbeddedChunk
│   ├── local.py               # LocalDeterministicEmbeddingProvider
│   └── similarity.py          # Vector similarity metrics
└── indexing/                  # Vector storage & index abstractions
    ├── __init__.py
    ├── base.py                # BaseVectorStore (ABC)
    └── memory.py              # InMemoryVectorStore
```

---

## Future Roadmap Evolution
- `src/production_rag/retrieval`: Hybrid search (dense semantic + sparse BM25) and reciprocal rank fusion.
- `src/production_rag/reranking`: Cross-encoder scoring models and contextual deduplication.
- `src/production_rag/generation`: Grounded prompt assembly, LLM streaming, and citation attribution.
- `src/production_rag/evaluation`: Faithfulness, answer relevance, context recall, and observability.
