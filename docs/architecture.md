# Production RAG Knowledge System Architecture

## Overview
This document outlines the architectural specification and roadmap for transitioning from a naive RAG setup to a robust, scalable, observable, and production-grade **Retrieval-Augmented Generation (RAG)** pipeline.

---

## Architectural Separation: Indexing Time vs. Query Time

A fundamental architectural principle of production RAG systems is the strict lifecycle separation between **Indexing Time** (offline/asynchronous knowledge preparation) and **Query Time** (online/real-time question answering).

```
========================================================================================
1. INDEXING TIME (Offline Ingestion & Storage — IMPLEMENTED)
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
2. QUERY TIME: RETRIEVAL PHASE (Online Search & Ranking — CURRENT MILESTONE)
========================================================================================

  User Query: "Why does ingestion quality matter in RAG?"
        │
        ▼
  VectorRetriever (BaseRetriever interface with dependency injection)
        │
        ▼
  EmbeddingProvider.embed_text(query)
        │
        ▼  [Query Vector: 1 x D]
  InMemoryVectorStore exact linear scan:
  cos(theta) = (q . c) / (|q| * |c|) computed against each stored chunk
        │
        ▼
  Threshold Filtering (optional: score >= similarity_threshold)
        │
        ▼
  Score Sorting (descending) & Top-K Slicing
        │
        ▼
  List[RetrievalResult] (chunk payload + query-time similarity score + metadata)


========================================================================================
3. QUERY TIME: SYNTHESIS PHASE (Generation & Evaluation — FUTURE MILESTONES)
========================================================================================

  Top-K Chunks with Source Lineage
        │
        ▼
  Re-ranking Layer (Cross-encoder scoring, context compression, deduplication) [Future]
        │
        ▼
  Prompt Assembly (Context bounding, injection protection, citation rules) [Future]
        │
        ▼
  LLM Generation & Evaluation (Faithfulness, answer relevance, source attribution) [Future]
```

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
  - `EmbeddingProvider` (ABC): Contract for embedding models: `dimension`, `embed_text()`, and `embed_texts()`. Keeps the RAG pipeline decoupled from specific model providers.
- **`models.py`**:
  - `EmbeddedChunk`: Extends the chunk representation to include the dense vector (`embedding: list[float]`). **Strictly preserves the original raw text and metadata**—the embedding is a numerical search coordinate, not a replacement for text payload.
- **`local.py`**:
  - `LocalDeterministicEmbeddingProvider`: Signed subword feature projection and $L_2$-normalization ($D=128$). Designed for offline, deterministic learning and continuous integration.
- **`similarity.py`**:
  - Mathematical distance functions (`cosine_similarity`, `dot_product`, `euclidean_distance`).

### 3. Vector Storage / Index Layer (`src/production_rag/indexing/`)
- **`base.py`**:
  - `BaseVectorStore` (ABC): Storage abstraction providing `add()`, `add_many()`, `get()`, `count()`, and `contains()`.
- **`memory.py`**:
  - `InMemoryVectorStore`: In-memory vector store with strict dimensionality checks. Performs exact brute-force linear comparisons ($O(N)$), providing the baseline for future Approximate Nearest Neighbor (ANN) indexes.

### 4. Retrieval Layer (`src/production_rag/retrieval/`)
- **`base.py`**:
  - `BaseRetriever` (ABC): Interface defining query-time retrieval: `retrieve(query, top_k, similarity_threshold) -> list[RetrievalResult]`.
- **`models.py`**:
  - `RetrievalResult`: Encapsulates an `EmbeddedChunk` and its query-time similarity `score`. Crucially, similarity scores are ephemeral and query-dependent; they are never permanently stored on the chunk.
- **`vector.py`**:
  - `VectorRetriever`: Concrete retriever using dependency injection to bind an `EmbeddingProvider` and a `BaseVectorStore`. Embeds queries, computes exact cosine similarities against stored chunks, applies optional threshold filtering, sorts descending, and returns Top-K results.

### 5. Indexing Pipeline Orchestration (`src/production_rag/pipeline.py`)
- **`IndexingPipeline`**:
  - Coordinates `LocalFileLoader` $\rightarrow$ `TextChunker` $\rightarrow$ `EmbeddingProvider` $\rightarrow$ `BaseVectorStore`.
  - Provides `index_file()` and `index_directory()` entry points.

---

## Key Technical Distinctions

### Exact Scan vs. Approximate Nearest Neighbor (ANN)
The current `InMemoryVectorStore` + `VectorRetriever` calculates the cosine similarity against **every single stored vector** (brute-force linear scan).
- **Complexity**: $O(N \cdot D)$, where $N$ is chunk count and $D$ is vector dimension.
- **Accuracy**: 100% exact mathematical recall.
- **Evolution Path**: When scaling to hundreds of thousands of documents, an Approximate Nearest Neighbor index (e.g. HNSW, IVF) will be introduced to achieve $O(\log N)$ query latencies.

### Development Feature Projection vs. Production Semantic Embeddings
`LocalDeterministicEmbeddingProvider` uses signed subword hash projections:
- **Nature**: Fast, zero-dependency, deterministic feature hashing.
- **Scope**: Captures lexical and subword overlap in continuous vector space.
- **Evolution Path**: Designed to be hot-swappable via `EmbeddingProvider` with dense pretrained neural transformer models (e.g. `bge-small-en-v1.5`, `all-MiniLM-L6-v2`, OpenAI `text-embedding-3-small`).

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
├── indexing/                  # Vector storage & index abstractions
│   ├── __init__.py
│   ├── base.py                # BaseVectorStore (ABC)
│   └── memory.py              # InMemoryVectorStore
└── retrieval/                 # Query-time retrieval components
    ├── __init__.py
    ├── base.py                # BaseRetriever (ABC)
    ├── models.py              # RetrievalResult
    └── vector.py              # VectorRetriever
```

---

## Future Roadmap Evolution
- `src/production_rag/retrieval/hybrid.py`: Hybrid search (dense vector + sparse BM25) and reciprocal rank fusion (RRF).
- `src/production_rag/reranking`: Cross-encoder scoring models and contextual deduplication.
- `src/production_rag/generation`: Grounded prompt assembly, LLM streaming, and citation attribution.
- `src/production_rag/evaluation`: Faithfulness, answer relevance, context recall, and observability.
