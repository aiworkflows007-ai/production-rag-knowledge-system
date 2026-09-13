# Production RAG Knowledge System

A systematic, production-grade Retrieval-Augmented Generation (RAG) learning project built to understand, architect, and implement high-performance knowledge retrieval systems.

---

## Project Vision & Scope

Most RAG tutorials demonstrate toy examples (simple chunking + naive vector search + single prompt). In contrast, this project is dedicated to engineering a **production-ready RAG pipeline** handling real-world challenges:

- **Data Ingestion & Preprocessing**: Clean parsing, document structure awareness, dynamic and semantic chunking.
- **Embedding Generation & Vector Storage**: Decoupled embedding providers, typed embedded chunks, and extensible vector storage.
- **Vector Retrieval**: Query-time vector embedding, exact cosine similarity calculation, score ranking, and Top-K chunk selection.
- **Hybrid Retrieval**: Combining dense vector embeddings (semantic similarity) with sparse retrieval (BM25 lexical search) via Reciprocal Rank Fusion (RRF).
- **Post-Retrieval Processing & Re-ranking**: Cross-encoder rerankers, contextual compression, and relevance filtering to combat the "lost in the middle" phenomenon.
- **Grounded Generation**: Prompt engineering with strict context bounding, source attribution, and citation enforcement.
- **Evaluation & Observability**: Quantitative evaluation (faithfulness, answer relevance, context recall/precision) and token/latency tracing.

> **Current Milestone**: **Dense Vector Retrieval (Top-K Similarity Search)**.
> Implemented query-time embedding, cosine similarity calculation, threshold filtering, and Top-K ranking using dependency injection without relying on third-party frameworks.

---

## Pipeline Lifecycle: Indexing Time vs. Query Time

A fundamental architectural principle of RAG systems is the strict separation between **Indexing Time** (offline knowledge storage) and **Query Time** (online question answering).

```text
========================================================================================
INDEXING TIME (Offline Ingestion & Storage)
========================================================================================
Raw Document
     ↓
 Ingestion            (LocalFileLoader reads UTF-8, extracts title, assigns deterministic document_id)
     ↓
  Chunking            (TextChunker splits text into structured Chunks with boundary snapping)
     ↓
 Embedding            (EmbeddingProvider transforms chunk text into dense numerical vectors)
     ↓
Vector Store / Index  (BaseVectorStore indexes EmbeddedChunks for fast retrieval & metadata lookup)

========================================================================================
QUERY TIME (Online Search & Retrieval)
========================================================================================
User Query
     ↓
Query Embedding       (EmbeddingProvider embeds the query into the exact same vector space)
     ↓
Similarity Search     (VectorRetriever computes cosine similarity against all stored chunk vectors)
     ↓
Threshold Filtering   (Optional similarity_threshold excludes low-confidence chunks)
     ↓
Ranking & Slicing     (Orders results descending by score and slices Top-K candidates)
     ↓
Top-K Chunks          (List[RetrievalResult] preserving raw text, metadata, and scores)
```

Detailed architectural visualization:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INDEXING TIME PIPELINE                                   │
│                                                             │
│   Raw Documents (data/raw/*.md, *.txt)                      │
│        │                                                    │
│        ▼                                                    │
│   LocalFileLoader (UTF-8, heading extraction, doc_id)       │
│        │ [Document]                                         │
│        ▼                                                    │
│   TextChunker (delimiter snapping, sliding window)          │
│        │ [List[Chunk]]                                      │
│        ▼                                                    │
│   EmbeddingProvider (dense normalized vectors)              │
│        │ [List[EmbeddedChunk]]                              │
│        ▼                                                    │
│   InMemoryVectorStore (exact vector storage + records)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2. QUERY TIME RETRIEVAL PIPELINE (CURRENT MILESTONE)        │
│                                                             │
│   User Query: "Why does ingestion quality matter in RAG?"   │
│        │                                                    │
│        ▼                                                    │
│   VectorRetriever.retrieve(query, top_k=3)                  │
│        │                                                    │
│        ▼                                                    │
│   EmbeddingProvider.embed_text(query) ──> Query Vector [D]  │
│        │                                                    │
│        ▼                                                    │
│   Exact Cosine Similarity Scan over Stored Chunk Vectors    │
│   cos(theta) = (q . c) / (|q| * |c|)                        │
│        │                                                    │
│        ▼                                                    │
│   Optional Threshold Cutoff (score >= threshold)            │
│        │                                                    │
│        ▼                                                    │
│   Descending Score Sorting + Top-K Slicing                  │
│        │                                                    │
│        ▼                                                    │
│   List[RetrievalResult]                                     │
│   ├── chunk: EmbeddedChunk (original text, metadata)        │
│   ├── score: float (similarity score for this query)        │
│   └── metric: "cosine_similarity"                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Important Architectural & Learning Principles

### 1. Learning Embedding Provider vs. Pretrained Semantic Models
> [!IMPORTANT]
> The current [`LocalDeterministicEmbeddingProvider`](file:///home/ashok/Projects/production-rag-knowledge-system/src/production_rag/embeddings/local.py) is a **deterministic hash-based learning/development implementation** (using signed subword feature projections and $L_2$-normalization).
> - It is **not** a pretrained neural semantic embedding model (like `all-MiniLM-L6-v2` or `text-embedding-3-small`).
> - It is designed to run 100% offline, deterministically, with zero external dependencies and zero API costs.
> - Because [`VectorRetriever`](file:///home/ashok/Projects/production-rag-knowledge-system/src/production_rag/retrieval/vector.py) relies strictly on the abstract [`EmbeddingProvider`](file:///home/ashok/Projects/production-rag-knowledge-system/src/production_rag/embeddings/base.py) interface via dependency injection, swapping to a neural model in future milestones requires changing only the injected provider class.

### 2. Exact Linear Comparison vs. ANN Index
> [!NOTE]
> The current [`InMemoryVectorStore`](file:///home/ashok/Projects/production-rag-knowledge-system/src/production_rag/indexing/memory.py) performs an **exact brute-force comparison ($O(N)$ linear scan)** over stored vectors.
> - Exact search guarantees 100% recall and mathematical precision for small-to-medium corpora.
> - It is **not yet an Approximate Nearest Neighbor (ANN)** index (such as HNSW, IVF, or ScaNN). ANN algorithms trade a tiny fraction of recall for sub-linear ($O(\log N)$) search speeds on million-scale vector collections.

### 3. Similarity Scores Are Dynamic Query-Time Values
> [!TIP]
> A chunk **never has a static similarity score**.
> - At **Indexing Time**: Chunks store their fixed embedding vector coordinate.
> - At **Query Time**: When a query arrives, the query vector is compared against chunk vectors. The resulting similarity score reflects relevance *specifically to that query*.
> - Scores are ephemeral and encapsulated inside [`RetrievalResult`](file:///home/ashok/Projects/production-rag-knowledge-system/src/production_rag/retrieval/models.py).

### 4. Similarity Thresholds Are Corpus- and Model-Dependent
- The retriever supports an optional `similarity_threshold: float | None = None`.
- We deliberately do **not** set an arbitrary default threshold (e.g. `0.7`). In production, similarity distributions vary radically based on the embedding model's dimensionality, pre-training objectives, normalization, and domain vocabulary.

---

## Example: Indexing and Top-K Vector Retrieval

```python
from pathlib import Path
from production_rag.embeddings import LocalDeterministicEmbeddingProvider
from production_rag.indexing import InMemoryVectorStore
from production_rag.ingestion import LocalFileLoader, TextChunker
from production_rag.pipeline import IndexingPipeline
from production_rag.retrieval import VectorRetriever

# 1. Initialize core components
loader = LocalFileLoader(base_dir=Path("data/raw"))
chunker = TextChunker(chunk_size=350, chunk_overlap=50)
embedder = LocalDeterministicEmbeddingProvider(dimension=128)
vector_store = InMemoryVectorStore()

# 2. Run Indexing Time pipeline
indexing_pipeline = IndexingPipeline(
    loader=loader,
    chunker=chunker,
    embedder=embedder,
    vector_store=vector_store,
)
indexing_pipeline.index_file("rag_principles.md")
print(f"Indexed {vector_store.count()} chunks.\n")

# 3. Initialize Query Time retriever via Dependency Injection
retriever = VectorRetriever(
    embedder=embedder,
    vector_store=vector_store,
)

# 4. Execute Top-K retrieval with optional threshold
query = "Why does ingestion quality matter for vector embeddings?"
results = retriever.retrieve(query, top_k=2, similarity_threshold=0.2)

print(f"Retrieved {len(results)} relevant chunks for query: '{query}'\n")
for rank, res in enumerate(results, start=1):
    print(f"Rank {rank} | Score: {res.score:.4f} | Chunk ID: {res.chunk_id}")
    print(f"Source: {res.chunk.source} | Title: {res.chunk.title}")
    print(f"Text:\n{res.text}\n" + "-" * 50)
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
│       ├── pipeline.py        # IndexingPipeline connecting ingestion to vector store
│       ├── ingestion/         # Ingestion layer components
│       │   ├── __init__.py    # Ingestion exports
│       │   ├── models.py      # Pydantic Document and Chunk models
│       │   ├── loader.py      # Local UTF-8 file and directory loader
│       │   └── chunker.py     # Deterministic boundary-snapping chunker
│       ├── embeddings/        # Embedding generation components
│       │   ├── __init__.py    # Embeddings exports
│       │   ├── base.py        # EmbeddingProvider abstract base class
│       │   ├── models.py      # EmbeddedChunk data model
│       │   ├── local.py       # LocalDeterministicEmbeddingProvider
│       │   └── similarity.py  # Cosine, dot product, and Euclidean similarity
│       ├── indexing/          # Vector storage & index components
│       │   ├── __init__.py    # Indexing exports
│       │   ├── base.py        # BaseVectorStore abstract interface
│       │   └── memory.py      # InMemoryVectorStore implementation
│       └── retrieval/         # Query-time retrieval components
│           ├── __init__.py    # Retrieval exports
│           ├── base.py        # BaseRetriever abstract interface
│           ├── models.py      # RetrievalResult Pydantic model
│           └── vector.py      # VectorRetriever implementation
└── tests/                     # Comprehensive test suites
    ├── __init__.py
    ├── test_smoke.py                  # Smoke tests verifying package metadata
    ├── test_ingestion.py              # Tests for loading, chunking, and metadata
    ├── test_embeddings_and_storage.py # Tests for embedding, storage, and similarity
    ├── test_pipeline.py               # Tests for end-to-end indexing pipeline
    └── test_retrieval.py              # Tests for query embedding, ranking, and Top-K
```

---

## Running the Tests

To run the complete test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run pytest across all test suites
pytest -v

# Run retrieval tests specifically
pytest tests/test_retrieval.py -v
```

---

## Phased Roadmap

1. [x] **Phase 0: Clean Baseline Project Setup**
2. [x] **Phase 1: Ingestion & Smart Chunking Strategies**
   - [x] Structured `Document` and `Chunk` schemas with metadata
   - [x] Local text and Markdown file/directory loader
   - [x] Deterministic sliding-window chunker with boundary snapping
   - [x] Ingestion unit test suite
3. [x] **Phase 2: Embedding Generation & Vector Indexing**
   - [x] Decoupled `EmbeddingProvider` abstract interface
   - [x] Deterministic local dense embedding provider (`LocalDeterministicEmbeddingProvider`)
   - [x] Structured `EmbeddedChunk` model preserving original text, metadata, and vectors
   - [x] `BaseVectorStore` abstract interface & `InMemoryVectorStore`
   - [x] Mathematical vector similarity foundations (Cosine, Dot Product, Euclidean Distance)
   - [x] End-to-end `IndexingPipeline` connecting ingestion, chunking, embedding, and storage
4. [x] **Phase 3: Dense Vector Retrieval (Top-K Similarity Search)**
   - [x] `BaseRetriever` abstract interface
   - [x] Structured `RetrievalResult` model with query-time score encapsulation
   - [x] `VectorRetriever` with dependency injection (`EmbeddingProvider`, `BaseVectorStore`)
   - [x] Exact cosine similarity search, descending ranking, and Top-K selection
   - [x] Configurable similarity threshold filtering
   - [x] Comprehensive test suite covering validation, empty stores, ranking, and determinism
5. [ ] **Phase 4: Hybrid Retrieval (Dense + BM25) & Re-ranking**
6. [ ] **Phase 5: Synthesis, Grounding & Citations**
7. [ ] **Phase 6: Evaluation, Guardrails & Benchmarking**
