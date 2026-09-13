# Production RAG Knowledge System

A systematic, production-grade Retrieval-Augmented Generation (RAG) learning project built to understand, architect, and implement high-performance knowledge retrieval systems.

---

## Project Vision & Scope

Most RAG tutorials demonstrate toy examples (simple chunking + naive vector search + single prompt). In contrast, this project is dedicated to engineering a **production-ready RAG pipeline** handling real-world challenges:

- **Data Ingestion & Preprocessing**: Clean parsing, document structure awareness, dynamic and semantic chunking.
- **Embedding Generation & Vector Storage**: Decoupled embedding providers, typed embedded chunks, and extensible vector storage.
- **Hybrid Retrieval**: Combining dense vector embeddings (semantic similarity) with sparse retrieval (BM25 lexical search) via Reciprocal Rank Fusion (RRF).
- **Post-Retrieval Processing & Re-ranking**: Cross-encoder rerankers, contextual compression, and relevance filtering to combat the "lost in the middle" phenomenon.
- **Grounded Generation**: Prompt engineering with strict context bounding, source attribution, and citation enforcement.
- **Evaluation & Observability**: Quantitative evaluation (faithfulness, answer relevance, context recall/precision) and token/latency tracing.

> **Current Milestone**: **Ingestion, Chunking, Embedding Generation, & Vector Storage/Index**.
> Architecture strictly decouples embedding models, data models, and vector stores without relying on bulky all-in-one frameworks.

---

## The Indexing Pipeline

```text
Raw Document
     ↓
 Ingestion            (LocalFileLoader reads UTF-8, extracts title, assigns deterministic document_id)
     ↓
  Chunking            (TextChunker splits text into structured Chunks with boundary snapping)
     ↓
 Embedding            (EmbeddingProvider transforms chunk text into dense numerical vectors)
     ↓
Vector Store / Index  (BaseVectorStore indexes EmbeddedChunks for fast retrieval & metadata lookup)
```

Detailed architectural visualization:

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
└──────────────────────┬───────────────────────┘
                       │ Document (Pydantic Model)
                       ▼
┌──────────────────────────────────────────────┐
│  TextChunker                                 │
│  - Sliding window (chunk_size, chunk_overlap)│
│  - Delimiter boundary snapping (no cut words)│
│  - Monotonic chunk_index (0, 1, 2, ...)      │
└──────────────────────┬───────────────────────┘
                       │ List[Chunk] (Pydantic Models)
                       ▼
┌──────────────────────────────────────────────┐
│  EmbeddingProvider                           │
│  - Decoupled interface (base.py)             │
│  - LocalDeterministicEmbeddingProvider       │
│  - Signed subword feature projection (D=128) │
│  - Unit L2-normalized dense vectors          │
└──────────────────────┬───────────────────────┘
                       │ List[EmbeddedChunk]
                       ▼
┌──────────────────────────────────────────────┐
│  VectorStore / Index (InMemoryVectorStore)   │
│  - O(1) ID lookups & dimension validation    │
│  - Preserves original chunk text & metadata  │
│  - Foundation for future similarity search   │
└──────────────────────────────────────────────┘
```

---

## Core Concepts: Embeddings & Vector Indexing

### 1. WHAT is an embedding?
An **embedding** is a dense numerical vector (a list of floating-point numbers, such as 128, 384, or 1536 dimensions) representing the syntactic and semantic essence of a text passage. Texts with similar topical meanings or vocabulary overlap map to coordinates that are mathematically close to one another in geometric vector space.

### 2. WHY do we create embeddings?
Computers cannot directly calculate semantic closeness using raw character strings. Traditional keyword matching fails when synonyms or rephrased queries are used (e.g., `"automobile repair"` vs `"car maintenance"`). Embeddings convert human language into geometric points, enabling algorithms to compute similarity using mathematical distance metrics.

### 3. WHAT does the embedding model do?
The **embedding model** (represented by `EmbeddingProvider`) is a pure transformation function:
$$\text{text} \longrightarrow \text{vector} \in \mathbb{R}^D$$
It accepts raw text, tokenizes it, and outputs a normalized fixed-dimensional vector. It does not store vectors or know about databases; its sole responsibility is vector generation.

### 4. WHAT does the vector index do?
The **vector index / vector store** (`BaseVectorStore`) is a specialized data structure and storage engine that stores the embeddings together with their associated chunk metadata and raw text. In later retrieval stages, it organizes these high-dimensional points so that nearest-neighbor queries can be executed in sub-linear time.

### 5. WHY do we keep the original chunk text?
The embedding is purely a mathematical coordinate for spatial indexing; **it is NOT a replacement for the text**.
- An embedding vector cannot be read by human users or passed directly into an LLM context window.
- When relevant chunks are retrieved at query time, the system passes the **original human-readable text** into the LLM's prompt context to synthesize a truthful, grounded response with exact citations.
- Therefore, `EmbeddedChunk` strictly preserves the original `text`, parent `document_id`, `chunk_id`, and `metadata`.

### 6. IMPORTANT: Similarity Scores Are Computed at Query Time
> **Critical Architectural Principle**:
> A similarity score is **never permanently assigned** to a chunk during indexing.
> - At **Indexing Time**: Chunks are embedded and their vectors are stored. There is no user query yet, so no similarity score exists.
> - At **Query Time**: When a user asks a question, the *query itself* is converted into an embedding vector by the embedding model. That query vector is compared against all stored chunk vectors using a similarity metric (e.g., Cosine Similarity) to calculate a dynamic, query-specific relevance score.

---

## Similarity Metrics Prepared

The system prepares for future similarity search by producing L2-normalized unit vectors ($\|v\|_2 = 1.0$) and providing standard distance functions:

1. **Cosine Similarity**:
   $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   Measures the angle between two vectors, invariant to scale, returning a value between $-1.0$ and $+1.0$.
2. **Dot Product (Inner Product)**:
   $$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^D u_i v_i$$
   For unit-normalized vectors ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$), dot product is mathematically equivalent to cosine similarity but computationally faster (requires no division).
3. **Euclidean Distance (L2)**:
   $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}$$
   Measures straight-line geometric distance. For unit vectors, Euclidean distance is monotonically related to cosine distance: $d^2 = 2 - 2\cos(\theta)$.

---

## Example: Ingestion → Chunking → Embedding → Vector Storage

```python
from pathlib import Path
from production_rag.embeddings import LocalDeterministicEmbeddingProvider, cosine_similarity
from production_rag.indexing import InMemoryVectorStore
from production_rag.ingestion import LocalFileLoader, TextChunker
from production_rag.pipeline import IndexingPipeline

# 1. Initialize components
loader = LocalFileLoader(base_dir=Path("data/raw"))
chunker = TextChunker(chunk_size=350, chunk_overlap=50)
embedder = LocalDeterministicEmbeddingProvider(dimension=128)
vector_store = InMemoryVectorStore()

# 2. Build and run the indexing pipeline
pipeline = IndexingPipeline(
    loader=loader,
    chunker=chunker,
    embedder=embedder,
    vector_store=vector_store,
)

embedded_chunks = pipeline.index_file("rag_principles.md")
print(f"Stored {vector_store.count()} embedded chunks in vector store.")

# 3. Retrieve a chunk and inspect vector representation
first_chunk_id = embedded_chunks[0].chunk_id
record = vector_store.get(first_chunk_id)

print(f"\nChunk ID:      {record.chunk_id}")
print(f"Dimension:     {record.dimension}")
print(f"Vector sample: {record.embedding[:5]}... (length={len(record.embedding)})")
print(f"Original text: {record.text[:100]}...\n")

# 4. Demonstrate query-time similarity foundation
query_text = "Why does ingestion quality matter in RAG?"
query_vector = embedder.embed_text(query_text)

for chunk in vector_store.all_chunks()[:3]:
    score = cosine_similarity(query_vector, chunk.embedding)
    print(f"Chunk [{chunk.chunk_id}] Similarity to query: {score:.4f}")
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
│       └── indexing/          # Vector storage & index components
│           ├── __init__.py    # Indexing exports
│           ├── base.py        # BaseVectorStore abstract interface
│           └── memory.py      # InMemoryVectorStore implementation
└── tests/                     # Comprehensive test suites
    ├── __init__.py
    ├── test_smoke.py                  # Smoke tests verifying package metadata
    ├── test_ingestion.py              # Tests for loading, chunking, and metadata
    ├── test_embeddings_and_storage.py # Tests for embedding, storage, and similarity
    └── test_pipeline.py               # Tests for end-to-end indexing pipeline
```

---

## Running the Tests

To run the complete test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run pytest across all test suites
pytest -v

# Run embedding and storage tests specifically
pytest tests/test_embeddings_and_storage.py -v

# Run end-to-end indexing pipeline tests
pytest tests/test_pipeline.py -v
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
4. [ ] **Phase 3: Hybrid Retrieval (Dense + BM25) & Re-ranking**
5. [ ] **Phase 4: Synthesis, Grounding & Citations**
6. [ ] **Phase 5: Evaluation, Guardrails & Benchmarking**
