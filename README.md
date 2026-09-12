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

> **Current Status**: **Project Inception & Clean Baseline Setup**.
> No heavy RAG frameworks (LangChain, LlamaIndex, vector databases) are installed yet. The architecture is being built methodically from clean fundamentals.

---

## Directory Structure

```text
production-rag-knowledge-system/
├── .env.example               # Template for environment variables and secrets
├── .gitignore                 # Python and data artifact ignore rules
├── pyproject.toml             # Project metadata, build settings, and dependencies
├── requirements.txt           # Minimal pinned base dependencies
├── README.md                  # Project overview and roadmap (this file)
├── data/                      # Knowledge assets and document store
│   ├── raw/                   # Unprocessed source documents (PDFs, Markdown, text, etc.)
│   ├── processed/             # Parsed chunks, indices, and local caches
│   └── README.md              # Documentation for data lifecycle and storage
├── docs/                      # Architectural specs, RFCs, and learning notes
│   └── architecture.md        # Comprehensive pipeline architectural overview
├── src/                       # Core application source code
│   └── production_rag/        # Primary Python package
│       ├── __init__.py        # Package exports and version
│       └── config.py          # Environment and application settings
└── tests/                     # Test suites (unit, integration, and evaluation)
    ├── __init__.py
    └── test_smoke.py          # Smoke test verifying baseline packaging
```

---

## Setup & Getting Started

### 1. Prerequisites
- Python >= 3.11
- `uv` (recommended) or standard `venv`

### 2. Environment Setup
```bash
# Navigate to the project directory
cd production-rag-knowledge-system

# Create and activate virtual environment using uv or python
uv venv
source .venv/bin/activate

# Or with python standard venv:
# python3 -m venv .venv
# source .venv/bin/activate
```

### 3. Install Baseline Dependencies
```bash
# Install minimal baseline dependencies (no heavy frameworks yet)
pip install -r requirements.txt
```

### 4. Run Smoke Tests
```bash
pytest
```

---

## Roadmap

1. [x] **Phase 0: Clean Baseline Project Setup**
2. [ ] **Phase 1: Ingestion & Smart Chunking Strategies**
3. [ ] **Phase 2: Embedding Generation & Vector Indexing**
4. [ ] **Phase 3: Hybrid Retrieval (Dense + BM25) & Re-ranking**
5. [ ] **Phase 4: Synthesis, Grounding & Citations**
6. [ ] **Phase 5: Evaluation, Guardrails & Benchmarking**
