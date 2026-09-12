# Data Directory

This directory is designated for data sources and artifacts used by the RAG system:

- **`raw/`**: Source documents (PDFs, Markdown, text files, HTML, JSON, etc.) to be ingested.
- **`processed/`**: Chunks, metadata indices, pre-processed documents, and local vector index stores.

> **Note**: Actual data files, raw documents, and vector databases in this directory are ignored by Git (except `.gitkeep` files) to prevent leaking proprietary knowledge or bloating version control.
