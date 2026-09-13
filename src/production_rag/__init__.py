"""Production RAG Knowledge System package."""

from production_rag.ingestion import Chunk, Document, LocalFileLoader, TextChunker

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Document",
    "LocalFileLoader",
    "TextChunker",
    "__version__",
]
