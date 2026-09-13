"""Pipeline orchestrating the indexing stage of the RAG system."""

from collections.abc import Sequence
from pathlib import Path

from production_rag.embeddings.base import EmbeddingProvider
from production_rag.embeddings.models import EmbeddedChunk
from production_rag.indexing.base import BaseVectorStore
from production_rag.ingestion.chunker import TextChunker
from production_rag.ingestion.loader import LocalFileLoader
from production_rag.ingestion.models import Document


class IndexingPipeline:
    """Orchestrates document loading, chunking, embedding, and vector storage.

    Pipeline Flow:
    Raw Document -> LocalFileLoader -> Document -> TextChunker -> Chunk ->
    EmbeddingProvider -> EmbeddedChunk -> VectorStore
    """

    def __init__(
        self,
        loader: LocalFileLoader,
        chunker: TextChunker,
        embedder: EmbeddingProvider,
        vector_store: BaseVectorStore,
    ) -> None:
        """Initialize pipeline with configured components."""
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index_file(self, file_path: str | Path) -> list[EmbeddedChunk]:
        """Process and index a single document file.

        Args:
            file_path: Path to the target document.

        Returns:
            List of indexed EmbeddedChunks stored in the vector store.
        """
        document = self.loader.load_file(file_path)
        return self._process_documents([document])

    def index_directory(
        self,
        directory_path: str | Path | None = None,
        recursive: bool = False,
    ) -> list[EmbeddedChunk]:
        """Process and index all matching documents in a directory.

        Args:
            directory_path: Directory path (defaults to loader base_dir).
            recursive: Whether to search subdirectories recursively.

        Returns:
            List of indexed EmbeddedChunks stored in the vector store.
        """
        documents = self.loader.load_directory(directory_path, recursive=recursive)
        return self._process_documents(documents)

    def _process_documents(self, documents: Sequence[Document]) -> list[EmbeddedChunk]:
        """Internal helper to chunk, embed, and store a sequence of documents."""
        all_chunks = self.chunker.chunk_documents(list(documents))
        if not all_chunks:
            return []

        embedded_chunks = self.embedder.embed_chunks(all_chunks)
        self.vector_store.add_many(embedded_chunks)
        return embedded_chunks
