"""Deterministic text chunker for document ingestion."""

from production_rag.ingestion.models import Chunk, Document


class TextChunker:
    """Splits Documents into deterministic structured Chunks using a sliding window.

    This chunker snaps both start and end boundaries to natural separators (paragraphs,
    newlines, sentence ends, or word boundaries) to avoid breaking semantic tokens or
    slicing words mid-word, while maintaining predictable character-based size and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Maximum character length for each chunk.
            chunk_overlap: Number of characters shared between consecutive chunks.
            separators: Prioritized list of delimiters to snap chunk boundaries to.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a Document into a list of ordered, structured Chunk objects.

        Args:
            document: Ingested Document to split.

        Returns:
            List of structured Chunk instances in sequential order.
        """
        text = document.content
        if not text or not text.strip():
            return []

        total_length = len(text)
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < total_length:
            target_end = min(start + self.chunk_size, total_length)

            if target_end < total_length:
                split_end = self._find_split_point(text, start, target_end)
            else:
                split_end = total_length

            chunk_raw = text[start:split_end]
            chunk_text = chunk_raw.strip()

            if chunk_text:
                chunk_id = f"{document.document_id}#chunk_{chunk_index:04d}"
                chunk_metadata = {
                    **document.metadata,
                    "start_char": start,
                    "end_char": split_end,
                    "character_count": len(chunk_text),
                }

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        content=chunk_text,
                        source=document.source,
                        title=document.title,
                        metadata=chunk_metadata,
                    )
                )
                chunk_index += 1

            if split_end >= total_length:
                break

            # Calculate next window start with overlap
            next_start = split_end - self.chunk_overlap

            # Snap next_start cleanly to a word boundary to prevent sliced words at chunk start
            if next_start > start and next_start < total_length:
                prev_space = text.rfind(
                    " ", max(start, next_start - 20), next_start + 1
                )
                if prev_space != -1 and prev_space > start:
                    next_start = prev_space + 1

            # Guarantee forward progress
            start = max(start + 1, next_start)

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a sequence of documents in order."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _find_split_point(self, text: str, start: int, target_end: int) -> int:
        """Find the most natural separator boundary before target_end."""
        # Search back up to halfway through the chunk window to prevent tiny degenerate chunks
        search_floor = start + max(1, self.chunk_size // 2)

        for sep in self.separators:
            last_idx = text.rfind(sep, search_floor, target_end)
            if last_idx != -1:
                return last_idx + len(sep)

        return target_end
