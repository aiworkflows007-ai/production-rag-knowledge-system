"""File and directory loader for local text and Markdown documents."""

import hashlib
import re
from pathlib import Path

from production_rag.ingestion.models import Document

DEFAULT_SUPPORTED_EXTENSIONS: set[str] = {".md", ".markdown", ".txt"}


class LocalFileLoader:
    """Loader for reading local UTF-8 text and Markdown documents."""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        supported_extensions: set[str] | None = None,
        allow_empty: bool = True,
    ) -> None:
        """Initialize loader with base directory and configuration.

        Args:
            base_dir: Optional base directory to resolve relative paths against.
            supported_extensions: Set of allowed file extensions (e.g. {'.md', '.txt'}).
            allow_empty: Whether to allow empty documents or raise a ValueError.
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.supported_extensions = {
            ext.lower()
            for ext in (supported_extensions or DEFAULT_SUPPORTED_EXTENSIONS)
        }
        self.allow_empty = allow_empty

    def load_file(self, file_path: str | Path) -> Document:
        """Load a single text or Markdown file into a Document.

        Args:
            file_path: Path to the file (absolute or relative to base_dir).

        Returns:
            Structured Document instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file is not a supported type, cannot be decoded, or is empty when allow_empty=False.
        """
        path = self._resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported extensions: {sorted(self.supported_extensions)}"
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File at {path} is not valid UTF-8 text: {exc}") from exc

        is_empty = not bool(content and content.strip())
        if is_empty and not self.allow_empty:
            raise ValueError(f"File is empty: {path}")

        # Determine relative source path representation
        rel_source = self._compute_source_path(path)
        title = self._extract_title(content, path)
        doc_id = self._generate_document_id(rel_source, content)

        metadata = {
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "relative_path": rel_source,
            "is_empty": is_empty,
        }

        return Document(
            document_id=doc_id,
            source=rel_source,
            title=title,
            content=content,
            metadata=metadata,
        )

    def load_directory(
        self,
        directory_path: str | Path | None = None,
        recursive: bool = False,
    ) -> list[Document]:
        """Load all matching documents from a directory in deterministic order.

        Args:
            directory_path: Directory to load from. Defaults to base_dir if set.
            recursive: If True, search subdirectories recursively.

        Returns:
            List of loaded Document objects sorted deterministically by source path.
        """
        target_dir = Path(directory_path).resolve() if directory_path else self.base_dir
        if target_dir is None:
            raise ValueError("No directory path specified and no base_dir set.")

        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {target_dir}")

        pattern = "**/*" if recursive else "*"
        all_candidates = [
            p
            for p in target_dir.glob(pattern)
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in self.supported_extensions
        ]

        # Sort paths to guarantee deterministic loading sequence
        sorted_paths = sorted(
            all_candidates, key=lambda p: str(p.relative_to(target_dir))
        )

        documents: list[Document] = []
        for path in sorted_paths:
            documents.append(self.load_file(path))

        return documents

    def _resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute() and self.base_dir:
            resolved = self.base_dir / resolved
        return resolved.resolve()

    def _compute_source_path(self, path: Path) -> str:
        if self.base_dir and path.is_relative_to(self.base_dir):
            return str(path.relative_to(self.base_dir))
        return str(path)

    @staticmethod
    def _extract_title(content: str, path: Path) -> str:
        """Extract title from first Markdown header or fallback to filename stem."""
        for line in content.splitlines():
            stripped = line.strip()
            # Match top-level Markdown header '# Title'
            match = re.match(r"^#\s+(.+)$", stripped)
            if match:
                return match.group(1).strip()

        # Fallback to humanized filename stem
        stem = path.stem.replace("_", " ").replace("-", " ")
        return stem.title()

    @staticmethod
    def _generate_document_id(source: str, content: str) -> str:
        """Generate a stable deterministic document ID based on source and content."""
        hasher = hashlib.sha256()
        hasher.update(source.encode("utf-8"))
        hasher.update(b"::")
        hasher.update(content.encode("utf-8"))
        digest = hasher.hexdigest()[:16]
        return f"doc_{digest}"
