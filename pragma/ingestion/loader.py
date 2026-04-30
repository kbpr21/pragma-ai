from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DocumentSegment:
    """Represents a loaded document segment."""

    def __init__(
        self,
        content: str,
        source: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.content = content
        self.source = source
        self.doc_type = doc_type
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"DocumentSegment(source={self.source}, doc_type={self.doc_type})"


class DocumentLoader:
    """Load documents from various sources."""

    def __init__(self) -> None:
        self._loaders: Dict[str, Any] = {}

    def load(
        self,
        source: Union[str, Path, list, dict],
        **kwargs: Any,
    ) -> List[DocumentSegment]:
        """Load document(s) from source."""
        if isinstance(source, list):
            segments = []
            for item in source:
                segments.extend(self._load(item, **kwargs))
            return segments

        if isinstance(source, dict):
            return self._load_dict(source)

        source = str(source)

        if source.startswith("http://") or source.startswith("https://"):
            return self._load_url(source)

        path = Path(source)
        if path.suffix.lower() in (".txt", ".md"):
            return self._load_text(path)
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        if path.suffix.lower() == ".csv":
            return self._load_csv(path)
        if path.suffix.lower() in (".json", ".jsonl"):
            return self._load_json(path)
        if path.suffix.lower() == ".docx":
            return self._load_docx(path)
        if path.suffix.lower() in (".html", ".htm"):
            return self._load_html(path)

        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load(self, source: str, **kwargs: Any) -> List[DocumentSegment]:
        """Dispatch to appropriate loader."""
        path = Path(source)
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md"):
            return self._load_text(path)
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".csv":
            return self._load_csv(path)
        if suffix in (".json", ".jsonl"):
            return self._load_json(path)
        if suffix == ".docx":
            return self._load_docx(path)
        if suffix in (".html", ".htm"):
            return self._load_html(path)

        raise ValueError(f"Unsupported file type: {suffix}")

    def _load_text(self, path: Path) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.text import load_text_file

        return [load_text_file(path)]

    def _load_pdf(self, path: Path) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.pdf import load_pdf_file

        return load_pdf_file(path)

    def _load_csv(self, path: Path) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.csv import load_csv_file

        return load_csv_file(path)

    def _load_json(self, path: Path) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.json import load_json_file

        return load_json_file(path)

    def _load_docx(self, path: Path) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.docx import load_docx_file

        return load_docx_file(path)

    def _load_html(self, path: Path) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.html import load_html_file

        return load_html_file(path)

    def _load_url(self, url: str) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.html import load_html_url

        return load_html_url(url)

    def _load_dict(self, data: dict) -> List[DocumentSegment]:
        from pragma.ingestion.loaders.json import load_json_dict

        return load_json_dict(data)
