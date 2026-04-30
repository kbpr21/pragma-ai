from pathlib import Path

from pragma.ingestion.loader import DocumentSegment


def load_text_file(path: Path) -> DocumentSegment:
    """Load text or markdown file."""
    content = path.read_text(encoding="utf-8")
    return DocumentSegment(
        content=content,
        source=str(path),
        doc_type="txt",
        metadata={
            "filename": path.name,
            "char_count": len(content),
        },
    )


def load_md_file(path: Path) -> DocumentSegment:
    """Load markdown file (alias for text)."""
    return load_text_file(path)
