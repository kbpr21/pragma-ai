import logging
from pathlib import Path
from typing import List

from pragma.ingestion.loader import DocumentSegment

logger = logging.getLogger(__name__)


def load_pdf_file(path: Path) -> List[DocumentSegment]:
    """Load PDF file with page-by-page extraction."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed. Install: pip install pdfplumber")
        return []

    segments = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""

                tables = page.tables
                if tables:
                    for table in tables:
                        if table:
                            table_text = _format_table(table)
                            text += "\n" + table_text

                if text.strip():
                    segments.append(
                        DocumentSegment(
                            content=text,
                            source=str(path),
                            doc_type="pdf",
                            metadata={
                                "filename": path.name,
                                "page": page_num,
                                "char_count": len(text),
                            },
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                continue

    if not segments:
        logger.warning(f"No text extracted from PDF: {path}")

    return segments


def _format_table(table: List[List[str]]) -> str:
    """Format table as pipe-delimited text."""
    if not table:
        return ""

    lines = []
    for row in table:
        formatted_row = " | ".join(str(cell) if cell else "" for cell in row)
        lines.append(formatted_row)

    return "\n".join(lines)
