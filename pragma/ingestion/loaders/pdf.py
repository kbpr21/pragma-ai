import logging
from pathlib import Path
from typing import List

from pragma.ingestion.loader import DocumentSegment

logger = logging.getLogger(__name__)


def load_pdf_file(path: Path) -> List[DocumentSegment]:
    """Load PDF file with page-by-page extraction.

    Uses ``pdfplumber`` as the primary extractor. When pdfplumber
    yields no text for a page (common with LaTeX PDFs that use
    pattern fills like ``pgfpat``), falls back to ``PyMuPDF`` (fitz)
    if installed.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed. Install: pip install pdfplumber")
        return []

    segments = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = ""
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")

            # Table extraction is best-effort — do not let it kill the
            # page's body text if the table API raises or is absent.
            try:
                table_rows = _extract_tables(page)
                for table in table_rows:
                    if table:
                        table_text = _format_table(table)
                        text += "\n" + table_text
            except Exception as e:
                logger.debug(f"Table extraction skipped for page {page_num}: {e}")

            # Fallback: LaTeX PDFs with pattern fills (pgfpat) often
            # yield empty text from pdfplumber. Try PyMuPDF if the
            # page came back empty.
            if not text.strip():
                text = _pymupdf_fallback(path, page_num) or ""

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

    if not segments:
        logger.warning(f"No text extracted from PDF: {path}")

    return segments


def _pymupdf_fallback(path: Path, page_num: int) -> str:
    """Try extracting page text with PyMuPDF (fitz).

    Returns empty string if PyMuPDF is not installed or extraction
    fails. This is a best-effort fallback — warnings are debug-level
    because the caller already handles the "no text" case.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        return ""
    try:
        doc = fitz.open(str(path))
        # fitz uses 0-indexed pages
        if page_num - 1 < len(doc):
            page = doc[page_num - 1]
            text = page.get_text() or ""
            doc.close()
            return text
        doc.close()
    except Exception as e:
        logger.debug(f"PyMuPDF fallback failed for page {page_num}: {e}")
    return ""


def _extract_tables(page) -> List[List[List[str]]]:
    """Extract tables from a pdfplumber Page, compatible with both
    old (``page.tables``) and new (``page.find_tables()``) APIs."""
    # pdfplumber >= 0.11 removed the ``.tables`` property in favour
    # of ``find_tables()`` which returns Table objects whose
    # ``.extract()`` method yields List[List[str]].
    if hasattr(page, "find_tables"):
        tables = page.find_tables()
        return [t.extract() for t in tables if t is not None]
    # Older pdfplumber: page.tables was a list of List[List[str]].
    if hasattr(page, "tables"):
        return page.tables  # type: ignore[attr-defined]
    # Last resort: extract_tables() returns List[List[List[str]]].
    if hasattr(page, "extract_tables"):
        return page.extract_tables()
    return []


def _format_table(table: List[List[str]]) -> str:
    """Format table as pipe-delimited text."""
    if not table:
        return ""

    lines = []
    for row in table:
        formatted_row = " | ".join(str(cell) if cell else "" for cell in row)
        lines.append(formatted_row)

    return "\n".join(lines)
