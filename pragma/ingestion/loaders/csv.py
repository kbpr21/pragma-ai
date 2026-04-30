import csv
from pathlib import Path
from typing import List

from pragma.ingestion.loader import DocumentSegment


def load_csv_file(path: Path) -> List[DocumentSegment]:
    """Load CSV file, converting rows to natural language."""
    segments = []
    rows_read = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        if not headers:
            return []

        header_context = "Columns: " + ", ".join(headers)

        for row_num, row in enumerate(reader, start=1):
            row_text = _format_row(row_num, row, headers)
            segments.append(
                DocumentSegment(
                    content=row_text,
                    source=str(path),
                    doc_type="csv",
                    metadata={
                        "filename": path.name,
                        "row": row_num,
                        "headers": headers,
                    },
                )
            )
            rows_read += 1

    if segments and header_context:
        first_segment = segments[0]
        segments[0] = DocumentSegment(
            content=header_context + "\n\n" + first_segment.content,
            source=first_segment.source,
            doc_type=first_segment.doc_type,
            metadata=first_segment.metadata,
        )

    return segments


def _format_row(row_num: int, row: dict, headers: List[str]) -> str:
    """Convert CSV row to natural language."""
    parts = [f"Row {row_num}:"]
    for header in headers:
        value = row.get(header, "")
        if value and value.strip():
            parts.append(f"{header}={value}")
        else:
            parts.append(f"{header}=")

    return ", ".join(parts)
