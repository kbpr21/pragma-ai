import json
from pathlib import Path
from typing import Any, List

from pragma.ingestion.loader import DocumentSegment


def load_json_file(path: Path) -> List[DocumentSegment]:
    """Load JSON/JSONL file."""
    content = path.read_text(encoding="utf-8")

    if path.suffix == ".jsonl":
        return load_jsonl(content, str(path))

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    return _flatten_json(data, str(path))


def load_jsonl(content: str, source: str) -> List[DocumentSegment]:
    """Load JSONL (JSON Lines) file."""
    segments = []
    for line_num, line in enumerate(content.strip().split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            flattened = _flatten_json_single(data, "", 0)
            segments.append(
                DocumentSegment(
                    content=flattened,
                    source=source,
                    doc_type="json",
                    metadata={"line": line_num},
                )
            )
        except json.JSONDecodeError:
            continue

    return segments


def load_json_dict(data: dict) -> List[DocumentSegment]:
    """Load JSON from dict."""
    flattened = _flatten_json_single(data, "", 0)
    return [
        DocumentSegment(
            content=flattened,
            source="dict",
            doc_type="json",
            metadata={},
        )
    ]


def _flatten_json(data: Any, source: str) -> List[DocumentSegment]:
    """Flatten JSON to segments."""
    if isinstance(data, list):
        segments = []
        for i, item in enumerate(data, start=1):
            flattened = _flatten_json_single(item, "", 0)
            segments.append(
                DocumentSegment(
                    content=flattened,
                    source=source,
                    doc_type="json",
                    metadata={"index": i},
                )
            )
        return segments

    flattened = _flatten_json_single(data, "", 0)
    return [
        DocumentSegment(
            content=flattened,
            source=source,
            doc_type="json",
            metadata={},
        )
    ]


def _flatten_json_single(data: Any, prefix: str, depth: int) -> str:
    """Recursively flatten JSON to key=value pairs."""
    if depth > 10:
        return str(data)

    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            parts.append(_flatten_json_single(value, new_key, depth + 1))
        return "; ".join(parts)

    if isinstance(data, list):
        parts = []
        for i, item in enumerate(data):
            new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            parts.append(_flatten_json_single(item, new_key, depth + 1))
        return "; ".join(parts)

    return f"{prefix}={data}"
