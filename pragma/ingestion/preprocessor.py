import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from pragma.ingestion.loader import DocumentSegment

logger = logging.getLogger(__name__)

SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")
MAX_TOKEN_ESTIMATE = 300
TOKENS_PER_WORD = 1.3


@dataclass
class ProcessedSegment:
    """Segment after preprocessing."""

    content: str
    source: str
    doc_type: str
    chunk_index: int
    content_hash: str
    metadata: dict = field(default_factory=dict)


class DocumentPreprocessor:
    """Preprocess documents for fact extraction."""

    def __init__(self, max_tokens: int = MAX_TOKEN_ESTIMATE) -> None:
        self.max_tokens = max_tokens
        self._seen_hashes: set = set()

    def preprocess(
        self,
        segments: List[DocumentSegment],
    ) -> List[ProcessedSegment]:
        """Preprocess document segments."""
        processed = []

        for segment in segments:
            content = segment.content.strip()
            if not content:
                continue

            doc_hash = self._compute_hash(content)

            if self._is_duplicate(doc_hash):
                logger.debug(f"Skipping duplicate content: {segment.source}")
                continue

            chunks = self._split_into_chunks(content)

            for chunk_idx, chunk_content in enumerate(chunks):
                chunk_hash = self._compute_hash(chunk_content)
                processed.append(
                    ProcessedSegment(
                        content=chunk_content,
                        source=segment.source,
                        doc_type=segment.doc_type,
                        chunk_index=chunk_idx,
                        content_hash=chunk_hash,
                        metadata={
                            "source_doc": segment.source,
                            "page": segment.metadata.get("page"),
                            "line": segment.metadata.get("line"),
                            "filename": segment.metadata.get("filename"),
                            "char_count": len(chunk_content),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )

        return processed

    def _compute_hash(self, content: str) -> str:
        normalized = content.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _is_duplicate(self, content_hash: str) -> bool:
        if content_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(content_hash)
        return False

    def _split_into_chunks(self, content: str) -> List[str]:
        sentences = SENTENCE_ENDINGS.split(content)
        if not sentences:
            return [content] if content else []

        chunks = []
        current_chunk = ""
        current_word_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            word_count = int(len(sentence.split()) * TOKENS_PER_WORD)
            current_word_count += word_count

            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

            if current_word_count >= self.max_tokens:
                chunks.append(current_chunk)
                current_chunk = ""
                current_word_count = 0

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    def reset_seen(self) -> None:
        """Reset the duplicate tracking set."""
        self._seen_hashes.clear()

    def add_seen_hash(self, content_hash: str) -> None:
        """Add a hash to the seen set (for incremental updates)."""
        self._seen_hashes.add(content_hash)
