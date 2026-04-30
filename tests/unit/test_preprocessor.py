from pragma.ingestion.loader import DocumentSegment
from pragma.ingestion.preprocessor import (
    DocumentPreprocessor,
    ProcessedSegment,
)


class TestDocumentPreprocessor:
    def test_single_segment_no_split(self):
        preprocessor = DocumentPreprocessor(max_tokens=300)
        segments = [
            DocumentSegment(
                content="Short content.",
                source="test.txt",
                doc_type="txt",
            )
        ]

        processed = preprocessor.preprocess(segments)
        assert len(processed) == 1
        assert processed[0].content == "Short content."

    def test_multiple_sentences(self):
        preprocessor = DocumentPreprocessor(max_tokens=10)
        text = "This is sentence one. This is sentence two. This is sentence three."
        segments = [DocumentSegment(content=text, source="test.txt", doc_type="txt")]

        processed = preprocessor.preprocess(segments)
        assert len(processed) >= 2

    def test_duplication_skip(self):
        preprocessor = DocumentPreprocessor()

        content = "Duplicate content."
        segments = [
            DocumentSegment(content=content, source="test1.txt", doc_type="txt"),
            DocumentSegment(content=content, source="test2.txt", doc_type="txt"),
        ]

        processed = preprocessor.preprocess(segments)
        assert len(processed) == 1

    def test_different_content_both_kept(self):
        preprocessor = DocumentPreprocessor()

        segments = [
            DocumentSegment(
                content="First content.", source="test1.txt", doc_type="txt"
            ),
            DocumentSegment(
                content="Second content.", source="test2.txt", doc_type="txt"
            ),
        ]

        processed = preprocessor.preprocess(segments)
        assert len(processed) == 2

    def test_normalized_dedup(self):
        preprocessor = DocumentPreprocessor()

        segments = [
            DocumentSegment(content="  CONTENT  ", source="test1.txt", doc_type="txt"),
            DocumentSegment(content="content", source="test2.txt", doc_type="txt"),
        ]

        processed = preprocessor.preprocess(segments)
        assert len(processed) == 1

    def test_metadata_enrichment(self):
        preprocessor = DocumentPreprocessor()

        segments = [
            DocumentSegment(
                content="Test content.",
                source="test.pdf",
                doc_type="pdf",
                metadata={"page": 5, "filename": "test.pdf"},
            )
        ]

        processed = preprocessor.preprocess(segments)
        assert len(processed) == 1
        assert processed[0].metadata["page"] == 5
        assert processed[0].metadata["filename"] == "test.pdf"
        assert "timestamp" in processed[0].metadata

    def test_chunk_metadata(self):
        preprocessor = DocumentPreprocessor(max_tokens=10)

        text = "One. Two. Three. Four. Five. Six. Seven."
        segments = [DocumentSegment(content=text, source="test.txt", doc_type="txt")]

        processed = preprocessor.preprocess(segments)

        chunk_indices = [p.chunk_index for p in processed]
        assert len(set(chunk_indices)) == len(chunk_indices)

    def test_reset_seen(self):
        preprocessor = DocumentPreprocessor()

        segments = [
            DocumentSegment(content="Same.", source="test1.txt", doc_type="txt"),
            DocumentSegment(content="Same.", source="test2.txt", doc_type="txt"),
        ]

        preprocessor.preprocess(segments)
        assert len(preprocessor._seen_hashes) == 1

        preprocessor.reset_seen()
        assert len(preprocessor._seen_hashes) == 0

    def test_add_seen_hash(self):
        preprocessor = DocumentPreprocessor()
        preprocessor.add_seen_hash("abc123")

        assert "abc123" in preprocessor._seen_hashes


class TestProcessedSegment:
    def test_content_hash(self):
        segment = ProcessedSegment(
            content="Test content",
            source="test.txt",
            doc_type="txt",
            chunk_index=0,
            content_hash="hash123",
        )

        assert segment.content_hash == "hash123"
        assert segment.chunk_index == 0


class TestSentenceBoundaryDetection:
    def test_sentence_splitting(self):
        from pragma.ingestion.preprocessor import SENTENCE_ENDINGS

        text = "First sentence. Second sentence! Third sentence? Fourth."
        sentences = SENTENCE_ENDINGS.split(text)

        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_sentences_with_abbreviations(self):
        from pragma.ingestion.preprocessor import SENTENCE_ENDINGS

        text = "Dr. Smith arrived. He went to the U.S. yesterday."
        sentences = SENTENCE_ENDINGS.split(text)

        assert len(sentences) >= 2

    def test_newline_handling(self):
        from pragma.ingestion.preprocessor import SENTENCE_ENDINGS

        text = "First.\n\nSecond.\r\nThird."
        sentences = SENTENCE_ENDINGS.split(text)

        joined = " ".join(s.strip() for s in sentences if s.strip())
        assert "First" in joined
        assert "Second" in joined
