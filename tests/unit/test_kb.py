import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch


from pragma.kb import IngestResult, KnowledgeBase


class MockLLMProvider:
    """Mock LLM provider."""

    def __init__(self, response: str = "[]"):
        self.response = response

    def complete(self, messages: List[dict], **kwargs) -> str:
        return self.response

    async def acomplete(self, messages: List[dict], **kwargs) -> str:
        return self.response

    @property
    def model_name(self) -> str:
        return "mock-model"


class TestIngestResult:
    """Tests for IngestResult."""

    def test_create_with_values(self):
        result = IngestResult(documents=5, facts=100, entities=20, skipped=2)
        assert result.documents == 5
        assert result.facts == 100
        assert result.entities == 20
        assert result.skipped == 2

    def test_repr(self):
        result = IngestResult(documents=1, facts=10, entities=5)
        assert "1" in repr(result)

    def test_summary_single(self):
        result = IngestResult(documents=1, facts=10, entities=5)
        summary = result.summary()
        assert "1 document" in summary
        assert "10 facts" in summary
        assert "5 entities" in summary

    def test_summary_plural(self):
        result = IngestResult(documents=2, facts=0, entities=0)
        summary = result.summary()
        assert "documents" in summary

    def test_summary_empty(self):
        result = IngestResult()
        summary = result.summary()
        assert "No changes" in summary


class TestKnowledgeBaseIngest:
    """Tests for KnowledgeBase.ingest()."""

    def test_ingest_single_file(self, tmp_path):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Test content about companies.")
            path = f.name

        try:
            kb = KnowledgeBase(llm=MockLLMProvider("[]"), kb_dir=str(tmp_path))

            with patch.object(kb._extractor, "extract", return_value=[]):
                result = kb.ingest(path, show_progress=False)

            assert result.documents == 1
            kb.close()
        finally:
            Path(path).unlink()

    def test_ingest_dict(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLMProvider("[]"), kb_dir=str(tmp_path))

        with patch.object(kb._extractor, "extract", return_value=[]):
            result = kb.ingest({"key": "value"}, show_progress=False)

        assert result.documents == 1
        kb.close()

    def test_ingest_list(self, tmp_path):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Content 1")
            path1 = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Content 2")
            path2 = f.name

        try:
            kb = KnowledgeBase(llm=MockLLMProvider("[]"), kb_dir=str(tmp_path))

            with patch.object(kb._extractor, "extract", return_value=[]):
                result = kb.ingest([path1, path2], show_progress=False)

            assert result.documents == 2
            kb.close()
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_ingest_skips_duplicates(self, tmp_path):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Duplicate content")
            path = f.name

        try:
            kb = KnowledgeBase(llm=MockLLMProvider("[]"), kb_dir=str(tmp_path))

            with patch.object(kb._extractor, "extract", return_value=[]):
                kb.ingest(path, show_progress=False)
                result = kb.ingest(path, show_progress=False)

            assert result.skipped == 1
            kb.close()
        finally:
            Path(path).unlink()

    def test_ingest_directory(self, tmp_path):
        dir_path = tmp_path / "docs"
        dir_path.mkdir()
        (dir_path / "a.txt").write_text("Content A")
        (dir_path / "b.txt").write_text("Content B")

        kb = KnowledgeBase(llm=MockLLMProvider("[]"), kb_dir=str(tmp_path / "kb"))

        with patch.object(kb._extractor, "extract", return_value=[]):
            result = kb.ingest(dir_path, show_progress=False)

        assert result.documents == 2
        kb.close()

    def test_ingest_extracts_facts(self, tmp_path):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Apple is a company.")
            path = f.name

        try:
            kb = KnowledgeBase(
                llm=MockLLMProvider(
                    '[{"subject": "Apple", "predicate": "is", "object": "company", "confidence": 1.0}]'
                ),
                kb_dir=str(tmp_path),
            )

            result = kb.ingest(path, show_progress=False)

            assert result.facts >= 0
            kb.close()
        finally:
            Path(path).unlink()

    def test_discover_files(self, tmp_path):
        dir_path = tmp_path / "docs"
        dir_path.mkdir()
        (dir_path / "a.txt").write_text("a")
        (dir_path / "b.md").write_text("b")
        (dir_path / "c.pdf").write_text("c")
        (dir_path / "d.json").write_text("{}")

        kb = KnowledgeBase(llm=MockLLMProvider(), kb_dir=str(tmp_path / "kb"))
        files = kb._discover_files(dir_path)

        assert len(files) == 4
        kb.close()


class TestKnowledgeBaseStats:
    """Tests for KnowledgeBase.stats()."""

    def test_stats_empty(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLMProvider(), kb_dir=str(tmp_path))
        stats = kb.stats()

        assert stats.documents == 0
        assert stats.facts == 0
        assert stats.entities == 0
        assert stats.relationships == 0
        kb.close()


class TestKnowledgeBaseContextManager:
    """Tests for context manager."""

    def test_context_manager(self, tmp_path):
        with KnowledgeBase(llm=MockLLMProvider(), kb_dir=str(tmp_path)) as kb:
            stats = kb.stats()
            assert stats is not None
