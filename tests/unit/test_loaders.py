import pytest
from pathlib import Path
import tempfile

from pragma.ingestion.loader import DocumentLoader, DocumentSegment


class TestDocumentLoaderDispatch:
    def test_load_txt_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello world")
            path = f.name

        try:
            loader = DocumentLoader()
            segments = loader.load(path)
            assert len(segments) == 1
            assert segments[0].content == "Hello world"
            assert segments[0].doc_type == "txt"
        finally:
            Path(path).unlink()

    def test_load_md_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Title\n\nSome content")
            path = f.name

        try:
            loader = DocumentLoader()
            segments = loader.load(path)
            assert len(segments) == 1
            assert segments[0].doc_type == "txt"
        finally:
            Path(path).unlink()

    def test_load_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"key": "value"}')
            path = f.name

        try:
            loader = DocumentLoader()
            segments = loader.load(path)
            assert len(segments) == 1
            assert "key=value" in segments[0].content
        finally:
            Path(path).unlink()

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("name,value\ntest,123")
            path = f.name

        try:
            loader = DocumentLoader()
            segments = loader.load(path)
            assert len(segments) == 1
            assert "name=value" in segments[0].content or "test" in segments[0].content
        finally:
            Path(path).unlink()

    def test_load_dict(self):
        loader = DocumentLoader()
        segments = loader.load({"key": "value"})
        assert len(segments) == 1

    def test_unsupported_file(self):
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load("test.xyz")


class TestTextLoader:
    def test_load_text_file(self):
        from pragma.ingestion.loaders.text import load_text_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Test content")
            path = f.name

        try:
            segment = load_text_file(Path(path))
            assert segment.content == "Test content"
            assert segment.doc_type == "txt"
            assert segment.metadata["char_count"] == 12
        finally:
            Path(path).unlink()


class TestJsonLoader:
    def test_flatten_json(self):
        from pragma.ingestion.loaders.json import _flatten_json_single

        result = _flatten_json_single({"a": {"b": "c"}}, "", 0)
        assert "a.b=c" in result

    def test_flatten_json_list(self):
        from pragma.ingestion.loaders.json import _flatten_json_single

        result = _flatten_json_single([1, 2, 3], "", 0)
        assert "[0]=1" in result


class TestCsvLoader:
    def test_format_row(self):
        from pragma.ingestion.loaders.csv import _format_row

        row = {"name": "test", "value": "123"}
        result = _format_row(1, row, ["name", "value"])
        assert "Row 1:" in result
        assert "name=test" in result
        assert "value=123" in result


class TestDocumentSegment:
    def test_repr(self):
        segment = DocumentSegment(
            content="test",
            source="test.txt",
            doc_type="txt",
        )
        assert "test.txt" in repr(segment)
        assert "txt" in repr(segment)
