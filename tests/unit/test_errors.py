from pragma.exceptions import (
    LLMError,
    IngestionError,
    StorageError,
    QueryError,
    GraphError,
    ConfigurationError,
    configure_logging,
)


class TestPragmaErrors:
    """Error class tests."""

    def test_llm_error_has_remediation(self):
        """LLMError shows remediation."""
        err = LLMError("API key invalid")
        assert "API key" in str(err)
        assert "INCEPTION_API_KEY" in err.remediation

    def test_ingestion_error_has_remediation(self):
        """IngestionError shows remediation."""
        err = IngestionError("Could not parse file", filename="test.pdf")
        assert "Supported formats" in err.remediation
        assert err.filename == "test.pdf"

    def test_storage_error_has_remediation(self):
        """StorageError shows remediation."""
        err = StorageError("Directory not writable", path="/tmp/kb")
        assert "writable" in err.remediation
        assert err.path == "/tmp/kb"

    def test_query_error_has_remediation(self):
        """QueryError shows remediation."""
        err = QueryError("Query failed")
        assert "simplifying" in err.remediation

    def test_graph_error_has_remediation(self):
        """GraphError shows remediation."""
        err = GraphError("Graph error")
        assert "re-ingesting" in err.remediation

    def test_config_error_has_remediation(self):
        """ConfigurationError shows remediation."""
        err = ConfigurationError("Config invalid")
        assert "PRAGMA_*" in err.remediation

    def test_configure_logging_sets_level(self):
        """configure_logging sets correct level."""
        import logging

        configure_logging("DEBUG")
        pragma_logger = logging.getLogger("pragma")
        assert pragma_logger.level == logging.DEBUG

        configure_logging("INFO")
        pragma_logger = logging.getLogger("pragma")
        assert pragma_logger.level == logging.INFO

    def test_error_with_custom_remediation(self):
        """Custom remediation overrides default."""
        err = LLMError("Error", remediation="Custom fix")
        assert err.remediation == "Custom fix"
