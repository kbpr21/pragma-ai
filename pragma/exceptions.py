"""Pragma exceptions and error handling."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure pragma logging.

    Args:
        level: DEBUG, INFO, WARNING, or ERROR
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("pragma").setLevel(numeric_level)


class PragmaError(Exception):
    """Base exception for pragma."""

    def __init__(self, message: str, remediation: Optional[str] = None):
        self.message = message
        self.remediation = remediation
        super().__init__(message)

    def __str__(self) -> str:
        if self.remediation:
            return f"{self.message}\nRemediation: {self.remediation}"
        return self.message


class LLMError(PragmaError):
    """Error from LLM provider."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        remediation: Optional[str] = None,
    ):
        default_remediation = (
            "Check that your API key is set: "
            "INCEPTION_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY"
        )
        super().__init__(message, remediation or default_remediation)
        self.provider = provider


class IngestionError(PragmaError):
    """Error during document ingestion."""

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        remediation: Optional[str] = None,
    ):
        default_remediation = (
            "Supported formats: pdf, csv, json, jsonl, txt, md, docx, html. "
            "Ensure the file is readable and properly formatted."
        )
        super().__init__(message, remediation or default_remediation)
        self.filename = filename


class StorageError(PragmaError):
    """Error during storage operations."""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        remediation: Optional[str] = None,
    ):
        default_remediation = (
            "Ensure the KB directory exists and is writable. Check file permissions."
        )
        super().__init__(message, remediation or default_remediation)
        self.path = path


class QueryError(PragmaError):
    """Error during query processing."""

    def __init__(
        self,
        message: str,
        remediation: Optional[str] = None,
    ):
        default_remediation = "Try simplifying your query or ingesting more documents."
        super().__init__(message, remediation or default_remediation)


class GraphError(PragmaError):
    """Error during graph operations."""

    def __init__(
        self,
        message: str,
        remediation: Optional[str] = None,
    ):
        default_remediation = "Try re-ingesting your documents to rebuild the graph."
        super().__init__(message, remediation or default_remediation)


class ConfigurationError(PragmaError):
    """Error in configuration."""

    def __init__(
        self,
        message: str,
        remediation: Optional[str] = None,
    ):
        default_remediation = (
            "Check PRAGMA_* environment variables or configuration file."
        )
        super().__init__(message, remediation or default_remediation)
