import pytest
from unittest.mock import patch, MagicMock

from pragma.llm import (
    InceptionProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)


class TestInceptionProviderCoverage:
    """Increase inception.py coverage."""

    def test_api_error_response(self):
        """Test API error handling."""
        import httpx

        with patch("pragma.llm.inception.httpx.Client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = "Invalid key"
            mock_client.return_value.post.side_effect = httpx.HTTPStatusError(
                "401", request=MagicMock(), response=mock_resp
            )
            with pytest.raises(Exception):
                p = InceptionProvider(api_key="test")
                p.complete([{"role": "user", "content": "hi"}])

    def test_request_error(self):
        """Test network error handling."""
        import httpx

        with patch("pragma.llm.inception.httpx.Client") as mock_client:
            mock_client.return_value.post.side_effect = httpx.RequestError(
                "Network error"
            )
            with pytest.raises(Exception):
                p = InceptionProvider(api_key="test")
                p.complete([{"role": "user", "content": "hi"}])


class TestOpenAIProviderCoverage:
    """Increase openai.py coverage."""

    def test_api_error(self):
        """Test OpenAI error handling."""
        import httpx

        with patch("pragma.llm.openai.httpx.Client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.text = "Rate limit"
            mock_client.return_value.post.side_effect = httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_resp
            )
            with pytest.raises(Exception):
                p = OpenAIProvider(api_key="test")
                p.complete([{"role": "user", "content": "hi"}])


class TestAnthropicProviderCoverage:
    """Increase anthropic.py coverage."""

    def test_anthropic_error(self):
        """Test Anthropic error handling."""
        import httpx

        with patch("pragma.llm.anthropic.httpx.Client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Server error"
            mock_client.return_value.post.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=mock_resp
            )
            with pytest.raises(Exception):
                p = AnthropicProvider(api_key="test")
                p.complete([{"role": "user", "content": "hi"}])


class TestOllamaProviderCoverage:
    """Increase ollama.py coverage."""

    def test_ollama_class_constant(self):
        """Test BASE_URL class constant exists."""
        assert hasattr(OllamaProvider, "BASE_URL")
        assert OllamaProvider.BASE_URL == "http://localhost:11434"

    def test_ollama_model_default(self):
        """Test default model."""
        p = OllamaProvider()
        assert p.model == "mistral"
