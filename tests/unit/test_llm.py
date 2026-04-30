import pytest
from unittest.mock import MagicMock, patch

from pragma.llm import (
    GroqProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    get_provider,
)
from pragma.llm.base import LLMError


class TestGroqProviderInit:
    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMError, match="GROQ_API_KEY not set"):
                GroqProvider()

    def test_with_api_key(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = GroqProvider()
            assert provider.api_key == "test-key"
            assert provider.model == "llama-3.3-70b-versatile"


class TestGroqProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_success(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = GroqProvider()
            with patch.object(provider, "_make_request") as mock_request:
                mock_request.return_value = {
                    "choices": [{"message": {"content": "test response"}}]
                }
                messages = [{"role": "user", "content": "hello"}]
                result = provider.complete(messages)
                assert result == "test response"

    def test_api_error_raises_llm_error(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = GroqProvider()
            with patch.object(provider, "_get_client") as mock_client:
                mock_resp = MagicMock()
                mock_resp.status_code = 401
                mock_resp.text = "Unauthorized"
                mock_resp.json.return_value = {}
                mock_client.return_value.post.return_value = mock_resp

                with pytest.raises(LLMError, match="Groq API error"):
                    provider.complete([{"role": "user", "content": "test"}])


class TestOllamaProvider:
    def test_ollama_not_running(self):
        provider = OllamaProvider()
        with patch.object(provider, "_check_ollama_running", return_value=False):
            with pytest.raises(LLMError, match="Ollama is not running"):
                provider.complete([{"role": "user", "content": "test"}])

    def test_default_model(self):
        provider = OllamaProvider()
        assert provider.model_name == "mistral"


class TestOpenAIProvider:
    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMError, match="OPENAI_API_KEY not set"):
                OpenAIProvider()

    def test_default_model(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            assert provider.model_name == "gpt-4o-mini"


class TestAnthropicProvider:
    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMError, match="ANTHROPIC_API_KEY not set"):
                AnthropicProvider()

    def test_default_model(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()
            assert provider.model_name == "claude-haiku-4-5-20251001"


class TestProviderFactory:
    def test_get_groq(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = get_provider("groq")
            assert isinstance(provider, GroqProvider)

    def test_get_ollama(self):
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_unknown_provider(self):
        with pytest.raises(LLMError, match="Unknown provider"):
            get_provider("unknown")

    def test_case_insensitive(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = get_provider("GROQ")
            assert isinstance(provider, GroqProvider)

    def test_custom_model(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = get_provider("groq", model="custom-model")
            assert provider.model_name == "custom-model"


class TestModelProperty:
    def test_groq_model(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = GroqProvider(model="llama-3.1-70b")
            assert provider.model_name == "llama-3.1-70b"

    def test_ollama_model(self):
        provider = OllamaProvider(model="llama3")
        assert provider.model_name == "llama3"

    def test_openai_model(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(model="gpt-4")
            assert provider.model_name == "gpt-4"

    def test_anthropic_model(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(model="claude-sonnet")
            assert provider.model_name == "claude-sonnet"
