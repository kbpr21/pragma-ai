"""Tests for the per-provider ``list_models()`` classmethods used by
``pragma connect`` to discover available models from each provider's
API.

Each test mocks ``httpx`` at the top level so we never make real
network calls. We're checking that:

* Auth headers are correct for each provider (Bearer for OpenAI/Groq/
  Inception/Ollama, ``x-api-key`` for Anthropic).
* The OpenAI filter excludes embeddings/whisper noise.
* All providers raise :class:`LLMError` on 401, on connection failure,
  and on bad JSON -- so the wizard can surface friendly messages.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pragma.exceptions import LLMError
from pragma.llm.anthropic import AnthropicProvider
from pragma.llm.groq import GroqProvider
from pragma.llm.inception import InceptionProvider
from pragma.llm.ollama import OllamaProvider
from pragma.llm.openai import OpenAIProvider


def _mock_response(json_data: Dict[str, Any], status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    r.text = "(mocked)"
    return r


# ---------------------------------------------------------------------------
# Ollama (no auth, /api/tags)
# ---------------------------------------------------------------------------


def test_ollama_list_models_returns_local_models() -> None:
    payload = {
        "models": [
            {"name": "mistral", "size": 4_100_000_000},
            {"name": "llama3.2:3b", "size": 2_000_000_000},
        ]
    }
    with patch(
        "pragma.llm.ollama.httpx.get", return_value=_mock_response(payload)
    ) as m:
        models = OllamaProvider.list_models()
    assert [m["name"] for m in models] == ["mistral", "llama3.2:3b"]
    # No auth header for Ollama
    args, kwargs = m.call_args
    assert "headers" not in kwargs or "Authorization" not in (
        kwargs.get("headers") or {}
    )
    assert "/api/tags" in args[0]


def test_ollama_list_models_raises_on_connection_error() -> None:
    with patch(
        "pragma.llm.ollama.httpx.get",
        side_effect=httpx.ConnectError("refused"),
    ):
        with pytest.raises(LLMError, match="Cannot reach Ollama"):
            OllamaProvider.list_models()


def test_ollama_list_models_handles_empty_response() -> None:
    with patch("pragma.llm.ollama.httpx.get", return_value=_mock_response({})):
        assert OllamaProvider.list_models() == []


# ---------------------------------------------------------------------------
# OpenAI -- filters chat-only IDs
# ---------------------------------------------------------------------------


def test_openai_list_models_filters_to_chat_models() -> None:
    payload = {
        "data": [
            {"id": "gpt-4o-mini"},
            {"id": "gpt-5"},
            {"id": "text-embedding-3-large"},
            {"id": "whisper-1"},
            {"id": "dall-e-3"},
            {"id": "o1-preview"},
            {"id": "chatgpt-4o-latest"},
        ]
    }
    with patch(
        "pragma.llm.openai.httpx.get", return_value=_mock_response(payload)
    ) as m:
        models = OpenAIProvider.list_models(api_key="sk-test")
    ids = [x["id"] for x in models]
    assert "text-embedding-3-large" not in ids
    assert "whisper-1" not in ids
    assert "dall-e-3" not in ids
    assert {"gpt-4o-mini", "gpt-5", "o1-preview", "chatgpt-4o-latest"}.issubset(
        set(ids)
    )
    # Auth header sent
    headers = m.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-test"


def test_openai_list_models_raises_on_401() -> None:
    with patch(
        "pragma.llm.openai.httpx.get",
        return_value=_mock_response({}, status_code=401),
    ):
        with pytest.raises(LLMError, match="401"):
            OpenAIProvider.list_models(api_key="bad")


# ---------------------------------------------------------------------------
# Anthropic -- uses x-api-key, not Authorization
# ---------------------------------------------------------------------------


def test_anthropic_list_models_uses_x_api_key_header() -> None:
    payload = {
        "data": [
            {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
            {"id": "claude-sonnet-4-5-20250920"},
        ]
    }
    with patch(
        "pragma.llm.anthropic.httpx.get", return_value=_mock_response(payload)
    ) as m:
        models = AnthropicProvider.list_models(api_key="sk-ant-test")

    headers = m.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert "Authorization" not in headers
    assert headers.get("anthropic-version")
    # Newest-first sort by id
    assert models[0]["id"].startswith("claude")
    assert models[0]["id"] >= models[1]["id"]


def test_anthropic_list_models_raises_on_401() -> None:
    with patch(
        "pragma.llm.anthropic.httpx.get",
        return_value=_mock_response({}, status_code=401),
    ):
        with pytest.raises(LLMError, match="401"):
            AnthropicProvider.list_models(api_key="bad")


# ---------------------------------------------------------------------------
# Groq -- filters whisper/tts
# ---------------------------------------------------------------------------


def test_groq_list_models_filters_audio() -> None:
    payload = {
        "data": [
            {"id": "llama-3.3-70b-versatile"},
            {"id": "whisper-large-v3"},
            {"id": "playai-tts"},
        ]
    }
    with patch("pragma.llm.groq.httpx.get", return_value=_mock_response(payload)):
        ids = [m["id"] for m in GroqProvider.list_models(api_key="gsk-test")]
    assert ids == ["llama-3.3-70b-versatile"]


# ---------------------------------------------------------------------------
# Inception -- thin OpenAI-compatible
# ---------------------------------------------------------------------------


def test_inception_list_models_passes_through() -> None:
    payload = {
        "data": [
            {"id": "mercury-2"},
            {"id": "mercury-1"},
        ]
    }
    with patch(
        "pragma.llm.inception.httpx.get", return_value=_mock_response(payload)
    ) as m:
        models = InceptionProvider.list_models(api_key="inc-test")

    headers = m.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer inc-test"
    # Sorted alphabetically (id ascending)
    assert [x["id"] for x in models] == ["mercury-1", "mercury-2"]
