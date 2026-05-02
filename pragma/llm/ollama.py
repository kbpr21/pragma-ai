import asyncio
import functools
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from pragma.llm.base import LLMError


class OllamaProvider:
    """Ollama local LLM provider."""

    BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "mistral",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _check_ollama_running(self) -> bool:
        try:
            response = self._get_client().get(f"{self.BASE_URL}/api/tags")
            return response.status_code == 200
        except httpx.ConnectError:
            return False

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        if not self._check_ollama_running():
            raise LLMError(
                "Ollama is not running. Start Ollama with: ollama serve\n"
                "Or install: curl -fsSL https://ollama.ai/install.sh | sh"
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self._get_client().post(
            f"{self.BASE_URL}/api/chat",
            json=payload,
        )
        if response.status_code != 200:
            raise LLMError(
                f"Ollama API error: {response.status_code} - {response.text}"
            )
        data = response.json()
        if "message" not in data:
            raise LLMError(f"Invalid response: {data}")
        return data

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Synchronous completion."""
        data = self._make_request(messages, **kwargs)
        return data["message"]["content"]

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Async completion (offloads sync call to thread pool)."""
        loop = asyncio.get_event_loop()
        func = functools.partial(self.complete, messages, **kwargs)
        return await loop.run_in_executor(None, func)

    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens via Ollama's NDJSON protocol."""
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.BASE_URL}/api/chat",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        message = chunk.get("message", {})
                        content = message.get("content")
                        if content:
                            yield content
                        if chunk.get("done"):
                            break
            except httpx.HTTPError as e:
                raise LLMError(f"Ollama stream failed: {e}")

    def __enter__(self) -> "OllamaProvider":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def list_models(
        cls,
        base_url: Optional[str] = None,
        timeout: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Return the locally-installed models.

        Hits Ollama's ``/api/tags`` endpoint (no auth required) and
        returns the list verbatim. Each item has at least a ``name``
        field; size, digest, modified_at etc. are passed through
        unchanged so the wizard can show a nicer picker.

        Raises :class:`LLMError` when Ollama is not reachable so
        callers can surface a friendly "is Ollama running?" message.
        """
        url = (base_url or cls.BASE_URL).rstrip("/")
        try:
            resp = httpx.get(f"{url}/api/tags", timeout=timeout)
        except httpx.HTTPError as e:
            raise LLMError(
                f"Cannot reach Ollama at {url}. Is it running? "
                f"Try: ollama serve  ({e})"
            )
        if resp.status_code != 200:
            raise LLMError(
                f"Ollama /api/tags returned {resp.status_code}: {resp.text[:200]}"
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise LLMError(f"Ollama returned non-JSON: {e}")
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []
        return models
