import asyncio
import functools
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from pragma.llm.base import LLMError


class GroqProvider:
    """Groq LLM provider."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise LLMError(
                "GROQ_API_KEY not set. Set GROQ_API_KEY environment variable or pass api_key."
            )
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._get_client().post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
        )
        if response.status_code != 200:
            raise LLMError(f"Groq API error: {response.status_code} - {response.text}")
        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise LLMError(f"Invalid response: {data}")
        return data

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Synchronous completion with retry."""
        last_error: Exception = LLMError("Unknown error")

        for attempt in range(3):
            try:
                data = self._make_request(messages, **kwargs)
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPError as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2**attempt)

        raise LLMError(f"Failed after 3 attempts: {last_error}")

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
        """Stream tokens via Server-Sent Events (OpenAI-compatible)."""
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "stream": True,
        }
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.BASE_URL}/chat/completions",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
            except httpx.HTTPError as e:
                raise LLMError(f"Groq stream failed: {e}")

    def __enter__(self) -> "GroqProvider":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def list_models(
        cls,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Return the available chat models via Groq's OpenAI-compatible
        ``/openai/v1/models`` endpoint.

        Groq's catalogue is small and curated, so we return everything
        the API surfaces (filtering only obvious whisper/audio entries).
        """
        url = (base_url or cls.BASE_URL).rstrip("/")
        try:
            resp = httpx.get(
                f"{url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Cannot reach Groq /v1/models at {url}: {e}")
        if resp.status_code == 401:
            raise LLMError("Groq API key was rejected (401 Unauthorized).")
        if resp.status_code != 200:
            raise LLMError(
                f"Groq /v1/models returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json().get("data", []) or []
        chat_models = [
            m
            for m in data
            if "whisper" not in m.get("id", "").lower()
            and "tts" not in m.get("id", "").lower()
        ]
        chat_models.sort(key=lambda m: m.get("id", ""))
        return chat_models
