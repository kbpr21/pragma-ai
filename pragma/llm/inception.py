"""
InceptionLabs Mercury API provider.
 Mercury's API is compatible with OpenAI's chat completions API.
"""

import os
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx

from pragma.llm.base import LLMError


class InceptionProvider:
    """InceptionLabs Mercury LLM provider."""

    BASE_URL = "https://api.inceptionlabs.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mercury-2",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("INCEPTION_API_KEY", "")
        if not self.api_key:
            raise LLMError(
                "INCEPTION_API_KEY not set. Set INCEPTION_API_KEY environment variable or pass api_key."
            )
        self.model = model
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Synchronous completion."""
        client = self._get_client()

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise LLMError(
                    f"Inception API error: {e.response.status_code} - Invalid API Key"
                )
            raise LLMError(
                f"Inception API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            raise LLMError(f"Inception API request failed: {e}")

        result = response.json()

        if "error" in result:
            raise LLMError(f"Inception API error: {result['error']}")

        return result["choices"][0]["message"]["content"]

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Asynchronous completion (offloads sync call to thread pool).

        Note: ``loop.run_in_executor`` does not accept ``**kwargs``; we use
        ``functools.partial`` to bind keyword arguments correctly.
        """
        import asyncio
        import functools

        loop = asyncio.get_event_loop()
        func = functools.partial(self.complete, messages, **kwargs)
        return await loop.run_in_executor(None, func)

    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion using async HTTP client."""
        import httpx

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
                    f"{self.base_url}/chat/completions",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            if not data:
                                continue
                            import json

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
            except httpx.HTTPStatusError as e:
                raise LLMError(
                    f"Inception API error: {e.response.status_code} - {e.response.text}"
                )
            except httpx.RequestError as e:
                raise LLMError(f"Inception API request failed: {e}")
