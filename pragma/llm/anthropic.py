import asyncio
import functools
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from pragma.llm.base import LLMError


class AnthropicProvider:
    """Anthropic LLM provider."""

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY not set. Set ANTHROPIC_API_KEY environment variable or pass api_key."
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
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
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
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        payload = {
            "model": self.model,
            "messages": filtered_messages,
            "system": system_content,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._get_client().post(
            f"{self.BASE_URL}/messages",
            json=payload,
        )
        if response.status_code != 200:
            raise LLMError(
                f"Anthropic API error: {response.status_code} - {response.text}"
            )
        data = response.json()
        if "content" not in data or not data["content"]:
            raise LLMError(f"Invalid response: {data}")
        return data

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Synchronous completion with retry."""
        last_error: Exception = LLMError("Unknown error")
        for attempt in range(3):
            try:
                data = self._make_request(messages, **kwargs)
                return data["content"][0]["text"]
            except httpx.HTTPError as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2**attempt)
        raise LLMError(f"Anthropic failed after 3 attempts: {last_error}")

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
        """Stream tokens via Anthropic SSE protocol."""
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": filtered_messages,
            "system": system_content,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "stream": True,
        }
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.BASE_URL}/messages",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            text = delta.get("text")
                            if text:
                                yield text
                        elif event.get("type") == "message_stop":
                            break
            except httpx.HTTPError as e:
                raise LLMError(f"Anthropic stream failed: {e}")

    def __enter__(self) -> "AnthropicProvider":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
