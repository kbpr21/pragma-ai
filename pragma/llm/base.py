from typing import Protocol, List, Dict, Any, AsyncGenerator

from pragma.exceptions import LLMError

__all__ = ["LLMProvider", "LLMError"]


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Synchronous completion."""
        ...

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Asynchronous completion."""
        ...

    async def stream_complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Streaming completion yielding tokens."""
        ...

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...
