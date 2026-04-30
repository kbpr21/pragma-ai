from typing import Any, Dict, Optional

from pragma.llm.anthropic import AnthropicProvider
from pragma.llm.base import LLMError, LLMProvider
from pragma.llm.groq import GroqProvider
from pragma.llm.inception import InceptionProvider
from pragma.llm.ollama import OllamaProvider
from pragma.llm.openai import OpenAIProvider


def get_provider(
    name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """Get an LLM provider by name."""
    providers: Dict[str, type] = {
        "groq": GroqProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "inception": InceptionProvider,
        "mercury": InceptionProvider,
    }

    name = name.lower()
    if name not in providers:
        available = ", ".join(providers.keys())
        raise LLMError(f"Unknown provider: {name}. Available: {available}")

    provider_class = providers[name]

    if name == "ollama":
        return provider_class(model=model or "mistral", **kwargs)
    elif name == "groq":
        return provider_class(
            api_key=api_key, model=model or "llama-3.3-70b-versatile", **kwargs
        )
    elif name == "openai":
        return provider_class(api_key=api_key, model=model or "gpt-4o-mini", **kwargs)
    elif name == "anthropic":
        return provider_class(
            api_key=api_key, model=model or "claude-haiku-4-5-20251001", **kwargs
        )
    elif name in ("inception", "mercury"):
        return provider_class(api_key=api_key, model=model or "mercury-2", **kwargs)

    return provider_class(api_key=api_key, model=model, **kwargs)


__all__ = [
    "GroqProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "InceptionProvider",
    "get_provider",
    "LLMProvider",
    "LLMError",
]
