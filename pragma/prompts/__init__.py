"""Prompt management — load LLM prompts from .txt files with sensible defaults.

This makes prompts user-customizable without code changes:
  - Edit ``pragma/prompts/<name>.txt`` to override.
  - Override file path via ``PRAGMA_PROMPT_<NAME>`` env var.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=16)
def load_prompt(name: str, default: str = "") -> str:
    """Load a prompt by logical name.

    Resolution order:
      1. ``PRAGMA_PROMPT_<NAME_UPPER>`` env var pointing at a file.
      2. ``pragma/prompts/<name>.txt`` shipped with the package.
      3. ``default`` argument (caller-supplied built-in fallback).

    Args:
        name: Logical prompt name, e.g. ``"fact_extraction"``.
        default: In-code fallback used when no file is found.

    Returns:
        The prompt text. Never raises; always returns a usable string.
    """
    env_key = f"PRAGMA_PROMPT_{name.upper()}"
    env_path = os.environ.get(env_key)
    if env_path:
        try:
            return Path(env_path).read_text(encoding="utf-8").strip()
        except OSError as e:
            logger.warning(f"Could not read prompt override {env_path}: {e}")

    path = _PROMPTS_DIR / f"{name}.txt"
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError as e:
            logger.warning(f"Could not read prompt file {path}: {e}")

    return default


__all__ = ["load_prompt"]
