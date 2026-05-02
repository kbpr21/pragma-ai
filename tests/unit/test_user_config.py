"""Tests for ``pragma.user_config`` -- the persisted user config layer
that backs ``pragma connect``.

We cover the four behaviours that downstream code relies on:

* round-trip: ``save`` -> ``load`` returns the same fields
* unknown keys are preserved (forward-compat)
* ``is_complete`` is honest for both Ollama and API-key providers
* the file path honours the ``PRAGMA_USER_CONFIG`` env override
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from pragma import user_config as uc


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    cfg = uc.UserConfig(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key="sk-ant-secret",
        base_url=None,
    )
    uc.save(cfg, path=p)

    loaded = uc.load(path=p)
    assert loaded.provider == "anthropic"
    assert loaded.model == "claude-haiku-4-5-20251001"
    assert loaded.api_key == "sk-ant-secret"
    assert loaded.base_url is None


def test_save_drops_none_values_for_compactness(tmp_path: Path) -> None:
    """The on-disk JSON should not contain ``"api_key": null`` for the
    Ollama path -- it makes the file noisy and would falsely suggest a
    key is needed."""
    p = tmp_path / "config.json"
    cfg = uc.UserConfig(provider="ollama", model="llama3.2:3b")
    uc.save(cfg, path=p)
    raw = json.loads(p.read_text())
    assert raw == {"provider": "ollama", "model": "llama3.2:3b"}
    assert "api_key" not in raw
    assert "base_url" not in raw


def test_unknown_keys_are_preserved_across_load_save(tmp_path: Path) -> None:
    """If a future pragma version adds a field, an older pragma must
    not silently drop it on round-trip."""
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-test",
                "future_field": {"a": 1},
            }
        )
    )
    cfg = uc.load(path=p)
    assert cfg.extra == {"future_field": {"a": 1}}
    uc.save(cfg, path=p)
    raw = json.loads(p.read_text())
    assert raw["future_field"] == {"a": 1}


def test_load_missing_file_returns_empty_config(tmp_path: Path) -> None:
    cfg = uc.load(path=tmp_path / "does-not-exist.json")
    assert not cfg.is_complete()
    assert cfg.provider is None


def test_load_corrupt_file_returns_empty_config_not_raises(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text("{not valid json}")
    # Must NOT raise -- a single corrupt config should not brick every
    # subsequent pragma command.
    cfg = uc.load(path=p)
    assert cfg.provider is None


def test_is_complete_for_ollama_does_not_require_api_key() -> None:
    cfg = uc.UserConfig(provider="ollama", model="mistral")
    assert cfg.is_complete()


def test_is_complete_for_api_provider_requires_api_key() -> None:
    cfg = uc.UserConfig(provider="openai", model="gpt-4o-mini")
    assert not cfg.is_complete()
    cfg.api_key = "sk-x"
    assert cfg.is_complete()


def test_path_honours_env_override(tmp_path: Path, monkeypatch) -> None:
    custom = tmp_path / "elsewhere" / "cfg.json"
    monkeypatch.setenv("PRAGMA_USER_CONFIG", str(custom))
    assert uc.user_config_path() == custom


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_save_sets_owner_only_perms(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    uc.save(uc.UserConfig(provider="ollama", model="m"), path=p)
    mode = stat.S_IMODE(os.stat(p).st_mode)
    # Owner read+write, no group/other access.
    assert mode & 0o077 == 0, f"unexpectedly broad perms: {oct(mode)}"


def test_clear_removes_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    uc.save(uc.UserConfig(provider="ollama", model="m"), path=p)
    assert uc.clear(path=p) is True
    assert not p.exists()
    # Second clear is a no-op, not an error.
    assert uc.clear(path=p) is False
