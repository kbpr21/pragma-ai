"""Tests for the ``pragma connect`` wizard.

The wizard's I/O dependencies are injected (``input_func`` for stdin,
``secret_func`` for hidden API-key input, ``console`` for output) so we
can drive the whole thing deterministically without touching the real
terminal or making network calls.

We also patch the per-provider ``list_models`` callables on the
``PROVIDERS`` registry so each test exercises the wizard's branching
logic without hitting real APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from pragma import user_config as uc
from pragma.cli import connect as connect_mod


def _fake_inputs(*answers: str):
    """Returns a callable that yields the given answers in order; used
    as the wizard's ``input_func`` so each prompt gets a scripted reply."""
    it = iter(answers)

    def _fn(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration as e:
            raise AssertionError(
                f"Wizard asked for more input than the test supplied; prompt={prompt!r}"
            ) from e

    return _fn


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path: Path, monkeypatch):
    """Every test gets its own user-config file so they cannot pollute
    each other or, crucially, the developer's real ``~/.pragma/config.json``."""
    monkeypatch.setenv("PRAGMA_USER_CONFIG", str(tmp_path / "user_config.json"))
    yield


def test_ollama_path_picks_first_local_model(tmp_path: Path) -> None:
    fake_models = [
        {"name": "mistral", "size": 4_100_000_000},
        {"name": "llama3.2:3b", "size": 2_000_000_000},
    ]
    with patch.object(
        connect_mod.OllamaProvider, "list_models", return_value=fake_models
    ):
        cfg = connect_mod.run_connect(
            input_func=_fake_inputs("1", "1"),  # provider=Ollama, model=#1
            secret_func=lambda _p: "",  # never asked
        )
    assert cfg is not None
    assert cfg.provider == "ollama"
    assert cfg.model == "mistral"
    assert cfg.api_key is None
    # Persisted to disk
    persisted = uc.load()
    assert persisted.provider == "ollama"
    assert persisted.model == "mistral"


def test_openai_path_uses_secret_input_and_lists_models(tmp_path: Path) -> None:
    fake_models = [
        {"id": "gpt-4o-mini"},
        {"id": "gpt-5"},
    ]
    secret_calls: List[str] = []

    def fake_secret(prompt: str) -> str:
        secret_calls.append(prompt)
        return "sk-test-1234"

    with patch.object(
        connect_mod.OpenAIProvider, "list_models", return_value=fake_models
    ) as mock_list:
        cfg = connect_mod.run_connect(
            input_func=_fake_inputs("2", "1"),  # provider=OpenAI, model=#1
            secret_func=fake_secret,
        )

    # Secret prompt fired exactly once and the key reached the provider
    assert len(secret_calls) == 1
    assert "OpenAI" in secret_calls[0]
    mock_list.assert_called_once()
    assert mock_list.call_args.kwargs.get("api_key") == "sk-test-1234"

    assert cfg is not None
    assert cfg.provider == "openai"
    assert cfg.model in {"gpt-4o-mini", "gpt-5"}
    assert cfg.api_key == "sk-test-1234"


def test_invalid_choice_then_valid_choice_re_prompts(tmp_path: Path) -> None:
    fake_models = [{"name": "mistral", "size": 0}]
    with patch.object(
        connect_mod.OllamaProvider, "list_models", return_value=fake_models
    ):
        cfg = connect_mod.run_connect(
            # First answer "99" is out-of-range, then "x" is non-numeric,
            # then "1" picks Ollama, then "1" picks the only model.
            input_func=_fake_inputs("99", "x", "1", "1"),
            secret_func=lambda _p: "",
        )
    assert cfg is not None
    assert cfg.provider == "ollama"


def test_empty_api_key_aborts_with_non_zero_exit() -> None:
    import typer

    with pytest.raises(typer.Exit) as excinfo:
        connect_mod.run_connect(
            input_func=_fake_inputs("2"),  # OpenAI
            secret_func=lambda _p: "   ",  # whitespace-only -> empty
        )
    assert excinfo.value.exit_code == 1


def test_reset_clears_existing_config(tmp_path: Path) -> None:
    # Pre-seed a config
    uc.save(uc.UserConfig(provider="ollama", model="mistral"))
    assert uc.load().is_complete()

    out = connect_mod.run_connect(reset=True)
    assert out is None
    assert not uc.load().is_complete()


def test_default_choice_when_user_just_presses_enter(tmp_path: Path) -> None:
    fake_models = [{"name": "mistral"}]
    with patch.object(
        connect_mod.OllamaProvider, "list_models", return_value=fake_models
    ):
        cfg = connect_mod.run_connect(
            input_func=_fake_inputs("", ""),  # accept default for both prompts
            secret_func=lambda _p: "",
        )
    assert cfg is not None
    assert cfg.provider == "ollama"
    assert cfg.model == "mistral"


def test_no_local_ollama_models_aborts() -> None:
    import typer

    with patch.object(connect_mod.OllamaProvider, "list_models", return_value=[]):
        with pytest.raises(typer.Exit):
            connect_mod.run_connect(
                input_func=_fake_inputs("1"),
                secret_func=lambda _p: "",
            )
