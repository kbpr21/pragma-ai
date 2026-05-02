"""``pragma connect`` interactive setup wizard.

Walks a first-time user from "I just installed pragma" to a fully
configured local config in three prompts:

1. Pick a provider from a numbered menu (Ollama / OpenAI / Anthropic /
   Groq / Inception).
2. For Ollama: confirm the URL and pick from the locally-installed
   models. For everyone else: paste the API key (input hidden) and we
   call the provider's ``/v1/models`` endpoint to verify the key AND
   discover the available models in one round-trip.
3. Pick a model from the verified list.

The wizard writes the result to ``pragma.user_config.user_config_path()``
which subsequent commands automatically pick up via
:func:`pragma.cli.main.get_llm`.

Design notes
------------

* **Idempotent**. Re-running ``pragma connect`` simply overwrites the
  file. ``--reset`` deletes it.
* **Fully testable**. The wizard is structured around small
  ``_choose_*`` helpers that take an ``input_func`` and a console -- so
  unit tests can drive it deterministically without touching stdin or a
  real network.
* **No new dependencies**. Uses :mod:`getpass` for hidden input and the
  existing :mod:`rich` console -- no questionary, no inquirer.
"""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pragma.exceptions import LLMError
from pragma.llm.anthropic import AnthropicProvider
from pragma.llm.groq import GroqProvider
from pragma.llm.inception import InceptionProvider
from pragma.llm.ollama import OllamaProvider
from pragma.llm.openai import OpenAIProvider
from pragma.user_config import UserConfig, save, user_config_path


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderInfo:
    """Static metadata used by the wizard to render menus and dispatch
    provider-specific list_models calls.

    We store the *class* rather than a bound classmethod reference so
    that test-time ``patch.object(OpenAIProvider, "list_models", ...)``
    is honoured. Bound references are captured at module import time
    and would otherwise route around the patch.
    """

    key: str  # the value we persist in user_config.provider
    label: str  # human-friendly menu name
    needs_api_key: bool
    default_base_url: str
    api_key_help: str  # one-line hint shown before the API-key prompt
    provider_cls: type

    def list_models(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Indirect dispatch -- resolves ``list_models`` on the class
        every call, so monkeypatching works naturally."""
        return self.provider_cls.list_models(**kwargs)


PROVIDERS: Sequence[ProviderInfo] = (
    ProviderInfo(
        key="ollama",
        label="Ollama (local, no API key)",
        needs_api_key=False,
        default_base_url="http://localhost:11434",
        api_key_help="",
        provider_cls=OllamaProvider,
    ),
    ProviderInfo(
        key="openai",
        label="OpenAI",
        needs_api_key=True,
        default_base_url="https://api.openai.com/v1",
        api_key_help="Get a key from https://platform.openai.com/api-keys",
        provider_cls=OpenAIProvider,
    ),
    ProviderInfo(
        key="anthropic",
        label="Anthropic (Claude)",
        needs_api_key=True,
        default_base_url="https://api.anthropic.com/v1",
        api_key_help="Get a key from https://console.anthropic.com/settings/keys",
        provider_cls=AnthropicProvider,
    ),
    ProviderInfo(
        key="groq",
        label="Groq",
        needs_api_key=True,
        default_base_url="https://api.groq.com/openai/v1",
        api_key_help="Get a key from https://console.groq.com/keys",
        provider_cls=GroqProvider,
    ),
    ProviderInfo(
        key="inception",
        label="Inception (Mercury)",
        needs_api_key=True,
        default_base_url="https://api.inceptionlabs.ai/v1",
        api_key_help="Get a key from https://platform.inceptionlabs.ai/keys",
        provider_cls=InceptionProvider,
    ),
)


# ---------------------------------------------------------------------------
# Small input helpers (kept tiny + injectable for testing)
# ---------------------------------------------------------------------------


PromptFn = Callable[[str], str]
SecretFn = Callable[[str], str]


def _default_prompt(prompt: str) -> str:  # pragma: no cover - thin shim
    return input(prompt)


def _default_secret(prompt: str) -> str:  # pragma: no cover - thin shim
    return getpass.getpass(prompt)


def _ask_choice(
    console: Console,
    prompt: str,
    options: Sequence[str],
    *,
    default: int = 1,
    input_func: PromptFn,
) -> int:
    """Render a numbered menu and return the 1-based choice.

    Re-prompts on invalid input rather than crashing so a fat-fingered
    keystroke does not throw the user back to the shell.
    """
    while True:
        console.print(prompt)
        for i, opt in enumerate(options, 1):
            marker = " *" if i == default else "  "
            console.print(f"  {i}){marker} {opt}")
        raw = input_func(f"Choice [{default}]: ").strip()
        if not raw:
            return default
        try:
            n = int(raw)
        except ValueError:
            console.print(f"[yellow]Not a number: {raw!r}. Try again.[/yellow]")
            continue
        if 1 <= n <= len(options):
            return n
        console.print(f"[yellow]Choose 1-{len(options)} (got {n}). Try again.[/yellow]")


# ---------------------------------------------------------------------------
# Step functions (each one returns the chosen value, or raises typer.Exit)
# ---------------------------------------------------------------------------


def _choose_provider(console: Console, *, input_func: PromptFn) -> ProviderInfo:
    labels = [p.label for p in PROVIDERS]
    idx = _ask_choice(
        console,
        "[bold]Select an LLM provider:[/bold]",
        labels,
        default=1,
        input_func=input_func,
    )
    return PROVIDERS[idx - 1]


def _choose_ollama_model(
    info: ProviderInfo,
    console: Console,
    *,
    input_func: PromptFn,
    base_url: Optional[str] = None,
) -> str:
    url = base_url or info.default_base_url
    console.print(f"[dim]\u2192 Detecting Ollama at {url}...[/dim]")
    try:
        models = info.list_models(base_url=url)
    except LLMError as e:
        console.print(f"[red]\u2717 {e}[/red]")
        raise typer.Exit(code=2)
    if not models:
        console.print(
            "[yellow]No Ollama models found locally.[/yellow]\n"
            "Install one first, e.g.:  [cyan]ollama pull llama3.2[/cyan]"
        )
        raise typer.Exit(code=2)

    labels = []
    for m in models:
        name = m.get("name") or m.get("model") or "?"
        size_gb = (m.get("size") or 0) / (1024**3)
        labels.append(f"{name}  [dim]({size_gb:.1f} GB)[/dim]" if size_gb else name)

    idx = _ask_choice(
        console,
        f"[bold]Available Ollama models ({len(models)}):[/bold]",
        labels,
        default=1,
        input_func=input_func,
    )
    return models[idx - 1].get("name") or models[idx - 1].get("model") or ""


def _choose_api_model(
    info: ProviderInfo,
    api_key: str,
    console: Console,
    *,
    input_func: PromptFn,
    base_url: Optional[str] = None,
) -> str:
    url = base_url or info.default_base_url
    console.print(f"[dim]\u2192 Verifying key against {url}/models ...[/dim]")
    try:
        models = info.list_models(api_key=api_key, base_url=url)
    except LLMError as e:
        console.print(f"[red]\u2717 {e}[/red]")
        raise typer.Exit(code=2)
    if not models:
        console.print(
            f"[yellow]The {info.label} API returned no models. "
            "Check that your account has access.[/yellow]"
        )
        raise typer.Exit(code=2)

    console.print(f"[green]\u2713 Key valid. {len(models)} models available.[/green]")
    labels = [_render_model_label(m) for m in models]
    idx = _ask_choice(
        console,
        f"[bold]Pick a {info.label} model:[/bold]",
        labels,
        default=1,
        input_func=input_func,
    )
    return models[idx - 1].get("id") or ""


def _render_model_label(m: Dict[str, Any]) -> str:
    """Human-friendly label for a /v1/models entry. Falls back to id."""
    mid = m.get("id", "")
    display = m.get("display_name")
    if display and display != mid:
        return f"{mid}  [dim]({display})[/dim]"
    return mid


# ---------------------------------------------------------------------------
# Public command (registered in pragma/cli/main.py)
# ---------------------------------------------------------------------------


def run_connect(
    *,
    console: Optional[Console] = None,
    input_func: Optional[PromptFn] = None,
    secret_func: Optional[SecretFn] = None,
    reset: bool = False,
) -> Optional[UserConfig]:
    """Run the interactive wizard end-to-end.

    All I/O dependencies are injectable so tests can drive the wizard
    without touching stdin, the network, or the real user-config file.
    Returns the saved :class:`UserConfig` on success, ``None`` when the
    user passed ``--reset`` (in which case the config file is deleted).
    """
    console = console or Console()
    input_func = input_func or _default_prompt
    secret_func = secret_func or _default_secret

    if reset:
        from pragma.user_config import clear

        cleared = clear()
        if cleared:
            console.print(
                f"[green]Cleared user config at {user_config_path()}.[/green]"
            )
        else:
            console.print("[yellow]No user config to clear.[/yellow]")
        return None

    console.print(
        Panel.fit(
            "[bold]pragma connect[/bold]\n\n"
            "Pick an LLM provider, paste an API key (or use local Ollama),\n"
            "and pragma will discover the available models for you.\n"
            "Settings are saved to a private local config file.",
            border_style="cyan",
        )
    )

    info = _choose_provider(console, input_func=input_func)

    api_key: Optional[str] = None
    if info.needs_api_key:
        if info.api_key_help:
            console.print(f"[dim]{info.api_key_help}[/dim]")
        api_key = secret_func(f"Enter {info.label} API key (input hidden): ").strip()
        if not api_key:
            console.print("[red]No API key entered. Aborting.[/red]")
            raise typer.Exit(code=1)
        model = _choose_api_model(info, api_key, console, input_func=input_func)
    else:
        model = _choose_ollama_model(info, console, input_func=input_func)

    if not model:
        console.print("[red]Could not determine a model. Aborting.[/red]")
        raise typer.Exit(code=1)

    cfg = UserConfig(
        provider=info.key,
        model=model,
        api_key=api_key,
        base_url=None,  # default; user can edit the file later if needed
    )
    path = save(cfg)

    table = Table(show_header=False, border_style="green")
    table.add_column(style="cyan")
    table.add_column(style="white")
    table.add_row("Provider", info.label)
    table.add_row("Model", model)
    table.add_row("Config file", str(path))
    console.print(
        Panel(table, title="[bold green]Saved[/bold green]", border_style="green")
    )
    console.print(
        "Now try:  [bold cyan]pragma ingest <path>[/bold cyan]  "
        'then  [bold cyan]pragma query "..."[/bold cyan]'
    )
    return cfg
