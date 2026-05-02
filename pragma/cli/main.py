import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from pragma import KnowledgeBase
from pragma.config import PragmaConfig
from pragma.llm import get_provider
from pragma.llm.base import LLMError
from pragma import user_config as user_cfg

app = typer.Typer(
    name="pragma",
    help="Atomic fact reasoning. No vector database.",
    add_completion=False,
)

console = Console()


# ---------------------------------------------------------------------------
# LLM resolution
# ---------------------------------------------------------------------------


def get_llm():
    """Build an LLM provider from the user's saved config or env vars.

    Resolution order (first hit wins):
      1. ``~/.pragma/config.json`` written by ``pragma connect`` -- the
         friendliest path because it bundles provider + model + key in
         one place. Honours the ``PRAGMA_USER_CONFIG`` env override.
      2. Provider-specific env vars (``INCEPTION_API_KEY``,
         ``OPENAI_API_KEY``, ``GROQ_API_KEY``, ``ANTHROPIC_API_KEY``).
         Kept for back-compat and CI/server use.

    Returns ``None`` when nothing is configured so callers can decide
    whether to error out (most commands) or just no-op.
    """
    cfg = user_cfg.load()
    if cfg.is_complete():
        kwargs: dict = {}
        if cfg.base_url:
            # Ollama uses BASE_URL as a class attribute, others take a
            # base_url= kwarg. We pass via kwarg where supported and fall
            # back to monkey-patching only for Ollama in the rare case
            # the user configured a non-default URL.
            if cfg.provider != "ollama":
                kwargs["base_url"] = cfg.base_url
        return get_provider(
            cfg.provider,
            api_key=cfg.api_key,
            model=cfg.model,
            **kwargs,
        )

    # Fall through to the legacy env-var path.
    if os.environ.get("INCEPTION_API_KEY"):
        return get_provider("inception")
    if os.environ.get("OPENAI_API_KEY"):
        return get_provider("openai")
    if os.environ.get("GROQ_API_KEY"):
        return get_provider("groq")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return get_provider("anthropic")

    return None


def require_llm():
    """Get LLM or raise a friendly error pointing at ``pragma connect``."""
    llm = get_llm()
    if llm is None:
        raise LLMError(
            "No LLM is configured.\n\n"
            "  Run  [bold cyan]pragma connect[/bold cyan]  to pick a provider "
            "and model interactively, or set one of:\n"
            "    INCEPTION_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / "
            "ANTHROPIC_API_KEY\n"
            "  in your environment."
        )
    return llm


def get_kb() -> KnowledgeBase:
    """Get KnowledgeBase instance, surfacing connect-hint on missing LLM."""
    config = PragmaConfig.default()
    try:
        llm = require_llm()
    except LLMError as e:
        # Render the friendly hint with rich markup intact, then exit
        # cleanly -- a stack trace here would be hostile UX.
        console.print(
            Panel(str(e), title="[red]Setup needed[/red]", border_style="red")
        )
        raise typer.Exit(code=1)
    return KnowledgeBase(llm=llm, kb_dir=config.kb_dir)


# ---------------------------------------------------------------------------
# pragma connect (interactive setup)
# ---------------------------------------------------------------------------


@app.command()
def connect(
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Delete the saved user config and exit.",
    ),
):
    """Interactively configure an LLM provider, key, and model.

    Walks you through picking a provider (Ollama / OpenAI / Anthropic /
    Groq / Inception), pasting an API key (input hidden), discovering
    the available models from the provider's own API, and saving the
    result to a private local config file. Re-run any time to switch
    providers or rotate keys.
    """
    # Lazy import keeps cold-start fast for non-connect commands and
    # avoids pulling typer into the wizard module's import graph.
    from pragma.cli.connect import run_connect

    run_connect(console=console, reset=reset)


@app.command()
def ingest(
    path: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to a file, directory, or URL. "
            "Omit to ingest every supported file in the current directory."
        ),
    ),
    show_progress: bool = typer.Option(False, "--progress", help="Show progress bar"),
):
    """Ingest documents into the knowledge base.

    UX rules of thumb:

    * No argument? Treat the current working directory as the target,
      after confirming with the user. This is the case the v1.0.1 CLI
      handled poorly -- folks dropping a single PDF into a folder and
      typing ``pragma ingest`` should not have to think about paths.
    * A bare filename? Resolve relative to ``cwd`` like every other
      Unix tool already does -- no special-case needed at this layer
      thanks to ``Path``'s implicit-cwd semantics.
    * URL? Pass straight through to :meth:`KnowledgeBase.ingest`,
      which already knows how to fetch + extract.
    """
    if path is None:
        cwd = Path.cwd()
        candidates = _supported_files_in(cwd)
        if not candidates:
            console.print(
                Panel(
                    f"No supported files found in [bold]{cwd}[/bold].\n\n"
                    "Supported types: .txt .md .pdf .csv .json .jsonl .docx .html\n\n"
                    "Pass an explicit path, e.g.  [cyan]pragma ingest "
                    "./paper.pdf[/cyan]",
                    title="[yellow]Nothing to ingest[/yellow]",
                    border_style="yellow",
                )
            )
            raise typer.Exit(code=1)
        preview = "\n  ".join(str(p.relative_to(cwd)) for p in candidates[:8])
        more = f"\n  ... and {len(candidates) - 8} more" if len(candidates) > 8 else ""
        if not typer.confirm(
            f"Ingest {len(candidates)} file(s) from {cwd}?\n  {preview}{more}",
            default=True,
        ):
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)
        path = str(cwd)

    kb = get_kb()

    with console.status("[bold green]Ingesting..."):
        try:
            result = kb.ingest(path, show_progress=show_progress)

            table = Table(title="Ingest Summary", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_row("Documents", str(result.documents))
            table.add_row("Facts", str(result.facts))
            table.add_row("Entities", str(result.entities))
            table.add_row("Skipped", str(result.skipped))

            console.print(table)

        finally:
            kb.close()


_SUPPORTED_INGEST_EXTS = {
    ".txt",
    ".md",
    ".pdf",
    ".csv",
    ".json",
    ".jsonl",
    ".docx",
    ".html",
    ".htm",
}


def _supported_files_in(directory: Path) -> list[Path]:
    """List supported files at the top level of *directory*.

    We deliberately do NOT recurse here -- the user asked us to ingest
    "this folder", not "this folder and everything underneath" -- and
    surprising the user with hundreds of nested files is exactly the
    kind of UX wart this command was rewritten to fix. If they want
    recursion they can pass the directory explicitly to ``kb.ingest``,
    which DOES recurse.
    """
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_INGEST_EXTS
    )


@app.command()
def query(
    query: str = typer.Argument(..., help="Natural language query"),
    hop_depth: Optional[int] = typer.Option(
        None, "--hop-depth", help="Max graph traversal depth"
    ),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Minimum confidence threshold"
    ),
):
    """Query the knowledge base."""
    kb = get_kb()

    with console.status("[bold green]Querying..."):
        try:
            result = kb.query(
                query,
                hop_depth=hop_depth,
                min_confidence=min_confidence,
            )

            console.print(
                Panel(result.answer, title="[bold]Answer[/bold]", border_style="green")
            )

            if result.confidence > 0:
                console.print(f"[dim]Confidence:[/dim] {result.confidence:.2f}")

            if result.reasoning_path:
                table = Table(title="[bold]Reasoning Path[/bold]")
                table.add_column("Fact ID", style="cyan")
                table.add_column("Explanation", style="white")

                for step in result.reasoning_path:
                    table.add_row(step.fact_id[:8], step.explanation)

                console.print(table)

            console.print(f"[dim]Latency:[/dim] {result.latency_ms:.0f}ms")
            console.print(f"[dim]Subgraph size:[/dim] {result.subgraph_size}")

        finally:
            kb.close()


@app.command()
def stats():
    """Show knowledge base statistics."""
    kb = get_kb()

    try:
        kb_stats = kb.stats()

        table = Table(title="[bold]Knowledge Base Statistics[/bold]", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")

        table.add_row("Documents", str(kb_stats.documents))
        table.add_row("Facts", str(kb_stats.facts))
        table.add_row("Entities", str(kb_stats.entities))
        table.add_row("Relationships", str(kb_stats.relationships))

        console.print(table)

    finally:
        kb.close()


@app.command()
def facts(
    entity: str = typer.Option(..., "--entity", help="Entity name to search"),
    limit: int = typer.Option(10, "--limit", help="Maximum facts to show"),
):
    """List facts for an entity."""
    from pragma.storage.sqlite import SQLiteStore

    config = PragmaConfig.default()
    store = SQLiteStore(config.kb_dir)

    try:
        entity_facts = store.get_facts_by_subject(entity)

        if not entity_facts:
            console.print(f"[yellow]No facts found for '{entity}'[/yellow]")
            return

        table = Table(title=f"[bold]Facts for '{entity}'[/bold]")
        table.add_column("Fact ID", style="cyan")
        table.add_column("Predicate", style="magenta")
        table.add_column("Object", style="white")
        table.add_column("Confidence", style="green")

        for fact in entity_facts[:limit]:
            table.add_row(
                fact.id[:8],
                fact.predicate,
                fact.object_value or fact.object_id or "",
                f"{fact.confidence:.2f}",
            )

        console.print(table)
        console.print(f"[dim]Total: {len(entity_facts)} facts[/dim]")

    finally:
        store.close()


@app.command()
def entities(
    limit: int = typer.Option(20, "--limit", help="Maximum entities to show"),
):
    """List all entities."""
    from pragma.storage.sqlite import SQLiteStore

    config = PragmaConfig.default()
    store = SQLiteStore(config.kb_dir)

    try:
        all_entities = store.get_all_entities()

        if not all_entities:
            console.print("[yellow]No entities in knowledge base[/yellow]")
            return

        table = Table(title="[bold]Entities[/bold]")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Aliases", style="white")

        for entity in all_entities[:limit]:
            aliases_str = ", ".join(entity.aliases[:3]) if entity.aliases else ""
            table.add_row(
                entity.name,
                entity.entity_type or "unknown",
                aliases_str,
            )

        console.print(table)
        console.print(f"[dim]Total: {len(all_entities)} entities[/dim]")

    finally:
        store.close()


@app.command()
def config():
    """Show current configuration."""
    config = PragmaConfig.default()

    table = Table(title="[bold]Configuration[/bold]", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("KB Directory", str(config.kb_dir))
    table.add_row("Default Hop Depth", str(config.default_hop_depth))
    table.add_row("Max Subgraph Nodes", str(config.max_subgraph_nodes))
    table.add_row("Max Subquestions", str(config.max_subquestions))
    table.add_row("LLM Provider", str(config.llm_provider))
    table.add_row("LLM Model", str(config.llm_model or "default"))
    table.add_row("Fact Confidence Threshold", str(config.fact_confidence_threshold))

    console.print(table)


@app.command()
def clear():
    """Delete the knowledge base."""
    confirm = typer.confirm("Are you sure you want to delete the knowledge base?")

    if not confirm:
        console.print("[yellow]Aborted[/yellow]")
        return

    config = PragmaConfig.default()
    import shutil

    kb_dir = Path(config.kb_dir)
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
        console.print("[green]Knowledge base deleted[/green]")
    else:
        console.print("[yellow]Knowledge base does not exist[/yellow]")


if __name__ == "__main__":
    app()
