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

app = typer.Typer(
    name="pragma",
    help="Atomic fact reasoning. No vector database.",
    add_completion=False,
)

console = Console()


def get_llm():
    """Get LLM provider from environment or config."""
    if os.environ.get("INCEPTION_API_KEY"):
        return get_provider("inception")
    if os.environ.get("OPENAI_API_KEY"):
        return get_provider("openai")
    if os.environ.get("GROQ_API_KEY"):
        return get_provider("groq")

    return None


def require_llm():
    """Get LLM or raise error."""
    llm = get_llm()
    if llm is None:
        raise LLMError(
            "No LLM API key found. Set INCEPTION_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY"
        )
    return llm


def get_kb() -> KnowledgeBase:
    """Get KnowledgeBase instance."""
    config = PragmaConfig.default()
    llm = require_llm()
    return KnowledgeBase(llm=llm, kb_dir=config.kb_dir)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file, directory, or URL to ingest"),
    show_progress: bool = typer.Option(False, "--progress", help="Show progress bar"),
):
    """Ingest documents into the knowledge base."""
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
