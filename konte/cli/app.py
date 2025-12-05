"""Main CLI application using Typer + Rich."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from konte import (
    Project,
    create_project,
    delete_project,
    get_project,
    list_projects,
    project_exists,
    settings,
)

app = typer.Typer(
    name="konte",
    help="Contextual RAG CLI - Build and query document indexes with LLM-generated context.",
    no_args_is_help=True,
)
console = Console()


@app.command("create")
def create(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage path (default: ~/.konte)",
    ),
) -> None:
    """Create a new project."""
    try:
        path = storage_path or settings.STORAGE_PATH
        if project_exists(name, storage_path=path):
            console.print(f"[red]Error:[/red] Project '{name}' already exists")
            raise typer.Exit(1)

        project = create_project(name, storage_path=path)
        project.save()
        console.print(f"[green]Created project:[/green] {name}")
        console.print(f"  Path: {project.project_dir}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_cmd(
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage path (default: ~/.konte)",
    ),
) -> None:
    """List all projects."""
    path = storage_path or settings.STORAGE_PATH
    projects = list_projects(storage_path=path)

    if not projects:
        console.print("[dim]No projects found[/dim]")
        return

    table = Table(title="Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")

    for proj_name in projects:
        proj_path = path / proj_name
        table.add_row(proj_name, str(proj_path))

    console.print(table)


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage path (default: ~/.konte)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """Delete a project."""
    path = storage_path or settings.STORAGE_PATH

    if not project_exists(name, storage_path=path):
        console.print(f"[red]Error:[/red] Project '{name}' not found")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete project '{name}'?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    try:
        delete_project(name, storage_path=path)
        console.print(f"[green]Deleted project:[/green] {name}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("add")
def add(
    name: str = typer.Argument(..., help="Project name"),
    files: list[Path] = typer.Argument(..., help="Document files to add"),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage path (default: ~/.konte)",
    ),
) -> None:
    """Add documents to a project."""
    path = storage_path or settings.STORAGE_PATH

    if not project_exists(name, storage_path=path):
        console.print(f"[red]Error:[/red] Project '{name}' not found")
        raise typer.Exit(1)

    # Validate files exist
    for f in files:
        if not f.exists():
            console.print(f"[red]Error:[/red] File not found: {f}")
            raise typer.Exit(1)

    try:
        project = get_project(name, storage_path=path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Adding documents...", total=None)
            num_chunks = project.add_documents(files)

        project.save()
        console.print(f"[green]Added {len(files)} document(s)[/green]")
        console.print(f"  Total chunks: {num_chunks}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("build")
def build(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage path (default: ~/.konte)",
    ),
    skip_context: bool = typer.Option(
        False,
        "--skip-context",
        help="Skip LLM context generation (standard RAG)",
    ),
    faiss_only: bool = typer.Option(
        False,
        "--faiss-only",
        help="Build only FAISS index",
    ),
    bm25_only: bool = typer.Option(
        False,
        "--bm25-only",
        help="Build only BM25 index",
    ),
) -> None:
    """Build indexes for a project."""
    path = storage_path or settings.STORAGE_PATH

    if not project_exists(name, storage_path=path):
        console.print(f"[red]Error:[/red] Project '{name}' not found")
        raise typer.Exit(1)

    try:
        project = get_project(name, storage_path=path)

        enable_faiss = not bm25_only
        enable_bm25 = not faiss_only

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building indexes...", total=None)

            async def run_build():
                await project.build(
                    skip_context=skip_context,
                    enable_faiss=enable_faiss,
                    enable_bm25=enable_bm25,
                )

            asyncio.run(run_build())

        project.save()
        console.print("[green]Build complete[/green]")
        if not skip_context:
            console.print("  Context generation: enabled")
        console.print(f"  FAISS index: {'enabled' if enable_faiss else 'disabled'}")
        console.print(f"  BM25 index: {'enabled' if enable_bm25 else 'disabled'}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("query")
def query(
    name: str = typer.Argument(..., help="Project name"),
    query_text: str = typer.Argument(..., help="Query text"),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage path (default: ~/.konte)",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return",
    ),
    mode: str = typer.Option(
        "hybrid",
        "--mode",
        "-m",
        help="Retrieval mode: hybrid, semantic, lexical",
    ),
) -> None:
    """Query a project."""
    path = storage_path or settings.STORAGE_PATH

    if not project_exists(name, storage_path=path):
        console.print(f"[red]Error:[/red] Project '{name}' not found")
        raise typer.Exit(1)

    if mode not in ("hybrid", "semantic", "lexical"):
        console.print(f"[red]Error:[/red] Invalid mode: {mode}")
        raise typer.Exit(1)

    try:
        project = get_project(name, storage_path=path)
        response = project.query(query_text, mode=mode, top_k=top_k)

        console.print(f"\n[bold]Query:[/bold] {query_text}")
        console.print(f"[bold]Mode:[/bold] {mode}")
        console.print(f"[bold]Results:[/bold] {response.total_found}")
        console.print(f"[bold]Top Score:[/bold] {response.top_score:.3f}")
        console.print(f"[bold]Suggested Action:[/bold] {response.suggested_action}")
        console.print()

        for i, result in enumerate(response.results, 1):
            console.print(f"[cyan]--- Result {i} (score: {result.score:.3f}) ---[/cyan]")
            console.print(f"[dim]Source:[/dim] {result.source}")
            if result.context:
                console.print(f"[dim]Context:[/dim] {result.context[:200]}...")
            console.print(f"\n{result.content[:500]}...")
            console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Storage path (default: ~/.konte)",
    ),
) -> None:
    """Show project information."""
    path = storage_path or settings.STORAGE_PATH

    if not project_exists(name, storage_path=path):
        console.print(f"[red]Error:[/red] Project '{name}' not found")
        raise typer.Exit(1)

    try:
        project = get_project(name, storage_path=path)
        config = project.config

        table = Table(title=f"Project: {name}")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        table.add_row("Name", config.name)
        table.add_row("Storage Path", str(config.storage_path))
        table.add_row("Segment Size", f"{config.segment_size} tokens")
        table.add_row("Chunk Size", f"{config.chunk_size} tokens")
        table.add_row("Embedding Model", config.embedding_model)
        table.add_row("Context Model", config.context_model)
        table.add_row("FAISS Enabled", str(config.enable_faiss))
        table.add_row("BM25 Enabled", str(config.enable_bm25))

        # Check for index files
        project_dir = project.project_dir
        has_faiss = (project_dir / "faiss.faiss").exists()
        has_bm25 = (project_dir / "bm25.pkl").exists()
        has_chunks = (project_dir / "chunks.json").exists()

        table.add_row("", "")
        table.add_row("FAISS Index", "[green]exists[/green]" if has_faiss else "[dim]not built[/dim]")
        table.add_row("BM25 Index", "[green]exists[/green]" if has_bm25 else "[dim]not built[/dim]")
        table.add_row("Chunks", "[green]exists[/green]" if has_chunks else "[dim]not built[/dim]")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
