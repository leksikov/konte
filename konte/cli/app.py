"""Main CLI application using Typer + Rich."""

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn, cast, get_args

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from konte import (
    __version__,
    create_project,
    delete_project,
    get_project,
    list_projects,
    project_exists,
    settings,
    trust_project,
)
from konte.domain.models import MetadataFilter, RetrievalMode

app = typer.Typer(
    name="konte",
    help="Contextual RAG CLI - Build and query document indexes with LLM-generated context.",
    no_args_is_help=True,
)
console = Console()

_RETRIEVAL_MODES: tuple[str, ...] = get_args(RetrievalMode)
_MODE_HELP = f"Retrieval mode: {', '.join(_RETRIEVAL_MODES)}"

_STORAGE_OPTION = typer.Option(
    None,
    "--storage",
    "-s",
    help="Storage path (default: ~/.konte)",
)

_KEYWORD_EXTRACTION_OPTION = typer.Option(
    None,
    "--keyword-extraction/--no-keyword-extraction",
    help=(
        "Reduce the query to keywords before BM25 search, at the cost of one "
        "LLM call (default: BM25_KEYWORD_EXTRACTION)"
    ),
)


def _fail(message: str) -> NoReturn:
    """Report a problem on stdout and exit with status 1."""
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


@contextmanager
def _reporting_errors() -> Iterator[None]:
    """Turn an unexpected failure into a one-line message instead of a traceback.

    typer.Exit derives from Exception, so it is re-raised untouched; without
    that a command which already reported a problem and exited would have its
    exit code printed as a second, meaningless error.
    """
    try:
        yield
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


def _resolve_storage(storage_path: Path | None) -> Path:
    """Return the storage root, falling back to the configured default."""
    return storage_path or settings.STORAGE_PATH


def _require_project(name: str, storage_path: Path) -> None:
    """Exit with an error unless the named project exists."""
    if not project_exists(name, storage_path=storage_path):
        _fail(f"Project '{name}' not found")


def _require_mode(mode: str) -> RetrievalMode:
    """Validate a --mode value and narrow it to the retrieval mode type."""
    if mode not in _RETRIEVAL_MODES:
        _fail(f"Invalid mode: {mode}")
    return cast(RetrievalMode, mode)


def _parse_metadata_filter(filter_json: str | None) -> MetadataFilter | None:
    """Parse the --filter argument, exiting with an error on malformed JSON."""
    if not filter_json:
        return None
    try:
        return json.loads(filter_json)
    except json.JSONDecodeError as e:
        _fail(f"Invalid JSON filter: {e}")


@contextmanager
def _spinner(description: str) -> Iterator[None]:
    """Show an indeterminate progress spinner for a long-running step."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description, total=None)
        yield


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"konte {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    # App help text lives in the typer.Typer(help=...) constructor above
    pass


@app.command("create")
def create(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Path | None = _STORAGE_OPTION,
    prompt: Path | None = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Path to custom context generation prompt file",
    ),
) -> None:
    """Create a new project."""
    path = _resolve_storage(storage_path)

    if project_exists(name, storage_path=path):
        _fail(f"Project '{name}' already exists")
    if prompt is not None and not prompt.exists():
        _fail(f"Prompt file not found: {prompt}")

    with _reporting_errors():
        overrides = {"context_prompt_path": prompt} if prompt is not None else {}
        project = create_project(name, storage_path=path, **overrides)
        project.save()

        console.print(f"[green]Created project:[/green] {name}")
        console.print(f"  Path: {project.project_dir}")
        if prompt:
            console.print(f"  Prompt: {prompt}")


@app.command("list")
def list_cmd(
    storage_path: Path | None = _STORAGE_OPTION,
) -> None:
    """List all projects."""
    path = _resolve_storage(storage_path)
    projects = list_projects(storage_path=path)

    if not projects:
        console.print("[dim]No projects found[/dim]")
        return

    table = Table(title="Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")

    for proj_name in projects:
        table.add_row(proj_name, str(path / proj_name))

    console.print(table)


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Path | None = _STORAGE_OPTION,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """Delete a project."""
    path = _resolve_storage(storage_path)
    _require_project(name, path)

    if not force and not typer.confirm(f"Delete project '{name}'?"):
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit(0)

    with _reporting_errors():
        delete_project(name, storage_path=path)
        console.print(f"[green]Deleted project:[/green] {name}")


@app.command("trust")
def trust(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Path | None = _STORAGE_OPTION,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """Record the indexes a project already has, so they will be loaded.

    Only indexes with a matching record are read. Record one you did not build
    yourself only when you know where it came from; otherwise build again.
    """
    path = _resolve_storage(storage_path)
    _require_project(name, path)

    if not force and not typer.confirm(f"Trust the index files in {path / name}?"):
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit(0)

    with _reporting_errors():
        recorded = trust_project(name, storage_path=path)

        if not recorded:
            console.print(f"[yellow]No index files to record in[/yellow] {path / name}")
            return

        anchor = settings.INDEX_MANIFEST or "this installation's signing key"
        console.print(
            f"[green]Recorded {len(recorded)} index file(s) against {anchor}:[/green] "
            f"{', '.join(recorded)}"
        )


@app.command("add")
def add(
    name: str = typer.Argument(..., help="Project name"),
    files: list[Path] = typer.Argument(..., help="Document files to add"),
    storage_path: Path | None = _STORAGE_OPTION,
) -> None:
    """Add documents to a project."""
    path = _resolve_storage(storage_path)
    _require_project(name, path)

    for file_path in files:
        if not file_path.exists():
            _fail(f"File not found: {file_path}")

    with _reporting_errors():
        project = get_project(name, storage_path=path)

        with _spinner("Adding documents..."):
            num_chunks = project.add_documents(files)

        project.save()
        console.print(f"[green]Added {len(files)} document(s)[/green]")
        console.print(f"  Total chunks: {num_chunks}")


@app.command("build")
def build(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Path | None = _STORAGE_OPTION,
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
    prompt: Path | None = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Path to custom context generation prompt file (overrides project config)",
    ),
) -> None:
    """Build indexes for a project."""
    path = _resolve_storage(storage_path)
    _require_project(name, path)

    if prompt is not None and not prompt.exists():
        _fail(f"Prompt file not found: {prompt}")

    enable_faiss = not bm25_only
    enable_bm25 = not faiss_only

    with _reporting_errors():
        project = get_project(name, storage_path=path)

        with _spinner("Building indexes..."):
            asyncio.run(
                project.build(
                    skip_context=skip_context,
                    enable_faiss=enable_faiss,
                    enable_bm25=enable_bm25,
                    prompt_path=prompt,
                )
            )

        project.save()
        console.print("[green]Build complete[/green]")
        if not skip_context:
            console.print("  Context generation: enabled")
            if prompt:
                console.print(f"  Prompt: {prompt}")
        console.print(f"  FAISS index: {'enabled' if enable_faiss else 'disabled'}")
        console.print(f"  BM25 index: {'enabled' if enable_bm25 else 'disabled'}")


@app.command("query")
def query(
    name: str = typer.Argument(..., help="Project name"),
    query_text: str = typer.Argument(..., help="Query text"),
    storage_path: Path | None = _STORAGE_OPTION,
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
        help=_MODE_HELP,
    ),
    filter_json: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help='Metadata filter as JSON, e.g. \'{"source": "doc.pdf", "year": 2024}\'',
    ),
    keyword_extraction: bool | None = _KEYWORD_EXTRACTION_OPTION,
) -> None:
    """Query a project."""
    path = _resolve_storage(storage_path)
    _require_project(name, path)

    retrieval_mode = _require_mode(mode)
    metadata_filter = _parse_metadata_filter(filter_json)

    with _reporting_errors():
        project = get_project(name, storage_path=path)
        response = project.query(
            query_text,
            mode=retrieval_mode,
            top_k=top_k,
            use_keyword_extraction=keyword_extraction,
            metadata_filter=metadata_filter,
        )

        console.print(f"\n[bold]Query:[/bold] {query_text}")
        console.print(f"[bold]Mode:[/bold] {mode}")
        if metadata_filter:
            console.print(f"[bold]Filter:[/bold] {metadata_filter}")
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


@app.command("ask")
def ask(
    name: str = typer.Argument(..., help="Project name"),
    question: str = typer.Argument(..., help="Question to answer"),
    storage_path: Path | None = _STORAGE_OPTION,
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of chunks to retrieve",
    ),
    max_chunks: int = typer.Option(
        10,
        "--max-chunks",
        help="Maximum chunks to use for answer generation",
    ),
    mode: str = typer.Option(
        "hybrid",
        "--mode",
        "-m",
        help=_MODE_HELP,
    ),
    filter_json: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help='Metadata filter as JSON, e.g. \'{"company": "ACME", "year": 2024}\'',
    ),
    keyword_extraction: bool | None = _KEYWORD_EXTRACTION_OPTION,
    show_sources: bool = typer.Option(
        False,
        "--show-sources",
        help="Show retrieved sources",
    ),
) -> None:
    """Ask a question and get an LLM-generated answer (full RAG pipeline)."""
    path = _resolve_storage(storage_path)
    _require_project(name, path)

    retrieval_mode = _require_mode(mode)
    metadata_filter = _parse_metadata_filter(filter_json)

    with _reporting_errors():
        project = get_project(name, storage_path=path)

        with _spinner("Generating answer..."):
            retrieval_response, answer = asyncio.run(
                project.query_with_answer(
                    query=question,
                    mode=retrieval_mode,
                    top_k=top_k,
                    max_chunks=max_chunks,
                    use_keyword_extraction=keyword_extraction,
                    metadata_filter=metadata_filter,
                )
            )

        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")
        if metadata_filter:
            console.print(f"[bold]Filter:[/bold] {metadata_filter}")
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(answer.answer)
        console.print(f"\n[dim]Model: {answer.model} | Sources used: {answer.sources_used}[/dim]")

        if show_sources:
            console.print(f"\n[bold]Retrieved Sources ({retrieval_response.total_found}):[/bold]")
            for i, result in enumerate(retrieval_response.results[:max_chunks], 1):
                console.print(f"\n[cyan]--- Source {i} (score: {result.score:.3f}) ---[/cyan]")
                console.print(f"[dim]File:[/dim] {result.source}")
                console.print(f"{result.content[:300]}...")


@app.command("serve")
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind to",
    ),
) -> None:
    """Start the FastAPI server."""
    try:
        from konte.api import run_server
    except ImportError:
        console.print("[red]Error:[/red] FastAPI not installed. Run: pip install konte[api]")
        raise typer.Exit(1) from None

    console.print(f"Starting API server at http://{host}:{port}")
    run_server(host=host, port=port)


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Project name"),
    storage_path: Path | None = _STORAGE_OPTION,
) -> None:
    """Show project information."""
    path = _resolve_storage(storage_path)
    _require_project(name, path)

    with _reporting_errors():
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
        table.add_row("Context Strategy", config.context_strategy or settings.CONTEXT_STRATEGY)
        table.add_row("FAISS Enabled", str(config.enable_faiss))
        table.add_row("BM25 Enabled", str(config.enable_bm25))

        project_dir = project.project_dir
        table.add_row("", "")
        for label, filename in (
            ("FAISS Index", "faiss.faiss"),
            ("BM25 Index", "bm25.pkl"),
            ("Chunks", "chunks.json"),
        ):
            exists = (project_dir / filename).exists()
            table.add_row(label, "[green]exists[/green]" if exists else "[dim]not built[/dim]")

        console.print(table)


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
