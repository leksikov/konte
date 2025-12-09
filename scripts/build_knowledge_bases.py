"""Build knowledge base projects from example_knowledge_base/ directory."""

import asyncio
import os
from pathlib import Path

# Set BackendAI endpoint for context generation
os.environ["BACKENDAI_ENDPOINT"] = "https://qwen3vl.asia03.app.backend.ai/v1"
os.environ["BACKENDAI_MODEL_NAME"] = "Qwen3-VL-8B-Instruct"

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from konte import Project, project_exists, delete_project

console = Console()

# Files to process (excluding gri_rules.md)
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "example_knowledge_base"
EXCLUDE_FILES = {"gri_rules.md"}


def get_project_name(file_path: Path) -> str:
    """Generate project name from file path."""
    name = file_path.stem
    # Clean up name for project naming
    name = name.replace(" ", "_").replace("[", "").replace("]", "")
    # Truncate if too long
    if len(name) > 50:
        name = name[:50]
    return name


async def build_project(file_path: Path, skip_context: bool = False) -> bool:
    """Build a single project from a file.

    Args:
        file_path: Path to the document file.
        skip_context: Skip LLM context generation for speed.

    Returns:
        True if successful, False otherwise.
    """
    project_name = get_project_name(file_path)

    console.print(f"\n[cyan]Building project:[/cyan] {project_name}")
    console.print(f"  Source: {file_path.name}")

    try:
        # Delete existing project if exists
        if project_exists(project_name):
            console.print(f"  [dim]Deleting existing project...[/dim]")
            delete_project(project_name)

        # Create project
        project = Project.create(project_name)

        # Add document
        console.print(f"  [dim]Adding document...[/dim]")
        num_chunks = project.add_documents([file_path])
        console.print(f"  Chunks created: {num_chunks}")

        # Build indexes
        console.print(f"  [dim]Building indexes (skip_context={skip_context})...[/dim]")
        await project.build(skip_context=skip_context)

        # Save
        project.save()
        console.print(f"  [green]✓ Project saved[/green]")

        return True
    except Exception as e:
        console.print(f"  [red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Build all knowledge base projects."""
    console.print("[bold magenta]Building Knowledge Base Projects[/bold magenta]")
    console.print("=" * 60)

    # Get all .md files except excluded ones
    md_files = [
        f for f in KNOWLEDGE_BASE_DIR.glob("*.md")
        if f.name not in EXCLUDE_FILES
    ]

    # Sort by file size (smallest first)
    md_files.sort(key=lambda f: f.stat().st_size)

    console.print(f"\nFound {len(md_files)} files to process (smallest first):")
    for f in md_files:
        size_kb = f.stat().st_size / 1024
        console.print(f"  - {f.name} ({size_kb:.1f} KB)")

    # Build each project (skip if already exists)
    results = {}
    for file_path in md_files:
        project_name = get_project_name(file_path)
        if project_exists(project_name):
            console.print(f"\n[dim]Skipping {project_name} (already exists)[/dim]")
            results[file_path.name] = True
            continue
        results[file_path.name] = await build_project(file_path, skip_context=False)

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Summary:[/bold]")
    success = sum(1 for v in results.values() if v)
    console.print(f"  Successful: {success}/{len(results)}")

    for name, status in results.items():
        icon = "[green]✓[/green]" if status else "[red]✗[/red]"
        console.print(f"  {icon} {name}")


if __name__ == "__main__":
    asyncio.run(main())
