"""Rebuild wco_hs_explanatory_notes project with Korean context.

This script rebuilds the project using the updated Korean context prompt.
"""

import asyncio
import os
from pathlib import Path

# Set BackendAI endpoint for context generation
os.environ["BACKENDAI_ENDPOINT"] = "https://qwen3vl.asia03.app.backend.ai/v1"
os.environ["BACKENDAI_MODEL_NAME"] = "Qwen3-VL-8B-Instruct"

from rich.console import Console

from konte import Project, project_exists, delete_project

console = Console()

SOURCE_FILE = Path(__file__).parent.parent / "example_knowledge_base" / "wco_hs_explanatory_notes.md"
PROJECT_NAME = "wco_hs_explanatory_notes_korean"


async def main():
    """Rebuild wco_hs_explanatory_notes with Korean context."""
    console.print(f"\n[bold cyan]Rebuilding project with Korean context[/bold cyan]")
    console.print(f"  Source: {SOURCE_FILE.name}")
    console.print(f"  Project: {PROJECT_NAME}")

    # Delete existing project if exists
    if project_exists(PROJECT_NAME):
        console.print(f"  [dim]Deleting existing project...[/dim]")
        delete_project(PROJECT_NAME)

    # Create project
    project = Project.create(PROJECT_NAME)

    # Add document
    console.print(f"  [dim]Adding document...[/dim]")
    num_chunks = project.add_documents([SOURCE_FILE])
    console.print(f"  Chunks created: {num_chunks}")

    # Build indexes with context generation
    console.print(f"  [dim]Building indexes with Korean context...[/dim]")
    console.print(f"  [yellow]This will take a while (generating context for {num_chunks} chunks)...[/yellow]")
    await project.build(skip_context=False)

    # Save
    project.save()
    console.print(f"  [green]✓ Project saved to {project.project_dir}[/green]")

    # Verify Korean context
    console.print(f"\n[dim]Verifying Korean context...[/dim]")
    import json
    chunks_path = project.project_dir / "chunks.json"
    with open(chunks_path) as f:
        chunks = json.load(f)

    # Check first few contexts
    korean_count = 0
    for chunk in chunks[:10]:
        context = chunk.get("context", "")
        # Check if contains Korean characters
        if any('\uac00' <= c <= '\ud7a3' for c in context):
            korean_count += 1

    console.print(f"  Korean contexts in first 10: {korean_count}/10")
    if korean_count >= 8:
        console.print(f"  [green]✓ Context generation in Korean working[/green]")
    else:
        console.print(f"  [red]✗ Context may still be in English[/red]")


if __name__ == "__main__":
    asyncio.run(main())
