#!/usr/bin/env python
"""Build WCO Korean project (February 2026) with Qwen3-30B."""

import asyncio
import os
from pathlib import Path

# Set BackendAI endpoint BEFORE importing konte
os.environ["BACKENDAI_ENDPOINT"] = "https://qwen_scaled.asia03.app.backend.ai/v1"
os.environ["BACKENDAI_MODEL_NAME"] = "Qwen3-30B-A3B-Instruct-2507"

from rich.console import Console
from konte import Project, delete_project, project_exists

console = Console()

PROJECT_NAME = "wco_korean_feb2026"
DOCS_DIR = Path(__file__).parent.parent / "example_knowledge_base"
DOCUMENT = DOCS_DIR / "wco_hs_explanatory_notes.md"


async def main():
    console.print(f"[bold blue]Building {PROJECT_NAME}[/bold blue]")
    console.print(f"Endpoint: {os.environ['BACKENDAI_ENDPOINT']}")
    console.print(f"Model: {os.environ['BACKENDAI_MODEL_NAME']}")

    # Delete existing project if exists
    if project_exists(PROJECT_NAME):
        console.print(f"[yellow]Deleting existing {PROJECT_NAME}...[/yellow]")
        delete_project(PROJECT_NAME)

    # Create new project
    project = Project.create(PROJECT_NAME)

    # Add document
    console.print(f"Adding: {DOCUMENT.name}")
    num_chunks = project.add_documents([DOCUMENT])
    console.print(f"  → {num_chunks} chunks")

    # Build with context generation
    console.print("[bold]Building indexes with context generation...[/bold]")
    await project.build(skip_context=False)
    project.save()

    # Validate Korean context
    console.print("\n[bold]Validating Korean context...[/bold]")
    response = project.query("관세율 분류", top_k=5)
    for i, result in enumerate(response.results[:3]):
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in result.context)
        status = "✓" if has_korean else "✗"
        console.print(f"  {status} Chunk {i+1}: Korean context {'detected' if has_korean else 'MISSING'}")

    console.print(f"\n[bold green]✓ {PROJECT_NAME} built successfully[/bold green]")
    console.print(f"  Total chunks: {len(project._chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
