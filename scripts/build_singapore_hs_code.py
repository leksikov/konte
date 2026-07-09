#!/usr/bin/env python
"""Build Singapore HS code project with Qwen3.6-35B-A3B context generation."""

import asyncio
import os
from pathlib import Path

os.environ["BACKENDAI_ENDPOINT"] = "https://qwen36_35b.asia03.app.backend.ai/v1"
os.environ["BACKENDAI_MODEL_NAME"] = "Qwen3.6-35B-A3B"

from rich.console import Console
from konte import Project, delete_project, project_exists

console = Console()

PROJECT_NAME = "singapore_hs_code"
DOCS_DIR = Path(__file__).parent.parent / "example_knowledge_base" / "singapore_hs_code"


async def main():
    console.print(f"[bold blue]Building {PROJECT_NAME}[/bold blue]")
    console.print(f"Endpoint: {os.environ['BACKENDAI_ENDPOINT']}")
    console.print(f"Model: {os.environ['BACKENDAI_MODEL_NAME']}")

    if project_exists(PROJECT_NAME):
        console.print(f"[yellow]Deleting existing {PROJECT_NAME}...[/yellow]")
        delete_project(PROJECT_NAME)

    project = Project.create(PROJECT_NAME)

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    console.print(f"Adding {len(pdfs)} PDFs from {DOCS_DIR}")
    for p in pdfs:
        console.print(f"  - {p.name}")
    num_chunks = project.add_documents(pdfs)
    console.print(f"  → {num_chunks} raw chunks")

    console.print("[bold]Building indexes with context generation...[/bold]")
    await project.build(skip_context=False)
    project.save()

    console.print("\n[bold]Smoke query: 'What is the HS code for chemicals?'[/bold]")
    response = project.query("What is the HS code for chemicals?", mode="hybrid", top_k=5)
    for i, result in enumerate(response.results[:3]):
        console.print(f"  [{result.score:.3f}] {result.content[:140]}...")

    console.print(f"\n[bold green]✓ {PROJECT_NAME} built successfully[/bold green]")
    console.print(f"  Total chunks: {len(project._chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
