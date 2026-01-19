"""Build a combined project with all chunks from all documents."""

import asyncio
import json
import os
from pathlib import Path

# Set BackendAI endpoint for context generation (if rebuilding)
os.environ["BACKENDAI_ENDPOINT"] = "https://qwen3vl.asia03.app.backend.ai/v1"
os.environ["BACKENDAI_MODEL_NAME"] = "Qwen3-VL-8B-Instruct"

from rich.console import Console

from konte import Project, project_exists, delete_project, list_projects
from konte.models import Chunk, ContextualizedChunk
from konte.stores.faiss_store import FAISSStore
from konte.stores.bm25_store import BM25Store
from konte.stores.retriever import Retriever

console = Console()

COMBINED_PROJECT_NAME = "all_tariff_documents"


async def main():
    """Merge all existing projects into one combined project."""
    console.print("[bold magenta]Building Combined Project[/bold magenta]")
    console.print("=" * 60)

    # List existing projects
    projects = list_projects()
    projects = [p for p in projects if p != COMBINED_PROJECT_NAME]

    console.print(f"\nFound {len(projects)} source projects:")
    for p in projects:
        console.print(f"  - {p}")

    # Collect all chunks with enhanced metadata
    all_chunks: list[ContextualizedChunk] = []
    all_segments: dict[int, str] = {}
    segment_offset = 0

    for project_name in projects:
        console.print(f"\n[cyan]Loading: {project_name}[/cyan]")

        project = Project.open(project_name)
        chunks_file = project.project_dir / "chunks.json"

        if not chunks_file.exists():
            console.print(f"  [yellow]No chunks.json found, skipping[/yellow]")
            continue

        with open(chunks_file) as f:
            chunks_data = json.load(f)

        console.print(f"  Chunks: {len(chunks_data)}")

        # Load segments
        segments_file = project.project_dir / "segments.json"
        project_segments = {}
        if segments_file.exists():
            with open(segments_file) as f:
                project_segments = json.load(f)

        for chunk_data in chunks_data:
            chunk_info = chunk_data["chunk"]

            # Create chunk with enhanced metadata
            chunk = Chunk(
                chunk_id=f"{project_name}_{chunk_info['chunk_id']}",
                content=chunk_info["content"],
                source=chunk_info["source"],
                segment_idx=chunk_info["segment_idx"] + segment_offset,
                chunk_idx=chunk_info["chunk_idx"],
                metadata={
                    "document_name": project_name,
                    "original_chunk_id": chunk_info["chunk_id"],
                    "original_segment_idx": chunk_info["segment_idx"],
                    "original_source": chunk_info["source"],
                    **chunk_info.get("metadata", {}),
                },
            )

            ctx_chunk = ContextualizedChunk(
                chunk=chunk,
                context=chunk_data.get("context", ""),
            )
            all_chunks.append(ctx_chunk)

        # Update segment offset for next project
        if project_segments:
            max_seg = max(int(k) for k in project_segments.keys())
            for k, v in project_segments.items():
                all_segments[int(k) + segment_offset] = v
            segment_offset += max_seg + 1

    console.print(f"\n[bold]Total chunks collected: {len(all_chunks)}[/bold]")

    # Delete existing combined project
    if project_exists(COMBINED_PROJECT_NAME):
        console.print(f"\n[dim]Deleting existing {COMBINED_PROJECT_NAME}...[/dim]")
        delete_project(COMBINED_PROJECT_NAME)

    # Create combined project
    console.print(f"\n[cyan]Creating combined project: {COMBINED_PROJECT_NAME}[/cyan]")
    combined = Project.create(COMBINED_PROJECT_NAME)

    # Set contextualized chunks directly (skip context generation)
    combined._contextualized_chunks = all_chunks
    combined._segments = all_segments

    # Build indexes directly from contextualized chunks
    console.print("[dim]Building FAISS index...[/dim]")
    combined._faiss = FAISSStore(embedding_model=combined._config.embedding_model)
    combined._faiss.build_index(all_chunks)

    console.print("[dim]Building BM25 index...[/dim]")
    combined._bm25 = BM25Store()
    combined._bm25.build_index(all_chunks)

    # Create retriever
    combined._retriever = Retriever(
        faiss_store=combined._faiss,
        bm25_store=combined._bm25,
    )

    # Save
    combined.save()
    console.print(f"\n[green]✓ Combined project saved: {COMBINED_PROJECT_NAME}[/green]")
    console.print(f"  Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
