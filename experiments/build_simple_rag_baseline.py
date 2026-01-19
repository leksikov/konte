"""Build Simple RAG baseline project for comparison with Contextual RAG.

Creates a project with:
- Same chunking strategy (8000 token segments, 800 token chunks)
- Same embedding model (text-embedding-3-small)
- Same hybrid retrieval (FAISS + BM25)
- NO LLM context generation (skip_context=True)
- NO reranking

This provides a fair comparison to measure the value of:
1. LLM-generated context prepended to chunks
2. LLM-based reranking
"""

import asyncio
from pathlib import Path

import structlog

from konte import Project

logger = structlog.get_logger()

# Source document (same as wco_hs_explanatory_notes_korean)
SOURCE_DOCUMENT = Path("example_knowledge_base/wco_hs_explanatory_notes.md")

# Project names
SIMPLE_RAG_PROJECT = "simple_rag_baseline"
CONTEXTUAL_RAG_PROJECT = "wco_hs_explanatory_notes_korean"


async def build_simple_rag_baseline(
    source_path: Path = SOURCE_DOCUMENT,
    project_name: str = SIMPLE_RAG_PROJECT,
    force_rebuild: bool = False,
) -> Project:
    """Build Simple RAG baseline project.

    Uses skip_context=True to skip LLM context generation.
    Same chunking and embedding as Contextual RAG project.

    Args:
        source_path: Path to source document.
        project_name: Name for the baseline project.
        force_rebuild: If True, delete and rebuild even if exists.

    Returns:
        Built Project instance.
    """
    from konte.config import settings

    project_dir = settings.STORAGE_PATH / project_name

    if project_dir.exists() and not force_rebuild:
        logger.info("loading_existing_project", name=project_name)
        return Project.open(project_name)

    if project_dir.exists() and force_rebuild:
        import shutil
        logger.info("deleting_existing_project", name=project_name)
        shutil.rmtree(project_dir)

    logger.info("creating_simple_rag_project", name=project_name, source=str(source_path))

    # Create project with same settings as contextual RAG
    project = Project.create(
        name=project_name,
        segment_size=8000,
        segment_overlap=800,
        chunk_size=800,
        chunk_overlap=80,
    )

    # Add source document
    project.add_documents([source_path])
    logger.info("documents_added", num_chunks=len(project._chunks))

    # Build with skip_context=True (Simple RAG)
    logger.info("building_indexes", skip_context=True)
    await project.build(skip_context=True)

    # Save project
    project.save()
    logger.info("project_saved", path=str(project_dir))

    return project


async def compare_project_sizes():
    """Compare sizes of Simple RAG vs Contextual RAG projects."""
    from konte.config import settings
    import os

    simple_dir = settings.STORAGE_PATH / SIMPLE_RAG_PROJECT
    contextual_dir = settings.STORAGE_PATH / CONTEXTUAL_RAG_PROJECT

    def get_dir_size(path: Path) -> int:
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    if simple_dir.exists():
        simple_size = get_dir_size(simple_dir)
        print(f"Simple RAG project size: {simple_size / 1024 / 1024:.2f} MB")

    if contextual_dir.exists():
        contextual_size = get_dir_size(contextual_dir)
        print(f"Contextual RAG project size: {contextual_size / 1024 / 1024:.2f} MB")

        if simple_dir.exists():
            diff = contextual_size - simple_size
            print(f"Context overhead: {diff / 1024 / 1024:.2f} MB ({diff / simple_size * 100:.1f}% increase)")


async def main():
    """Build Simple RAG baseline project."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Simple RAG baseline project")
    parser.add_argument("--source", type=Path, default=SOURCE_DOCUMENT,
                        help="Path to source document")
    parser.add_argument("--name", default=SIMPLE_RAG_PROJECT,
                        help="Project name")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if project exists")
    parser.add_argument("--compare-sizes", action="store_true",
                        help="Compare project sizes after building")
    args = parser.parse_args()

    print(f"Building Simple RAG baseline project: {args.name}")
    print(f"Source document: {args.source}")
    print(f"Force rebuild: {args.force}")
    print()

    project = await build_simple_rag_baseline(
        source_path=args.source,
        project_name=args.name,
        force_rebuild=args.force,
    )

    print(f"\nProject built successfully!")
    print(f"  Name: {project.config.name}")
    print(f"  Chunks: {len(project._contextualized_chunks)}")
    print(f"  Location: {project.project_dir}")

    if args.compare_sizes:
        print("\n--- Project Size Comparison ---")
        await compare_project_sizes()


if __name__ == "__main__":
    asyncio.run(main())
