"""Build baseline RAG project without contextual retrieval.

Creates a project with skip_context=True for baseline comparison.
Uses the same source document as wco_hs_explanatory_notes_korean.
"""

import asyncio
from pathlib import Path

import structlog

from konte import Project

logger = structlog.get_logger()


async def main():
    """Build baseline project without context generation."""
    project_name = "baseline_wco_korean"
    source_doc = Path("example_knowledge_base/wco_hs_explanatory_notes.md")

    if not source_doc.exists():
        print(f"Source document not found: {source_doc}")
        return

    print(f"Building baseline project: {project_name}")
    print(f"Source: {source_doc}")
    print("Context generation: DISABLED (skip_context=True)")

    project = Project.create(project_name)
    num_chunks = project.add_documents([source_doc])
    print(f"Documents chunked: {num_chunks} chunks")

    print("Building indexes (no context generation)...")
    await project.build(skip_context=True)

    project.save()
    print(f"\nBaseline project saved: {project_name}")
    print(f"Total chunks: {len(project._chunks)}")
    print(f"Contextualized chunks: {len(project._contextualized_chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
