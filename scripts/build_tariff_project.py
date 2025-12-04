#!/usr/bin/env python3
"""Build a Konte project from Korean tariff documents."""

import asyncio
from pathlib import Path

from konte import Project


async def main():
    """Build project from example_knowledge_base."""
    # Project setup
    project_name = "korean_tariff"
    storage_path = Path("/tmp/konte_tariff")

    # Document to process
    docs_dir = Path(__file__).parent.parent / "example_knowledge_base"
    documents = [
        docs_dir / "[별표 ] 관세율표(제50조 관련)(관세법).md",
    ]

    # Verify documents exist
    for doc in documents:
        if not doc.exists():
            print(f"Document not found: {doc}")
            return

    print(f"Creating project: {project_name}")
    print(f"Storage path: {storage_path}")

    # Create project (uses settings defaults: gpt-4.1-mini)
    project = Project.create(
        name=project_name,
        storage_path=storage_path,
    )

    print(f"\nAdding {len(documents)} document(s)...")
    for doc in documents:
        print(f"  - {doc.name}")

    num_chunks = project.add_documents(documents)
    print(f"Created {num_chunks} chunks")

    print("\nBuilding indexes with context generation...")
    print("(This may take a few minutes for context generation)")
    await project.build(skip_context=False)

    # Save project
    project.save()
    print(f"\nProject saved to: {project.project_dir}")

    # Test queries
    print("\n" + "=" * 60)
    print("Testing queries...")
    print("=" * 60)

    queries = [
        "쇠고기 관세율",  # Beef tariff rate
        "닭고기 냉동",    # Frozen chicken
        "돼지고기 세율",  # Pork tariff
        "0203",          # HS code for pork
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = project.query(query, top_k=3)

        print(f"Total found: {response.total_found}")
        print(f"Top score: {response.top_score:.3f}")
        print(f"Suggested action: {response.suggested_action}")

        for i, result in enumerate(response.results[:2], 1):
            print(f"\n  Result {i} (score: {result.score:.3f}):")
            print(f"  Source: {result.source}")
            print(f"  Context: {result.context[:100]}..." if result.context else "  Context: (none)")
            print(f"  Content: {result.content[:150]}...")


if __name__ == "__main__":
    asyncio.run(main())
