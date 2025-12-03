#!/usr/bin/env python3
"""Basic usage example for Konte contextual RAG library."""

import asyncio
from pathlib import Path

from konte import Project, create_project, get_project, list_projects, delete_project


async def main():
    """Demonstrate basic Konte usage."""
    # Define storage path for this example
    storage_path = Path("/tmp/konte_example")
    project_name = "tariff_docs"

    # Create a new project
    print("Creating project...")
    project = Project.create(
        name=project_name,
        storage_path=storage_path,
    )

    # Get path to sample documents
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    documents = [
        fixtures_dir / "sample.txt",
        fixtures_dir / "sample.md",
    ]

    # Add documents
    print(f"Adding {len(documents)} documents...")
    num_chunks = project.add_documents(documents)
    print(f"Created {num_chunks} chunks")

    # Build indexes (with context generation)
    print("Building indexes with context generation...")
    await project.build(skip_context=False)

    # Save project for later use
    project.save()
    print(f"Project saved to {project.project_dir}")

    # Query the project
    print("\n--- Querying ---")

    queries = [
        "What is the Harmonized System?",
        "How is import duty calculated?",
        "What are anti-dumping duties?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = project.query(query, top_k=3)

        print(f"Found: {response.total_found} results")
        print(f"Top score: {response.top_score:.2f}")
        print(f"Suggested action: {response.suggested_action}")

        for i, result in enumerate(response.results[:2], 1):
            print(f"\n  Result {i} (score: {result.score:.2f}):")
            print(f"  Source: {result.source}")
            print(f"  Content: {result.content[:150]}...")

    # Demonstrate different retrieval modes
    print("\n--- Retrieval Modes ---")

    query = "tariff code 8542"

    print(f"\nQuery: {query}")

    # Hybrid (default)
    response = project.query(query, mode="hybrid", top_k=1)
    print(f"Hybrid: score={response.top_score:.2f}")

    # Semantic only
    response = project.query(query, mode="semantic", top_k=1)
    print(f"Semantic: score={response.top_score:.2f}")

    # Lexical only
    response = project.query(query, mode="lexical", top_k=1)
    print(f"Lexical: score={response.top_score:.2f}")

    # Demonstrate loading existing project
    print("\n--- Loading Existing Project ---")

    project2 = Project.open(project_name, storage_path=storage_path)
    response = project2.query("classification rules")
    print(f"Loaded project, query returned {response.total_found} results")

    # List projects
    print("\n--- Project Management ---")
    projects = list_projects(storage_path=storage_path)
    print(f"Projects: {projects}")

    # Clean up
    print("\nCleaning up...")
    delete_project(project_name, storage_path=storage_path)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
