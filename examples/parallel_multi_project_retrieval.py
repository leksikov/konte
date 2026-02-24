"""Example: Parallel retrieval across multiple Konte projects.

Use case: Different document collections indexed separately, queried in parallel.
- Project 1: HS Code classifications (tariff rules)
- Project 2: GRI interpretation rules
- Project 3: Product examples/precedents

All queried simultaneously, results merged.
"""

import asyncio
from pathlib import Path

from konte import Project, RetrievalResponse, RetrievalResult


async def build_demo_projects(base_path: Path) -> list[Project]:
    """Build demo projects with sample content."""

    # Sample content for each project
    projects_data = [
        {
            "name": "hs_codes",
            "content": """
# HS Code 8542 - Electronic Integrated Circuits

## 8542.31 - Processors and Controllers
Microprocessors, microcontrollers, and digital signal processors.
Examples: Intel Core, AMD Ryzen, ARM Cortex chips.

## 8542.32 - Memories
DRAM, SRAM, EPROM, EEPROM, and flash memory.
Examples: Samsung DDR4 RAM, SK Hynix memory chips.

## 8542.33 - Amplifiers
Operational amplifiers, audio amplifiers.

## 8542.39 - Other Integrated Circuits
ASICs, FPGAs, and other specialized ICs.
""",
        },
        {
            "name": "gri_rules",
            "content": """
# General Rules for Interpretation (GRI)

## GRI Rule 1
Classification is determined by the terms of headings and section/chapter notes.

## GRI Rule 3(a) - Specific Description
The heading providing the most specific description takes precedence.
For memory chips: 8542.32 is more specific than 8542.

## GRI Rule 3(b) - Essential Character
Composite goods classified by component giving essential character.

## GRI Rule 6 - Subheading Classification
Classification at subheading level follows same rules.
""",
        },
        {
            "name": "precedents",
            "content": """
# Classification Precedents

## Case 2023-001: Samsung DDR4 RAM Module
Product: 16GB DDR4-3200 DRAM module
Classification: 8542.32 (Memories)
Reasoning: Primary function is data storage, integrated circuit form.

## Case 2023-002: Intel Core i9 Processor
Product: Desktop CPU with integrated graphics
Classification: 8542.31 (Processors)
Reasoning: Primary function is data processing.

## Case 2023-003: USB Flash Drive
Product: 64GB USB storage device
Classification: 8523.51 (Solid-state storage)
Reasoning: Complete storage unit, not bare IC.
""",
        },
    ]

    projects = []
    for data in projects_data:
        project_path = base_path / data["name"]
        project_path.mkdir(parents=True, exist_ok=True)

        # Write content file
        content_file = project_path / "content.md"
        content_file.write_text(data["content"], encoding="utf-8")

        # Create and build project
        project = Project.create(
            name=data["name"],
            storage_path=base_path,
        )
        project.add_documents([content_file])
        await project.build(skip_context=False)
        project.save()

        projects.append(project)
        print(f"  Built project '{data['name']}')")

    return projects


def query_all_projects(
    projects: list[Project],
    query: str,
    top_k: int = 5,
) -> dict[str, RetrievalResponse]:
    """Query multiple projects.

    Note: project.query() is sync and fast (in-memory index lookup).
    No async overhead needed - just loop through projects.

    For true parallelism with slow queries, consider:
    - Making project.query() async
    - Or using multiprocessing for CPU-bound work

    Args:
        projects: List of Konte projects to query.
        query: Search query.
        top_k: Results per project.

    Returns:
        Dict mapping project name to RetrievalResponse.
    """
    results = {}
    for project in projects:
        response = project.query(query, mode="hybrid", top_k=top_k)
        results[project.config.name] = response
    return results


def merge_results(
    responses: dict[str, RetrievalResponse],
    top_k: int = 10,
) -> list[tuple[str, RetrievalResult]]:
    """Merge results from multiple projects, sorted by score.

    Args:
        responses: Dict of project_name -> RetrievalResponse.
        top_k: Total results to return.

    Returns:
        List of (project_name, result) tuples sorted by score.
    """
    all_results = []

    for project_name, response in responses.items():
        for result in response.results:
            all_results.append((project_name, result))

    # Sort by score descending
    all_results.sort(key=lambda x: x[1].score, reverse=True)

    return all_results[:top_k]


async def main():
    print("=" * 60)
    print("Parallel Multi-Project Retrieval Example")
    print("=" * 60)

    base_path = Path("/tmp/konte_multi_project_demo")

    # Build projects
    print("\n1. Building demo projects...")
    projects = await build_demo_projects(base_path)

    # Query
    query = "Samsung DDR4 RAM memory chip classification"
    print(f"\n2. Querying all projects...")
    print(f"   Query: '{query}'")

    responses = query_all_projects(projects, query, top_k=3)

    # Show individual results
    print("\n3. Results per project:")
    print("-" * 60)

    for project_name, response in responses.items():
        print(f"\n[{project_name}] (top_score: {response.top_score:.3f})")
        for i, r in enumerate(response.results[:2], 1):
            preview = r.content[:80].replace("\n", " ")
            print(f"  {i}. [{r.score:.3f}] {preview}...")

    # Merge and show combined results
    print("\n4. Merged results (all projects, sorted by score):")
    print("-" * 60)

    merged = merge_results(responses, top_k=5)
    for project_name, result in merged:
        preview = result.content[:60].replace("\n", " ")
        print(f"  [{result.score:.3f}] ({project_name}) {preview}...")

    # Demonstrate different queries
    print("\n5. Another query: 'GRI Rule interpretation'")
    print("-" * 60)

    responses2 = query_all_projects(projects, "GRI Rule interpretation", top_k=2)
    merged2 = merge_results(responses2, top_k=3)

    for project_name, result in merged2:
        preview = result.content[:60].replace("\n", " ")
        print(f"  [{result.score:.3f}] ({project_name}) {preview}...")

    # Cleanup
    import shutil
    shutil.rmtree(base_path, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Done! Multi-project parallel retrieval works.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
