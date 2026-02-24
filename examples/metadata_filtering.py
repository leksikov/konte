#!/usr/bin/env python3
"""Example: Metadata filtering on retrieval queries.

Demonstrates:
- Adding custom metadata to chunks
- metadata_filter for equality-based filtering (AND logic)
- source_filter for substring matching on source field
- Combining filters with different retrieval modes
"""

import asyncio
from pathlib import Path

from konte import Project, delete_project


# Separate documents to demonstrate multi-source filtering
DOCS = {
    "SAMSUNG_2024_ANNUAL.md": """
# Samsung Electronics Annual Report 2024

## Semiconductor Division
Samsung leads global DRAM market with 42% share.
DDR5 adoption accelerated across data center and consumer segments.
Revenue from memory chips reached $45B, up 25% year-over-year.

## Display Division
OLED panels for smartphones and tablets drove growth.
QD-OLED technology expanded to monitor and TV segments.
Display revenue reached $28B in fiscal 2024.
""",
    "TSMC_2024_ANNUAL.md": """
# TSMC Annual Report 2024

## Foundry Services
TSMC maintained 55% market share in semiconductor foundry.
3nm process node reached full production capacity.
Revenue from advanced nodes (7nm and below) exceeded 60% of total.

## Financial Performance
Total revenue reached $80B in 2024.
Capital expenditure of $30B invested in capacity expansion.
""",
    "SAMSUNG_2023_ANNUAL.md": """
# Samsung Electronics Annual Report 2023

## Semiconductor Division
Memory chip downturn impacted revenue significantly.
DRAM prices declined 40% year-over-year in H1 2023.
Samsung reduced capital expenditure to adjust for market conditions.

## Consumer Electronics
Galaxy S23 series launched with strong initial demand.
Home appliance division maintained stable performance.
""",
}


async def main():
    storage_path = Path("/tmp/konte_metadata_example")
    project_name = "company_reports"

    # --- Setup: Create project with metadata-rich chunks ---
    print("Building project with multiple documents...")
    project = Project.create(name=project_name, storage_path=storage_path)

    doc_dir = storage_path / project_name / "docs"
    doc_dir.mkdir(parents=True, exist_ok=True)

    doc_paths = []
    for filename, content in DOCS.items():
        path = doc_dir / filename
        path.write_text(content, encoding="utf-8")
        doc_paths.append(path)

    project.add_documents(doc_paths)
    await project.build(skip_context=True)
    project.save()
    print("Project ready.\n")

    # --- 1. No filter (baseline) ---
    print("=" * 60)
    print("1. No filter - returns results from all documents")
    print("=" * 60)

    query = "semiconductor revenue"
    response = project.query(query, top_k=5)

    print(f"Query: '{query}'")
    print(f"Results: {response.total_found}")
    for r in response.results[:5]:
        print(f"  [{r.score:.2f}] {r.source}: {r.content[:70]}...")

    # --- 2. source_filter - substring match on source ---
    print("\n" + "=" * 60)
    print("2. source_filter='SAMSUNG' - only Samsung documents")
    print("=" * 60)

    response = project.query(query, top_k=5, source_filter="SAMSUNG")

    print(f"Query: '{query}' (source_filter='SAMSUNG')")
    print(f"Results: {response.total_found}")
    for r in response.results[:5]:
        print(f"  [{r.score:.2f}] {r.source}: {r.content[:70]}...")

    # --- 3. source_filter with year ---
    print("\n" + "=" * 60)
    print("3. source_filter='2024' - only 2024 reports")
    print("=" * 60)

    response = project.query(query, top_k=5, source_filter="2024")

    print(f"Query: '{query}' (source_filter='2024')")
    print(f"Results: {response.total_found}")
    for r in response.results[:5]:
        print(f"  [{r.score:.2f}] {r.source}: {r.content[:70]}...")

    # --- 4. metadata_filter - equality match ---
    print("\n" + "=" * 60)
    print("4. metadata_filter - filter by chunk metadata fields")
    print("=" * 60)

    # Filter by exact source name
    response = project.query(
        query,
        top_k=5,
        metadata_filter={"source": "TSMC_2024_ANNUAL.md"},
    )

    print(f"Query: '{query}' (metadata_filter={{source: 'TSMC_2024_ANNUAL.md'}})")
    print(f"Results: {response.total_found}")
    for r in response.results[:5]:
        print(f"  [{r.score:.2f}] {r.source}: {r.content[:70]}...")

    # --- 5. Different retrieval modes with filter ---
    print("\n" + "=" * 60)
    print("5. Combining filters with retrieval modes")
    print("=" * 60)

    query = "DRAM memory market"
    for mode in ("hybrid", "semantic", "lexical"):
        response = project.query(
            query,
            mode=mode,
            top_k=3,
            source_filter="SAMSUNG",
        )
        top = response.results[0] if response.results else None
        score_str = f"{top.score:.2f}" if top else "N/A"
        print(f"  {mode:10s} | top_score={score_str} | results={response.total_found}")

    # --- Cleanup ---
    delete_project(project_name, storage_path=storage_path)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
