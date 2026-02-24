#!/usr/bin/env python3
"""Example: Async querying with LLM reranking.

Demonstrates:
- query_async() for async retrieval
- LLM reranking to improve result quality
- Comparing results with and without reranking
- rerank_initial_k parameter for controlling candidate pool

Requires: OPENAI_API_KEY or BACKENDAI_ENDPOINT configured.
"""

import asyncio
from pathlib import Path

from konte import Project, delete_project


SAMPLE_CONTENT = """
# International Trade Classification Rules

## General Rules for Interpretation (GRI)

### GRI Rule 1 - Terms of Headings
Classification shall be determined according to the terms of the headings
and any relative section or chapter notes.

### GRI Rule 2(a) - Incomplete Articles
Any reference to an article shall include that article incomplete or unfinished,
provided it has the essential character of the complete article.

### GRI Rule 2(b) - Mixtures and Combinations
Any reference to a material includes mixtures or combinations of that material
with other materials.

### GRI Rule 3(a) - Most Specific Description
When goods are classifiable under two or more headings,
the heading which provides the most specific description shall be preferred.

### GRI Rule 3(b) - Essential Character
Mixtures, composite goods, and goods in sets shall be classified
by the component which gives them their essential character.

### GRI Rule 3(c) - Last in Numerical Order
When neither 3(a) nor 3(b) applies, classify under the last heading
in numerical order among those which equally merit consideration.

### GRI Rule 4 - Most Akin
Goods which cannot be classified by the above rules shall be classified
under the heading appropriate to the goods to which they are most akin.

### GRI Rule 5 - Containers and Packing
Cases, boxes, and similar containers presented with the goods
shall be classified with the goods if they are of a kind normally used for such goods.

### GRI Rule 6 - Subheading Classification
Classification at the subheading level follows the same principles as heading level.
Subheadings at the same level are comparable.
Only subheadings at the same level can be compared for classification purposes.

## Section XVI Notes (Chapters 84-85)

### Note 1 - Exclusions
This section does not cover: conveyor belts of plastics (Ch.39),
transmission belts of rubber (Ch.40), articles of leather (Ch.42).

### Note 2 - Parts Classification
Parts which are goods included in any of the headings of Chapter 84 or 85
are classified in their respective headings.
Parts suitable for use solely or principally with a particular machine
are classified with that machine.

### Note 3 - Combined Machines
A machine designed to perform two or more functions shall be classified
under the heading for its principal function.

## Examples: Applying GRI Rules

### Example 1: Smartphone
A smartphone combines: telephone (8517), camera (9006), computer (8471).
Apply GRI 3(b): essential character is telephone function -> 8517.13

### Example 2: Laptop with Built-in Printer
A laptop with integrated printer: computer (8471), printer (8443).
Apply GRI 3(a): more specific description is computer -> 8471.30

### Example 3: Parts of a Lathe
A spindle for a metalworking lathe: part of machine (8466).
Apply Section XVI Note 2: classified with the machine -> 8466.93
"""


async def main():
    storage_path = Path("/tmp/konte_reranking_example")
    project_name = "trade_rules"

    # --- Setup ---
    print("Building project...")
    project = Project.create(name=project_name, storage_path=storage_path)

    content_file = storage_path / project_name / "rules.md"
    content_file.parent.mkdir(parents=True, exist_ok=True)
    content_file.write_text(SAMPLE_CONTENT, encoding="utf-8")

    project.add_documents([content_file])
    await project.build(skip_context=True)
    project.save()
    print("Project ready.\n")

    # --- 1. Standard query (no reranking) ---
    print("=" * 60)
    print("1. Standard query (no reranking)")
    print("=" * 60)

    query = "How to classify a product with multiple functions?"

    response_standard = await project.query_async(query=query, top_k=5)

    print(f"Query: '{query}'")
    print(f"Top score: {response_standard.top_score:.3f}")
    for i, r in enumerate(response_standard.results[:5], 1):
        print(f"  [{i}] score={r.score:.3f} | {r.content[:70]}...")

    # --- 2. With LLM reranking ---
    print("\n" + "=" * 60)
    print("2. With LLM reranking (rerank=True)")
    print("=" * 60)

    response_reranked = await project.query_async(
        query=query,
        top_k=5,
        rerank=True,
        rerank_initial_k=20,  # retrieve 20, rerank to top 5
    )

    print(f"Query: '{query}'")
    print(f"Top score: {response_reranked.top_score:.3f}")
    for i, r in enumerate(response_reranked.results[:5], 1):
        print(f"  [{i}] score={r.score:.3f} | {r.content[:70]}...")

    # --- 3. Compare ordering ---
    print("\n" + "=" * 60)
    print("3. Comparison: top result with vs without reranking")
    print("=" * 60)

    if response_standard.results and response_reranked.results:
        std_top = response_standard.results[0]
        rnk_top = response_reranked.results[0]
        print(f"Standard top:  [{std_top.score:.3f}] {std_top.content[:80]}...")
        print(f"Reranked top:  [{rnk_top.score:.3f}] {rnk_top.content[:80]}...")
        same = std_top.chunk_id == rnk_top.chunk_id
        print(f"Same top result: {same}")

    # --- Cleanup ---
    delete_project(project_name, storage_path=storage_path)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
