#!/usr/bin/env python3
"""Demo: source_filter, regex metadata extraction, and get_llm() fix.

Shows the three fixes from the retriever regression:
1. Regex fix - multi-word company names parse correctly
2. source_filter - substring match on chunk.source (bypasses metadata_filter)
3. get_llm() reuse - query_processor uses Backend.AI when configured
"""

from konte.chunker import extract_metadata_from_source
from konte.models import Chunk, ContextualizedChunk
from konte.stores.bm25_store import BM25Store


def demo_regex_fix():
    """Demo 1: Regex now handles multi-word company names."""
    print("=== Regex Fix: extract_metadata_from_source ===\n")

    test_cases = [
        "ADOBE_2022_10K.md",
        "JOHNSON_JOHNSON_2022_10K.md",
        "3M_2018_10K.md",
        "PROCTER_GAMBLE_2023_ANNUAL.pdf",
        "readme.md",
    ]

    for filename in test_cases:
        result = extract_metadata_from_source(filename)
        if result:
            print(f"  {filename:45s} -> company={result['company']}, year={result['year']}")
        else:
            print(f"  {filename:45s} -> no match")

    print()


def demo_source_filter():
    """Demo 2: source_filter for substring matching on chunk.source."""
    print("=== source_filter: BM25 Store ===\n")

    # Build a small BM25 index with chunks from multiple sources
    chunks = []
    data = [
        ("ADOBE_2022_10K.md", "Adobe revenue grew 15% year-over-year in fiscal 2022."),
        ("ADOBE_2022_10K.md", "Adobe Creative Cloud subscriptions reached 30M users."),
        ("JOHNSON_JOHNSON_2022_10K.md", "Johnson & Johnson pharmaceutical sales exceeded $50B."),
        ("JOHNSON_JOHNSON_2022_10K.md", "Johnson & Johnson medical devices segment grew 8%."),
        ("3M_2018_10K.md", "3M industrial segment revenue was $12B in 2018."),
        ("3M_2018_10K.md", "3M safety and graphics division performance improved."),
    ]

    for i, (source, content) in enumerate(data):
        chunk = Chunk(
            chunk_id=f"{source}_s0_c{i}",
            content=content,
            source=source,
            segment_idx=0,
            chunk_idx=i,
        )
        chunks.append(ContextualizedChunk(chunk=chunk, context=""))

    store = BM25Store()
    store.build_index(chunks)

    query = "revenue"

    # No filter - returns from all sources
    results = store.query(query, top_k=10)
    print(f'  query="{query}", no filter:')
    for chunk, score in results:
        print(f"    [{score:.2f}] {chunk.chunk.source}: {chunk.chunk.content[:60]}...")

    print()

    # source_filter="ADOBE" - only ADOBE chunks
    results = store.query(query, top_k=10, source_filter="ADOBE")
    print(f'  query="{query}", source_filter="ADOBE":')
    for chunk, score in results:
        print(f"    [{score:.2f}] {chunk.chunk.source}: {chunk.chunk.content[:60]}...")

    print()

    # source_filter="JOHNSON_JOHNSON" - only J&J chunks
    results = store.query(query, top_k=10, source_filter="JOHNSON_JOHNSON")
    print(f'  query="{query}", source_filter="JOHNSON_JOHNSON":')
    for chunk, score in results:
        print(f"    [{score:.2f}] {chunk.chunk.source}: {chunk.chunk.content[:60]}...")

    print()

    # source_filter with no match
    results = store.query(query, top_k=10, source_filter="NONEXISTENT")
    print(f'  query="{query}", source_filter="NONEXISTENT": {len(results)} results')

    print()


def demo_get_llm_reuse():
    """Demo 3: query_processor now reuses get_llm() from context.py."""
    print("=== get_llm() Reuse (DRY/SSoT) ===\n")

    # Show that query_processor imports get_llm from context (not its own _get_llm)
    import inspect
    import konte.query_processor as qp

    source = inspect.getsource(qp)

    has_get_llm_import = "from konte.context import get_llm" in source
    has_old_get_llm = "def _get_llm" in source

    print(f"  imports get_llm from context.py:  {has_get_llm_import}")
    print(f"  has old _get_llm() function:      {has_old_get_llm}")
    print()


if __name__ == "__main__":
    demo_regex_fix()
    demo_source_filter()
    demo_get_llm_reuse()
