"""What a query's vector costs, and how often it is asked for.

Claim under test: a query used to be embedded once per search of it; it should
now be embedded once per model and text for the process, whichever store
searches with it. Measured over the three shapes of repetition that matter -
the same query searched again, one query fanned out across projects, and one
query searched plainly and then under a filter.

Requests are counted against the local stub rather than timed against a real
endpoint: what changed is how many are sent, and a round trip's long tail would
only add variance to a count.
"""

from __future__ import annotations

import time

from benchmarks.harness import Context, konte_module, stub_embeddings, summarize

QUERY = "Which heading covers parts of electrical machinery?"
REPEATS = 8
PROJECTS = 3
CHUNK_COUNT = 24
SOURCE = "tariff_notes.md"
TOP_K = 5

# Small enough to build in a moment; the case counts requests, not vectors.
DIMENSIONS = 64


def _clear_cache() -> bool:
    """Clear the query embedding cache if this revision has one."""
    module = konte_module("embeddings", required=False)
    clear = getattr(module, "clear_query_embedding_cache", None) if module else None
    if clear is None:
        return False
    clear()
    return True


def _chunks():
    """A corpus small enough to build three times over."""
    models = konte_module("models")
    return [
        models.ContextualizedChunk(
            chunk=models.Chunk(
                chunk_id=f"{SOURCE}_s0_c{index}",
                content=f"Heading 85{index:02d} covers parts of electrical machinery.",
                source=SOURCE,
                segment_idx=0,
                chunk_idx=index,
                metadata={"chapter": "85"},
            ),
            context="This section covers machinery classifications.",
        )
        for index in range(CHUNK_COUNT)
    ]


def _stores(count: int):
    """Build `count` independent indexes, standing in for separate projects.

    Each holds its own embedding client, which is what a shared cache reaches across.
    """
    faiss_store = konte_module("faiss_store")
    chunks = _chunks()
    built = []
    for _ in range(count):
        store = faiss_store.FAISSStore()
        store.build_index(chunks)
        built.append(store)
    return built


def run(ctx: Context) -> dict:
    with stub_embeddings(dim=DIMENSIONS) as tally:
        stores = _stores(PROJECTS)

        _clear_cache()
        tally.reset()
        durations = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            stores[0].query(QUERY, top_k=TOP_K)
            durations.append(time.perf_counter() - start)
        repeated = {
            "searches": REPEATS,
            "requests_sent": len(tally.queries),
            "latency_ms": summarize(durations),
        }

        _clear_cache()
        tally.reset()
        for store in stores:
            store.query(QUERY, top_k=TOP_K)
        fanned_out = {"searches": PROJECTS, "requests_sent": len(tally.queries)}

        _clear_cache()
        tally.reset()
        stores[0].query(QUERY, top_k=TOP_K)
        stores[0].query(QUERY, top_k=TOP_K, source_filter=SOURCE)
        both_paths = {"searches": 2, "requests_sent": len(tally.queries)}

        _clear_cache()
        tally.reset()
        for index in range(REPEATS):
            stores[0].query(f"{QUERY} variant {index}", top_k=TOP_K)
        distinct = {"searches": REPEATS, "requests_sent": len(tally.queries)}

    return {
        "has_query_embedding_cache": _clear_cache(),
        "repeated_query": repeated,
        "one_query_across_projects": fanned_out,
        "filtered_and_unfiltered": both_paths,
        "distinct_queries": distinct,
    }
