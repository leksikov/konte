"""What a filtered vector query costs.

Claim under test: a filtered query used to walk the whole docstore, looking up
every document and testing it against the filter, once per query. It should now
resolve filters through an inverted index built on the first filtered query, so
an unfiltered workload never pays for it.

Embeddings are answered locally. The effect being measured is tens of
microseconds to tens of milliseconds; a real embedding round trip is hundreds of
milliseconds with a long tail and would bury it. Both revisions search with the
same vector for the same query, so the comparison is unaffected.

Runs against a copy of a real built project - filter selectivity depends on how
sources and metadata are actually distributed, which synthetic data would not
reproduce faithfully.
"""

from __future__ import annotations

import time
from collections import Counter

from benchmarks.build import open_project
from benchmarks.corpus import available_real_projects, real_project
from benchmarks.harness import (
    Context,
    peak_rss_mb,
    pin_keyword_extraction,
    stub_embeddings,
    summarize,
)

DEFAULT_PROJECT = "all_tariff_documents"
QUERY_COUNT = 40


def _queries(count: int) -> list[str]:
    return [f"classification of parts and accessories variant {i}" for i in range(count)]


def _pick_source(project) -> tuple[str | None, int, int]:
    """Choose a source substring that selects a useful slice of the corpus.

    Reads the chunk list directly. On the newer revision that materializes the
    deferred corpus, which is why it happens before anything is timed.
    """
    chunks = getattr(project, "_contextualized_chunks", None) or []
    sources = Counter(chunk.chunk.source for chunk in chunks)
    if not sources:
        return None, 0, 0
    # Prefer a source that is a real subset, so the filter actually excludes work.
    total = sum(sources.values())
    ranked = sources.most_common()
    for name, count in ranked:
        if count < total:
            return name, count, total
    return ranked[0][0], ranked[0][1], total


def run(ctx: Context) -> dict:
    name = ctx.options.get("project", DEFAULT_PROJECT)
    count = int(ctx.options.get("queries", QUERY_COUNT))

    if name not in available_real_projects():
        return {"status": "skipped", "reason": f"no built project {name!r}"}

    with stub_embeddings():
        storage, project_name = real_project(name, ctx.revision, reindex_lexical=True)
        project = open_project(project_name, storage)
        pin_keyword_extraction()

        source, matching, total = _pick_source(project)
        queries = _queries(count)

        # Warm the vector search path so the first unfiltered timing is not
        # paying for lazily initialized FAISS state.
        project.query(queries[0], mode="semantic", top_k=20)

        unfiltered = []
        for query in queries:
            start = time.perf_counter()
            project.query(query, mode="semantic", top_k=20)
            unfiltered.append(time.perf_counter() - start)

        rss_before_filter = peak_rss_mb()

        # First filtered query: where the newer revision builds its postings.
        start = time.perf_counter()
        first = project.query(queries[0], mode="semantic", top_k=20, source_filter=source)
        first_filtered_seconds = time.perf_counter() - start
        rss_after_filter = peak_rss_mb()

        filtered = []
        for query in queries:
            start = time.perf_counter()
            project.query(query, mode="semantic", top_k=20, source_filter=source)
            filtered.append(time.perf_counter() - start)

    return {
        "project": name,
        "filter": {"source": source, "matching_chunks": matching, "total_chunks": total},
        "unfiltered_query_ms": summarize(unfiltered),
        "first_filtered_query_ms": first_filtered_seconds * 1000,
        "first_filtered_results": len(first.results),
        "filtered_query_ms": summarize(filtered),
        "rss_mb": {
            "before_first_filter": rss_before_filter,
            "after_first_filter": rss_after_filter,
            "postings_mb": rss_after_filter - rss_before_filter,
        },
    }
