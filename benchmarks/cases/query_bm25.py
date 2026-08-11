"""What a lexical query costs.

Claim under test: the lexical query path made four full passes over the corpus
in Python - materializing a candidate list covering every chunk even when
nothing was filtered, sorting every chunk through a Python comparison key, and
scanning for the normalization bounds - and now ranks with NumPy instead.

Also visible here: the corpus behind the lexical index is no longer parsed when
the project is opened, so the first lexical query pays for it and later ones do
not. Cold and warm are therefore reported separately; averaging them together
would hide both effects.

Keyword extraction is pinned to local tokenization on both revisions. The older
one cannot turn it off, and an LLM round trip inside the timed region would
swamp the millisecond differences this case exists to measure.
"""

from __future__ import annotations

import time

from benchmarks.build import ensure_project, open_project
from benchmarks.corpus import projects_dir, synthetic_document
from benchmarks.harness import Context, peak_rss_mb, pin_keyword_extraction, summarize

DEFAULT_CHUNKS = 20000
QUERY_COUNT = 200

_TERMS = [
    "classification heading parts",
    "essential character composite goods",
    "retail sale sets components",
    "machinery mechanical appliances",
    "electrical equipment parts",
    "optical precision instruments",
    "base metals articles",
    "plastics articles thereof",
    "textile fabrics made up",
    "vehicles transport equipment",
]


def _queries(count: int) -> list[str]:
    """Distinct queries, so nothing is answered from a cache."""
    return [f"{_TERMS[i % len(_TERMS)]} {i}" for i in range(count)]


def run(ctx: Context) -> dict:
    size = int(ctx.options.get("chunks", DEFAULT_CHUNKS))
    count = int(ctx.options.get("queries", QUERY_COUNT))

    document = synthetic_document(size)
    storage = projects_dir(f"bm25-{size}-{ctx.revision}")
    ensure_project(f"bm25_{size}", storage, document, enable_faiss=False, skip_context=True)

    pin_keyword_extraction()
    rss_before_open = peak_rss_mb()

    start = time.perf_counter()
    project = open_project(f"bm25_{size}", storage)
    open_seconds = time.perf_counter() - start
    rss_after_open = peak_rss_mb()

    queries = _queries(count)

    # First lexical query: on the newer revision this is where the corpus is
    # parsed, which the older revision paid for during open().
    start = time.perf_counter()
    first = project.query(queries[0], mode="lexical", top_k=20)
    cold_seconds = time.perf_counter() - start
    rss_after_first = peak_rss_mb()

    warm = []
    for query in queries[1:]:
        start = time.perf_counter()
        project.query(query, mode="lexical", top_k=20)
        warm.append(time.perf_counter() - start)

    return {
        "corpus": {"requested_chunks": size, "queries": count},
        "open_seconds": open_seconds,
        "first_query_seconds": cold_seconds,
        "first_query_results": len(first.results),
        "warm_query_ms": summarize(warm),
        "rss_mb": {
            "before_open": rss_before_open,
            "after_open": rss_after_open,
            "after_first_query": rss_after_first,
        },
    }
