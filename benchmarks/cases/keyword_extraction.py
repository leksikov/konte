"""What BM25 keyword extraction costs a query.

Extraction sits between a caller and their search results, so its failure modes
are latency the caller pays. Two claims:

- a stalled endpoint used to cost a query the batch-sized timeout multiplied by
  the retry budget; it should now cost one short timeout
- a repeated query used to go back on the wire every time; it should now be
  answered from cache

Both are measured against the local stub, because neither can be provoked from
a real endpoint and both are counts and timeouts rather than throughput.
"""

from __future__ import annotations

import importlib
import time

from benchmarks.harness import Context, point_llm_at, summarize
from benchmarks.stub_endpoint import MODEL_NAME, stub_endpoint

QUERY = "Which heading covers parts of electrical machinery?"
REPEATS = 8


def _extractor():
    """Return this revision's synchronous keyword extraction entry point."""
    return importlib.import_module("konte.query_processor").extract_search_keywords


def _clear_keyword_cache() -> bool:
    """Clear the extraction cache if this revision has one."""
    module = importlib.import_module("konte.query_processor")
    clear = getattr(module, "clear_keyword_cache", None)
    if clear is None:
        return False
    clear()
    return True


def _measure_stall(ctx: Context) -> dict:
    """Time one extraction against an endpoint that never answers."""
    extract = _extractor()
    with stub_endpoint(stall=True) as (base_url, state), point_llm_at(base_url, MODEL_NAME):
        _clear_keyword_cache()

        start = time.perf_counter()
        keywords = extract(QUERY)
        elapsed = time.perf_counter() - start

    return {
        "seconds_until_fallback": elapsed,
        "requests_sent": state.requests,
        "fell_back_to_tokenization": bool(keywords),
    }


def _measure_cache(ctx: Context) -> dict:
    """Count requests for a repeated query versus distinct queries."""
    extract = _extractor()
    results = {}

    with stub_endpoint() as (base_url, state), point_llm_at(base_url, MODEL_NAME):
        _clear_keyword_cache()
        state.reset()
        durations = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            extract(QUERY)
            durations.append(time.perf_counter() - start)
        results["repeated_query"] = {
            "calls": REPEATS,
            "requests_sent": state.requests,
            "latency_ms": summarize(durations),
        }

        _clear_keyword_cache()
        state.reset()
        for index in range(REPEATS):
            extract(f"{QUERY} variant {index}")
        results["distinct_queries"] = {"calls": REPEATS, "requests_sent": state.requests}

    return results


def run(ctx: Context) -> dict:
    return {
        "has_extraction_cache": _clear_keyword_cache(),
        "stalled_endpoint": _measure_stall(ctx),
        "cache": _measure_cache(ctx),
    }
