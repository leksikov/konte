"""How a build overlaps its model calls, and what one rate limit costs.

Three things are measured, because "the build got faster" is not one claim:

``retry``
    A single rate-limit response used to put the whole segment back on the wire,
    resending answers that had already arrived. It should now retry the one
    chunk that failed. Measured as request count against an injected 429 - a
    count, not a latency, and not something a real endpoint can be asked for.

``context``
    Segments used to run strictly one after another, while the chunks inside a
    segment went out with no ceiling at all. They should now run concurrently
    under a single ceiling on requests in flight. Measured against a stub with a
    fixed round trip, so the change in scheduling is visible without an
    endpoint's variance on top.

``index``
    Embedding batches used to go out one at a time, each waiting for the last.
    They should now be kept in a sliding window. Measured with local embeddings
    and an injected round trip, for the same reason.

Pass ``endpoint=live`` to additionally build against the configured endpoint for
a real end-to-end wall-clock.
"""

from __future__ import annotations

import shutil
import time

from benchmarks.build import build_project
from benchmarks.corpus import document_excerpt, projects_dir, synthetic_document
from benchmarks.harness import Context, assert_live_endpoint, point_llm_at, stub_embeddings
from benchmarks.stub_endpoint import MODEL_NAME, stub_endpoint

CHUNKS_PER_SEGMENT = 10
RETRY_SEGMENTS = 2
OVERLAP_SEGMENTS = 12
INDEX_CHUNKS = 800
ROUND_TRIP = 0.25
EMBED_ROUND_TRIP = 0.20


def _fresh(storage, name: str):
    directory = storage / name
    if directory.exists():
        shutil.rmtree(directory)
    return directory


def _measure_retry(ctx: Context) -> dict:
    """Count requests when one chunk in a segment is rate limited once."""
    document = synthetic_document(RETRY_SEGMENTS * CHUNKS_PER_SEGMENT)
    storage = projects_dir(f"retry-{ctx.revision}")
    name = "retry_build"
    _fresh(storage, name)

    # Refuse one chunk's prompt often enough to exhaust the client library's own
    # retry budget. Below that threshold the client absorbs the 429 silently and
    # konte's retry path never runs at all - which is what makes a naive
    # "fail one request once" probe show no difference between the revisions.
    with (
        stub_endpoint(rate_limit_nth_prompt=4, rate_limit_attempts=3) as (
            base_url,
            state,
        ),
        point_llm_at(base_url, MODEL_NAME),
    ):
        project = build_project(name, storage, document, enable_faiss=False, skip_context=False)

    chunks = len(getattr(project, "_contextualized_chunks", []) or [])
    with_context = sum(
        1 for chunk in (getattr(project, "_contextualized_chunks", []) or []) if chunk.context
    )
    return {
        "chunks": chunks,
        "chunks_with_context": with_context,
        "rate_limit_responses": state.rate_limited,
        "total_requests": state.requests,
        "distinct_prompts": state.distinct_prompts,
        # One request per chunk is the floor. Everything above that, minus the
        # refusals themselves, is answers that were fetched more than once.
        "requests_resent": state.requests - chunks - state.rate_limited,
    }


def _measure_context(ctx: Context) -> dict:
    """Time context generation against a fixed per-request round trip."""
    document = synthetic_document(OVERLAP_SEGMENTS * CHUNKS_PER_SEGMENT)
    storage = projects_dir(f"overlap-{ctx.revision}")
    name = "overlap_build"
    _fresh(storage, name)

    with stub_endpoint(latency=ROUND_TRIP) as (base_url, state), point_llm_at(base_url, MODEL_NAME):
        start = time.perf_counter()
        project = build_project(name, storage, document, enable_faiss=False, skip_context=False)
        seconds = time.perf_counter() - start

    chunks = len(getattr(project, "_contextualized_chunks", []) or [])
    return {
        "chunks": chunks,
        "requests": state.requests,
        "round_trip_seconds": ROUND_TRIP,
        "build_seconds": seconds,
        # How much of the theoretical serial cost was actually overlapped.
        "effective_parallelism": (state.requests * ROUND_TRIP / seconds) if seconds else None,
    }


def _measure_index(ctx: Context) -> dict:
    """Time index construction against a fixed per-batch embedding round trip."""
    document = synthetic_document(INDEX_CHUNKS)
    storage = projects_dir(f"index-{ctx.revision}")
    name = "index_build"
    _fresh(storage, name)

    with stub_embeddings(latency=EMBED_ROUND_TRIP):
        start = time.perf_counter()
        project = build_project(name, storage, document, enable_faiss=True, skip_context=True)
        seconds = time.perf_counter() - start

    chunks = len(getattr(project, "_contextualized_chunks", []) or [])
    return {
        "chunks": chunks,
        "embed_round_trip_seconds": EMBED_ROUND_TRIP,
        "build_seconds": seconds,
    }


def _measure_live(ctx: Context) -> dict:
    """Build against the configured endpoint, end to end."""
    size = int(ctx.options.get("live_chunks", 200))
    document = document_excerpt(size)
    storage = projects_dir(f"live-{ctx.revision}")
    name = f"live_build_{size}"
    _fresh(storage, name)

    # A failed context call yields empty context rather than an error, so a
    # build aimed at the wrong endpoint still "succeeds" - slowly, having
    # generated nothing. Refuse to start rather than produce that number.
    endpoint = assert_live_endpoint()

    start = time.perf_counter()
    project = build_project(name, storage, document, enable_faiss=True, skip_context=False)
    seconds = time.perf_counter() - start

    chunks = getattr(project, "_contextualized_chunks", []) or []
    contextualized = sum(1 for chunk in chunks if chunk.context)
    if not contextualized:
        raise RuntimeError(
            f"live build generated context for 0 of {len(chunks)} chunks; "
            f"the wall-clock would measure retry exhaustion, not a build"
        )
    return {
        "requested_chunks": size,
        "chunks": len(chunks),
        "chunks_with_context": contextualized,
        "build_seconds": seconds,
        # Recorded so a stored result carries the endpoint it was measured
        # against; comparing two runs made against different servers is
        # otherwise indistinguishable from comparing two revisions.
        "endpoint": endpoint,
        "context_model": getattr(project._config, "context_model", None),
    }


def run(ctx: Context) -> dict:
    result = {
        "retry": _measure_retry(ctx),
        "context": _measure_context(ctx),
        "index": _measure_index(ctx),
    }
    if ctx.options.get("endpoint") == "live":
        result["live"] = _measure_live(ctx)
    return result
