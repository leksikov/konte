"""What it costs to get hold of a project before answering a query.

Two claims:

- opening a project used to parse its whole corpus back into models; it should
  now read the indexes only, and parse the corpus when something first needs it
- a server used to reopen the project for every request; it should now serve
  requests from a process-wide cache, at the price of a staleness check

The older revision has no cache at all, so the honest comparison is: what does
the ordinary lookup cost on each revision, and what does the cached lookup cost
where one exists. Both are reported rather than collapsed into one number.

Runs against a copy of a real built project. Index deserialization time depends
on real index size, which is the thing being avoided per request.
"""

from __future__ import annotations

import importlib
import time

from benchmarks.build import open_project
from benchmarks.corpus import available_real_projects, real_project
from benchmarks.harness import Context, dir_bytes, peak_rss_mb, pin_keyword_extraction, summarize

DEFAULT_PROJECT = "all_tariff_documents"
OPEN_TRIALS = 5
LOOKUP_TRIALS = 20


def _shared_lookup():
    """Return this revision's cached project lookup, if it has one."""
    konte = importlib.import_module("konte")
    return getattr(konte, "get_shared_project", None)


def run(ctx: Context) -> dict:
    name = ctx.options.get("project", DEFAULT_PROJECT)
    if name not in available_real_projects():
        return {"status": "skipped", "reason": f"no built project {name!r}"}

    storage, project_name = real_project(name, ctx.revision, reindex_lexical=True)
    konte = importlib.import_module("konte")
    pin_keyword_extraction()

    rss_start = peak_rss_mb()

    # Plain open, repeated. This is what a server did per request before there
    # was a cache, and what get_project still does on both revisions.
    opens = []
    for _ in range(OPEN_TRIALS):
        start = time.perf_counter()
        project = open_project(project_name, storage)
        opens.append(time.perf_counter() - start)
    rss_after_open = peak_rss_mb()

    # Whether opening alone materialized the corpus, or left it for first use.
    # Read through __dict__ rather than getattr: on the newer revision the
    # corpus attributes are descriptors that parse on access, so getattr would
    # perform the very work it is trying to detect and always answer "yes".
    instance = vars(project)
    corpus_parsed_at_open = {
        name: name in instance for name in ("_chunks", "_segments", "_contextualized_chunks")
    }

    start = time.perf_counter()
    project.query("classification of parts", mode="lexical", top_k=10)
    first_query_seconds = time.perf_counter() - start
    rss_after_query = peak_rss_mb()

    result: dict = {
        "project": name,
        "project_dir_bytes": dir_bytes(storage / name),
        "open_ms": summarize(opens),
        "first_lexical_query_ms": first_query_seconds * 1000,
        "corpus_parsed_during_open": corpus_parsed_at_open,
        "rss_mb": {
            "start": rss_start,
            "after_opens": rss_after_open,
            "after_first_query": rss_after_query,
        },
    }

    # get_project: present on both revisions, reopens every time.
    plain = []
    for _ in range(OPEN_TRIALS):
        start = time.perf_counter()
        konte.get_project(project_name, storage_path=storage)
        plain.append(time.perf_counter() - start)
    result["get_project_ms"] = summarize(plain)

    shared = _shared_lookup()
    if shared is None:
        result["get_shared_project_ms"] = {"status": "unsupported"}
        return result

    shared(project_name, storage_path=storage)  # prime
    cached = []
    for _ in range(LOOKUP_TRIALS):
        start = time.perf_counter()
        shared(project_name, storage_path=storage)
        cached.append(time.perf_counter() - start)
    result["get_shared_project_ms"] = summarize(cached)
    return result
