# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `use_keyword_extraction` on `Project.query()`, `query_async()`,
  `query_with_answer()`, every `Retriever` entry point, the `/query` and `/ask`
  request bodies, the `konte query`/`konte ask` CLI, and the Gradio UI. Unset
  follows the new `BM25_KEYWORD_EXTRACTION` setting, so BM25 keyword extraction
  can be turned off per call or per deployment instead of being unconditional
- `Retriever.retrieve_async()`, the non-blocking counterpart to `retrieve()`
- `clear_keyword_cache()` and `extract_search_keywords_async()` are now exported
- `max_retries` on `get_llm()`, for callers that cannot afford a retry storm
- `get_shared_project()`, a process-wide cache of projects opened for querying,
  with `preload_projects()`, `invalidate_project()` and `clear_project_cache()`
  alongside it. `get_project()` keeps returning a private, mutable instance
- `PROJECT_CACHE_SIZE` (default 4) and `PRELOAD_PROJECTS` settings, bounding how
  many projects a server holds and which ones it opens at startup

### Changed

- The API and the UI serve every request from a cached project instead of
  reading the whole project back from disk each time. On a 20k-chunk project a
  request spent ~330ms reopening before answering; it now costs a dictionary
  lookup, and the FastAPI routes take the project through a dependency so a cold
  open runs in a worker thread rather than on the event loop
- Build checkpoints are an append-only log (`context_checkpoint.jsonl`), one
  line per finished segment, replacing a whole-file rewrite after every segment
  that wrote each chunk once per remaining segment. A 500-segment build wrote
  ~1.7GB of checkpoints; it now writes ~7MB. A checkpoint in the old format is
  converted on first read, so an interrupted build still resumes
- Opening a project reads its indexes only. The stored chunks and segments are
  parsed on first use, which a process that only answers queries never triggers
- `bm25.pkl` no longer stores the tokenized corpus: BM25Okapi scores from the
  term frequencies it already holds, so that was a second copy of the corpus to
  write and unpickle (~25MB → ~2MB on a 20k-chunk project). Existing index files
  still load; one written by this version cannot be read by an earlier release

- A filtered FAISS query resolves its filters through an inverted index over the
  docstore instead of reading every stored document and testing it. The index is
  built on the first filtered query and dropped whenever the vectorstore is
  replaced, so an unfiltered workload never pays for it. On a 100k-chunk project
  selecting the matching ids cost 16–42ms per query; it now costs 0.05–0.4ms,
  for ~30MB of postings. A filter on a field whose values cannot be hashed still
  falls back to the scan
- The BM25 query path ranks with NumPy rather than four full passes over the
  corpus in Python: an unfiltered query no longer materializes an N-element
  candidate list, the top k comes from a linear partition instead of sorting
  every chunk through a Python comparison key, and the normalization bounds come
  from the score array directly. On 100k chunks an unfiltered query went 84ms →
  68ms, leaving 3.6ms above what scoring itself costs. Results are unchanged,
  ties included
- Keyword extraction runs on `KEYWORD_EXTRACTION_TIMEOUT` (5s) without retries
  instead of the 120s context-generation budget with two retries, cutting the
  worst case a stalled endpoint can impose on one query from ~360s to 5s
- Extraction results are cached per query and shared by the sync and async
  paths, so a repeated or refined search no longer repeats the round trip
- `query_async()` and `query_with_answer()` await keyword extraction instead of
  blocking the event loop, which had stalled every other request in an ASGI
  worker for the duration of the call

### Fixed

- An extraction that returned no keywords searched BM25 for the empty string,
  scoring every chunk zero and returning whichever ones came first; the
  original query is now used instead
- Project files were written in place, so a build interrupted mid-write left a
  truncated file that the next resume could not parse — the exact accident
  checkpointing exists to prevent. Every file is now written beside its target
  and moved into place in one step, and a checkpoint log with a half-written
  last line drops that line instead of failing

## [0.1.0] - 2026-07-29

### Added

- Contextual retrieval pipeline: segment (~8000 tokens) → chunk (800 tokens) → LLM-generated context prepended to each chunk before indexing
- Hybrid retrieval: FAISS (semantic) + BM25 (lexical) combined via reciprocal rank fusion, with `semantic` and `lexical` single-index modes
- `Project` API: create, open, add documents (PDF/TXT/MD), build with per-segment checkpointing, save/load, query
- Full RAG answer generation via `query_with_answer()` with custom prompt templates
- Optional reranking through any vLLM `/score` endpoint (`RERANKER_BASE_URL`)
- Metadata filtering (`metadata_filter`, `source_filter`) and multi-project retrieval
- Agent integration: `suggested_action` confidence hints, `as_retriever()` callable, LangChain/Agno guide
- Typer + Rich CLI: `create`, `add`, `build`, `query`, `ask`, `list`, `info`, `delete`, `serve` (FastAPI), `ui` (Gradio)
- Prefix-caching-aware context generation (sequential segments, parallel chunks per segment)
- Structured logging with structlog
- Evaluation harness (DeepEval synthesizer + GEval LLM-as-judge metrics)

### Changed

- `dev` and `eval` installs moved from package extras to PEP 735 dependency
  groups: use `uv sync` / `uv sync --group eval` (or pip ≥ 25.1:
  `pip install --group dev`). `pip install -e ".[dev]"` no longer installs
  anything and pip only warns about the unknown extra.

[0.1.0]: https://github.com/leksikov/konte/releases/tag/v0.1.0
