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

### Changed

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
