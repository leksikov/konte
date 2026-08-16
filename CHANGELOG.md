# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Index files are signed (HMAC-SHA256) when written and authenticated before
  they are read. Opening a project deserializes `faiss.pkl` and `bm25.pkl`, and
  deserializing runs whatever the file holds, so anyone who could write into
  `STORAGE_PATH` could run code as the process that opened the project. An index
  that fails the check raises `IntegrityError` instead of being loaded. The key
  is generated on first use and kept in the storage root as `.signing-key`
  (mode 0600), outside the project directories it signs; `INDEX_SIGNING_KEY`
  supplies it from the environment instead, for a shared storage root
- Indexes written by earlier versions carry no signature and no longer load.
  `konte trust <project>` / `trust_project()` signs the files already on disk
  for installations that trust them; anything else has to be rebuilt

### Added

- `use_keyword_extraction` on `Project.query()`, `query_async()`,
  `query_with_answer()`, every `Retriever` entry point, the `/query` and `/ask`
  request bodies, and the `konte query`/`konte ask` CLI. Unset
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
- `FAISSStore.abuild_index()`, the concurrent counterpart to `build_index()`

### Changed

- The API serves every request from a cached project instead of
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
- A project keeps one copy of its chunk text where it kept three. `bm25_chunks.json`
  is gone — it was `chunks.json` written a second time under another name, and the
  lexical index now ranks the project's own chunks, reading them on the first
  lexical query rather than at open. The FAISS docstore no longer repeats the
  chunk and its context in metadata either: the page content already holds both
  joined, so only the length of the cut is stored beside it. On a 2,000-chunk
  corpus the project directory went from 4.97x to 2.94x the size of the text in
  it, a 41% cut, and neither copy was ever read back. Indexes written before this
  still load, both halves of their payload included
- `BM25Store.load()` takes the corpus its index was built over; it no longer owns
  a file of its own to read one from. `Store` no longer declares `load()` at all,
  since the two stores legitimately need different things to come back
- Context generation runs its segments concurrently instead of one after another,
  against a single ceiling on requests in flight. `MAX_CONCURRENT_CALLS` is what
  sets that ceiling — the setting existed but nothing read it, so a build was
  paced by the sum of its segments' round trips while each segment's chunks went
  out with no limit at all. It now defaults to 16 rather than a nominal 1, and a
  build costs its total request volume divided by that number. Chunks stay in
  segment order however the responses arrive, and each segment still checkpoints
  as it lands, so an interrupted build resumes exactly as before
- A chunk that hits a rate limit retries on its own. The whole segment used to go
  back on the wire together, so one 429 among twenty chunks resent nineteen
  answers that had already arrived — and a segment that exhausted its retries
  lost the context for every chunk in it, not just the one that failed
- Building the FAISS index embeds batches concurrently under the same ceiling,
  keeping requests on the wire in a sliding window instead of sending the next
  one only after the last came back. Batches are folded into the index in the
  order they were submitted, so the index does not depend on which response
  arrived first, and only the window's vectors are held before FAISS packs them

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

### Removed

- The Gradio UI: the `konte.ui` package, the `konte ui` command, and the `ui`
  and `server` extras. It was a demo front end over the public API that nothing
  else imported, and the library ships without it. `server` was `api` plus
  Gradio, so `api` is now the only server extra
- `ProjectConfig.context_min_tokens` and `context_max_tokens`, and the
  `CONTEXT_MIN_TOKENS` / `CONTEXT_MAX_TOKENS` settings behind them. Nothing read
  any of the four — context length comes from the prompt template — so setting
  them only wrote a number into `config.json` and the `/config` response. A
  `config.json` carrying them still opens

### Fixed

- `fusion_weight_semantic` and `fusion_weight_lexical` were persisted in
  `config.json`, shown in the UI's config tab and drawn into the architecture
  diagram, but no retrieval path read them. Fusion is now weighted,
  `sum(weight / (k + rank))`, so their ratio decides which index wins a
  disagreement. Existing projects rank identically — a factor shared by both
  lists cancels in the rescaling, so the 0.5/0.5 default is the old behavior
- The keyword-extraction prompt asked for a compound term to be kept "as one
  keyword", but the keywords are joined into one string that BM25 re-tokenizes
  on word and CJK-bigram boundaries, so no phrase was ever matched as a unit.
  It now asks for the modifier to survive ("working capital", not "capital")
- `top_score`, `score_spread`, `has_high_confidence` and `suggested_action` were
  read off the ranking scores, which measure nothing. Rank fusion rescales its
  winner to 1.0 and BM25 normalizes against its own candidates, so in `hybrid`
  and `lexical` mode every response holding at least one result reported
  `top_score` 1.0 and `suggested_action` "deliver" — a query matching a single
  stopword scored exactly as high as one answered outright, and the agent hints
  the README documents were constants. They are now computed from a measure of
  the query against the chunk: vector similarity, the share of the query's
  terms the lexical index matched, or the reranker's score when reranking
  actually ran. Result ordering and `RetrievalResult.score` are unchanged, and
  `semantic` mode — where the ranking score was already absolute — is
  unaffected
- A reranker that could not be reached returned the initial retrieval's ranking
  scores, so an outage was reported as a perfect match. `rerank_chunks_with_score()`
  now says whether it scored anything, and the retrieval's own measurements
  stand in when it did not
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
