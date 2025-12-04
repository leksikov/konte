# Contextual RAG Library - Tasks Checklist

## Setup

- [x] Initialize project structure
- [x] Create pyproject.toml with dependencies:
  - [x] langchain, langchain-openai, langchain-community
  - [x] faiss-cpu
  - [x] rank-bm25
  - [x] pydantic, pydantic-settings
  - [x] structlog
  - [x] tiktoken
  - [x] aiofiles
  - [x] pypdf
  - [x] pytest, pytest-asyncio
  - [x] python-dotenv
- [x] Create .env.example
- [x] Create prompts/context_prompt.txt (tariff-domain)
- [x] Create tests/fixtures/ with sample TXT, MD
- [ ] Create tests/fixtures/sample.pdf (optional - PDF loading works, just no test file)

## Step 1: Settings + Models

- [x] Write unit tests for settings validation
- [x] Write unit tests for model serialization
- [x] Implement settings.py with pydantic-settings
  - [x] OPENAI_API_KEY
  - [x] STORAGE_PATH
  - [x] EMBEDDING_MODEL (text-embedding-3-small)
  - [x] CONTEXT_MODEL (gpt-4.1)
  - [x] SEGMENT_SIZE (8000)
  - [x] SEGMENT_OVERLAP (800)
  - [x] CHUNK_SIZE (800)
  - [x] CHUNK_OVERLAP (80)
  - [x] CONTEXT_TOKENS (100-200)
  - [x] DEFAULT_TOP_K (20)
  - [x] MAX_CONCURRENT_CALLS (10)
  - [x] CONFIDENCE_THRESHOLD (0.7)
  - [x] PROMPT_PATH
- [x] Implement models.py
  - [x] Chunk
  - [x] ContextualizedChunk
  - [x] RetrievalResult
  - [x] RetrievalResponse
  - [x] ProjectConfig
- [x] Run tests, verify passing

## Step 2: Loader

- [x] Add sample TXT, MD to tests/fixtures/
- [x] Write unit tests for loader
- [x] Implement loader.py
  - [x] load_pdf()
  - [x] load_txt()
  - [x] load_md()
  - [x] load_document() (auto-detect type)
  - [x] Async versions (load_txt_async, load_md_async, load_document_async)
- [x] Run tests, verify passing

## Step 3: Chunker

- [x] Write unit tests for chunker
  - [x] Test segment splitting with overlap
  - [x] Test chunk splitting with overlap
  - [x] Test no word breakage
  - [x] Test create_chunks returns segments_map (data flow contract)
- [x] Implement chunker.py
  - [x] Token counting (tiktoken)
  - [x] segment_document() - split into 8000 token segments with 800 overlap
  - [x] chunk_segment() - split into 800 token chunks with 80 overlap
  - [x] Word boundary adjustment
  - [x] create_chunks() - full pipeline returning (Chunk objects, segments_map)
- [x] Run tests, verify passing

## Step 4: Context Generator

- [x] Create tariff-domain prompt in prompts/context_prompt.txt
- [x] Write integration tests (real OpenAI calls)
- [x] Implement context.py
  - [x] Load prompt template from file
  - [x] Structure prompt: segment first (cacheable), then chunk
  - [x] generate_context() - single chunk, returns plain text
  - [x] generate_contexts_batch() - uses LangChain abatch() for parallel processing
  - [x] Reuse ChatOpenAI instance (module-level cache for prompt caching)
  - [x] Timeout handling (via LangChain)
  - [x] Rate limit handling (via LangChain max_retries)
  - [x] Skip option (return empty context)
- [x] Run integration tests, verify passing

## Step 5: FAISS Store

- [x] Write integration tests
- [x] Implement faiss_store.py
  - [x] build_index() - from contextualized chunks
  - [x] save() - persist to disk
  - [x] load() - load from disk
  - [x] query() - return top-k with scores (normalized 0-1)
  - [x] is_empty property
- [x] Run tests, verify passing

## Step 6: BM25 Store

- [x] Write unit tests
- [x] Implement bm25_store.py
  - [x] build_index() - from contextualized text
  - [x] save() - pickle to disk
  - [x] load() - load from disk
  - [x] query() - return top-k with scores (normalized 0-1)
  - [x] is_empty property
- [x] Run tests, verify passing

## Step 7: Retriever

- [x] Write unit tests for fusion logic
- [x] Implement retriever.py
  - [x] reciprocal_rank_fusion()
  - [x] retrieve_semantic() - FAISS only
  - [x] retrieve_lexical() - BM25 only
  - [x] retrieve_hybrid() - both + fusion
  - [x] retrieve() - unified interface with mode parameter
  - [x] Fallback logic with warning
  - [x] Build RetrievalResponse with agent hints
    - [x] Calculate top_score
    - [x] Calculate score_spread
    - [x] Determine has_high_confidence (top_score >= 0.7)
    - [x] Determine suggested_action:
      - [x] "deliver" if top_score >= 0.7
      - [x] "query_more" if 0.4 <= top_score < 0.7
      - [x] "refine_query" if top_score < 0.4
- [x] Run tests, verify passing

## Step 8: Project

- [x] Write integration tests
- [x] Implement project.py
  - [x] Project class
  - [x] add_documents() - load + segment + chunk
  - [x] build() - async, context generation + indexing
    - [x] Option to skip context
    - [x] Option to enable/disable FAISS
    - [x] Option to enable/disable BM25
  - [x] query() - retrieval, default top_k=20
  - [x] as_retriever() - callable for Agno
  - [x] save() - persist project state
  - [x] load() - load project state
  - [x] Project.create() - factory method
  - [x] Project.open() - load existing project
- [x] Data flow contract tests (integration)
  - [x] Test segments stored are actual segments, not full document
  - [x] Test each chunk maps to valid segment in _segments
- [x] Run tests, verify passing

## Step 9: Manager

- [x] Write integration tests
- [x] Implement manager.py
  - [x] create_project()
  - [x] list_projects()
  - [x] get_project()
  - [x] delete_project()
  - [x] project_exists()
- [x] Run tests, verify passing

## Step 10: Public API

- [x] Implement __init__.py with clean exports
- [x] Create examples/basic_usage.py
- [ ] Create examples/agno_integration.py
- [x] End-to-end test: ingest → build → query (see tests/e2e/)

## Documentation

- [x] Update README.md with correct API examples
- [x] Docstrings on all public functions
- [x] Update PRD.md and PLAN.md

## Final Verification

- [x] All unit tests passing (74 tests)
- [x] All integration tests passing (44 tests)
- [x] All e2e tests passing (18 tests)
- [ ] Example scripts work
- [ ] Test with real tariff documents

---

## E2E Tests (tests/e2e/)

End-to-end tests with real API calls validating complete workflows.

- [x] Create tests/e2e/ directory
- [x] test_full_pipeline.py (4 tests)
  - [x] Test: Ingest document → build with context → query → verify results
  - [x] Test: Verify segment sizes are bounded (not full document)
  - [x] Test: Verify context generation produces valid context text
  - [x] Test: Verify retrieval returns relevant results for known queries
  - [x] Test: Verify suggested_action computed correctly
- [x] test_persistence.py (7 tests)
  - [x] Test: Save project → reload → query returns same results
  - [x] Test: Verify config preserved after reload
  - [x] Test: Verify chunks preserved after reload
  - [x] Test: Verify FAISS index persisted and loadable
  - [x] Test: Verify BM25 index persisted and loadable
  - [x] Test: Verify both indexes work for hybrid mode
  - [x] Test: Verify context preserved after reload
- [x] test_large_document.py (5 tests)
  - [x] Test: Process 50k token document
  - [x] Test: Verify segments bounded for large document
  - [x] Test: Verify all chunks have valid segment reference
  - [x] Test: Process 10k document with context generation
  - [x] Test: Save/load large document project
- [x] pytest markers: `@pytest.mark.e2e` for slow, API-dependent tests
- [ ] Add to CI with `pytest tests/e2e/ -v --timeout=300`

**Total: 18 e2e tests (14 fast, 4 with context generation)**

---

## Known Issues / Notes

1. **No sample.pdf in fixtures**: PDF loading is implemented and tested for errors, but no actual PDF test file exists. This is optional since the pypdf library works.

2. ~~**Segment storage in Project**: Currently storing full document text for each segment index in `_segments` dict - works but could be memory-inefficient for large documents.~~ **FIXED**: Now `create_chunks()` returns segments map with actual segment text, not full document.

3. **BM25 score normalization**: Fixed issue where BM25 could return negative scores. Now uses min-max normalization to ensure 0-1 range.

4. **LangChain batch vs OpenAI Batch API**: Using LangChain's `abatch()` for parallel processing (immediate results). OpenAI's Batch API is different - 24hr latency for 50% cost savings, not suitable for real-time use.

5. **Rate limit handling**: Added exponential backoff (1s, 2s, 4s, 8s, 16s up to 60s max) with 5 retries for OpenAI 429 errors in `context.py`.

6. **Chunker word boundary bug**: Fixed bug in `_split_by_tokens()` where word boundary adjustment was using `prefix_text[:boundary]` (from document start) instead of `chunk_text[:boundary]` (within the chunk). This caused segments to accumulate content from the beginning of the document, growing progressively larger.
