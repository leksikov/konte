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
- [x] Create tests/fixtures/ with sample PDF, TXT, MD

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

- [x] Add sample PDF, TXT, MD to tests/fixtures/
- [x] Write unit tests for loader
- [x] Implement loader.py
  - [x] load_pdf()
  - [x] load_txt()
  - [x] load_md()
  - [x] load_document() (auto-detect type)
- [x] Run tests, verify passing

## Step 3: Chunker

- [x] Write unit tests for chunker
  - [x] Test segment splitting with overlap
  - [x] Test chunk splitting with overlap
  - [x] Test no word breakage
- [x] Implement chunker.py
  - [x] Token counting (tiktoken)
  - [x] segment_document() - split into 8000 token segments with 800 overlap
  - [x] chunk_segment() - split into 800 token chunks with 80 overlap
  - [x] Word boundary adjustment
- [x] Run tests, verify passing

## Step 4: Context Generator

- [x] Create tariff-domain prompt in prompts/context_prompt.txt
- [x] Write integration tests (real OpenAI calls)
- [x] Implement context.py
  - [x] Load prompt template from file
  - [x] Structure prompt: segment first (cacheable), then chunk
  - [x] generate_context() - single chunk, returns plain text
  - [x] generate_contexts_batch() - async, 10 parallel
  - [x] Timeout handling
  - [x] Rate limit handling with exponential backoff
  - [x] Skip option (return empty context)
- [x] Run integration tests, verify passing

## Step 5: FAISS Store

- [x] Write integration tests
- [x] Implement faiss_store.py
  - [x] build_index() - from contextualized chunks
  - [x] save() - persist to disk
  - [x] load() - load from disk
  - [x] query() - return top-k with scores
- [x] Run tests, verify passing

## Step 6: BM25 Store

- [x] Write unit tests
- [x] Implement bm25_store.py
  - [x] build_index() - from contextualized text
  - [x] save() - pickle to disk
  - [x] load() - load from disk
  - [x] query() - return top-k with scores
- [x] Run tests, verify passing

## Step 7: Retriever

- [x] Write unit tests for fusion logic
- [x] Implement retriever.py
  - [x] reciprocal_rank_fusion()
  - [x] retrieve_semantic() - FAISS only
  - [x] retrieve_lexical() - BM25 only
  - [x] retrieve_hybrid() - both + fusion
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
  - [x] build() - context generation + indexing
    - [x] Option to skip context
    - [x] Option to enable/disable FAISS
    - [x] Option to enable/disable BM25
  - [x] query() - retrieval, default top_k=20
  - [x] as_retriever() - callable for Agno
  - [x] save() - persist project state
  - [x] load() - load project state
- [x] Run tests, verify passing

## Step 9: Manager

- [x] Write integration tests
- [x] Implement manager.py
  - [x] create_project()
  - [x] list_projects()
  - [x] get_project()
  - [x] delete_project()
- [x] Run tests, verify passing

## Step 10: Public API

- [x] Implement __init__.py with clean exports
- [ ] Create examples/basic_usage.py
- [ ] Create examples/agno_integration.py
- [ ] End-to-end test: ingest → build → query

## Documentation

- [ ] README.md with installation and usage
- [ ] Docstrings on all public functions
- [ ] Update PRD.md and PLAN.md if needed

## Final Verification

- [x] All unit tests passing (70 tests)
- [x] All integration tests passing (42 tests)
- [ ] Example scripts work
- [ ] Test with real tariff documents
