# Contextual RAG Library - Tasks Checklist

## Setup

- [ ] Initialize project structure
- [ ] Create pyproject.toml with dependencies:
  - [ ] langchain, langchain-openai, langchain-community
  - [ ] faiss-cpu
  - [ ] rank-bm25
  - [ ] pydantic, pydantic-settings
  - [ ] structlog
  - [ ] tiktoken
  - [ ] aiofiles
  - [ ] pypdf
  - [ ] pytest, pytest-asyncio
  - [ ] python-dotenv
- [ ] Create .env.example
- [ ] Create prompts/context_prompt.txt (tariff-domain)
- [ ] Create tests/fixtures/ with sample PDF, TXT, MD

## Step 1: Settings + Models

- [ ] Write unit tests for settings validation
- [ ] Write unit tests for model serialization
- [ ] Implement settings.py with pydantic-settings
  - [ ] OPENAI_API_KEY
  - [ ] STORAGE_PATH
  - [ ] EMBEDDING_MODEL (text-embedding-3-small)
  - [ ] CONTEXT_MODEL (gpt-4.1)
  - [ ] SEGMENT_SIZE (8000)
  - [ ] SEGMENT_OVERLAP (800)
  - [ ] CHUNK_SIZE (800)
  - [ ] CHUNK_OVERLAP (80)
  - [ ] CONTEXT_TOKENS (100-200)
  - [ ] DEFAULT_TOP_K (20)
  - [ ] MAX_CONCURRENT_CALLS (10)
  - [ ] CONFIDENCE_THRESHOLD (0.7)
  - [ ] PROMPT_PATH
- [ ] Implement models.py
  - [ ] Chunk
  - [ ] ContextualizedChunk
  - [ ] RetrievalResult
  - [ ] RetrievalResponse
  - [ ] ProjectConfig
- [ ] Run tests, verify passing

## Step 2: Loader

- [ ] Add sample PDF, TXT, MD to tests/fixtures/
- [ ] Write unit tests for loader
- [ ] Implement loader.py
  - [ ] load_pdf()
  - [ ] load_txt()
  - [ ] load_md()
  - [ ] load_document() (auto-detect type)
- [ ] Run tests, verify passing

## Step 3: Chunker

- [ ] Write unit tests for chunker
  - [ ] Test segment splitting with overlap
  - [ ] Test chunk splitting with overlap
  - [ ] Test no word breakage
- [ ] Implement chunker.py
  - [ ] Token counting (tiktoken)
  - [ ] segment_document() - split into 8000 token segments with 800 overlap
  - [ ] chunk_segment() - split into 800 token chunks with 80 overlap
  - [ ] Word boundary adjustment
- [ ] Run tests, verify passing

## Step 4: Context Generator

- [ ] Create tariff-domain prompt in prompts/context_prompt.txt
- [ ] Write integration tests (real OpenAI calls)
- [ ] Implement context.py
  - [ ] Load prompt template from file
  - [ ] Structure prompt: segment first (cacheable), then chunk
  - [ ] generate_context() - single chunk, returns plain text
  - [ ] generate_contexts_batch() - async, 10 parallel
  - [ ] Timeout handling
  - [ ] Rate limit handling with exponential backoff
  - [ ] Skip option (return empty context)
- [ ] Run integration tests, verify passing

## Step 5: FAISS Store

- [ ] Write integration tests
- [ ] Implement faiss_store.py
  - [ ] build_index() - from contextualized chunks
  - [ ] save() - persist to disk
  - [ ] load() - load from disk
  - [ ] query() - return top-k with scores
- [ ] Run tests, verify passing

## Step 6: BM25 Store

- [ ] Write unit tests
- [ ] Implement bm25_store.py
  - [ ] build_index() - from contextualized text
  - [ ] save() - pickle to disk
  - [ ] load() - load from disk
  - [ ] query() - return top-k with scores
- [ ] Run tests, verify passing

## Step 7: Retriever

- [ ] Write unit tests for fusion logic
- [ ] Implement retriever.py
  - [ ] reciprocal_rank_fusion()
  - [ ] retrieve_semantic() - FAISS only
  - [ ] retrieve_lexical() - BM25 only
  - [ ] retrieve_hybrid() - both + fusion
  - [ ] Fallback logic with warning
  - [ ] Build RetrievalResponse with agent hints
    - [ ] Calculate top_score
    - [ ] Calculate score_spread
    - [ ] Determine has_high_confidence (top_score >= 0.7)
    - [ ] Determine suggested_action:
      - [ ] "deliver" if top_score >= 0.7
      - [ ] "query_more" if 0.4 <= top_score < 0.7
      - [ ] "refine_query" if top_score < 0.4
- [ ] Run tests, verify passing

## Step 8: Project

- [ ] Write integration tests
- [ ] Implement project.py
  - [ ] Project class
  - [ ] add_documents() - load + segment + chunk
  - [ ] build() - context generation + indexing
    - [ ] Option to skip context
    - [ ] Option to enable/disable FAISS
    - [ ] Option to enable/disable BM25
  - [ ] query() - retrieval, default top_k=20
  - [ ] as_retriever() - callable for Agno
  - [ ] save() - persist project state
  - [ ] load() - load project state
- [ ] Run tests, verify passing

## Step 9: Manager

- [ ] Write integration tests
- [ ] Implement manager.py
  - [ ] create_project()
  - [ ] list_projects()
  - [ ] get_project()
  - [ ] delete_project()
- [ ] Run tests, verify passing

## Step 10: Public API

- [ ] Implement __init__.py with clean exports
- [ ] Create examples/basic_usage.py
- [ ] Create examples/agno_integration.py
- [ ] End-to-end test: ingest → build → query

## Documentation

- [ ] README.md with installation and usage
- [ ] Docstrings on all public functions
- [ ] Update PRD.md and PLAN.md if needed

## Final Verification

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Example scripts work
- [ ] Test with real tariff documents
