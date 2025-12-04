# Contextual RAG Library - Implementation Plan

## Development Principles

- KISS, YAGNI, DRY priority over SOLID
- Pydantic V2 for all models (no dataclass)
- Async-first, no threads
- pathlib, no hardcoded paths
- structlog for logging
- LangChain v0.3 for LLM and embeddings
- Imports at top of file only
- Minimal nesting, minimal abstraction

## Package Structure

```
contextual_rag/
├── __init__.py          # Public API exports
├── settings.py          # pydantic-settings, env vars, SSoT for config
├── models.py            # All Pydantic models
├── loader.py            # Document loading (PDF, TXT, MD)
├── chunker.py           # Segment + chunk documents
├── context.py           # LLM context generation
├── faiss_store.py       # FAISS index operations
├── bm25_store.py        # BM25 index operations
├── retriever.py         # Hybrid retrieval + rank fusion
├── project.py           # Project class (main interface)
└── manager.py           # Create/list/load/delete projects

prompts/
└── context_prompt.txt   # Tariff-domain context generation prompt

tests/
├── unit/                # Mocks allowed here only
├── integration/         # Real API calls, real files
└── fixtures/            # Test documents

examples/                # Usage examples
```

Flat structure. No nested modules unless truly needed.

## Models (models.py)

**Chunk**: chunk_id, content, metadata (source, segment_idx, chunk_idx)

**ContextualizedChunk**: chunk + context + contextualized_content

**RetrievalResult**: 
- content: str
- context: str  
- score: float (0-1 normalized)
- source: str (filename)
- chunk_id: str
- metadata: dict

**RetrievalResponse** (for agent decision-making):
- results: list[RetrievalResult]
- query: str
- total_found: int
- top_score: float
- score_spread: float (difference between top and bottom scores)
- has_high_confidence: bool (top_score > CONFIDENCE_THRESHOLD)
- suggested_action: str

suggested_action logic:
- "deliver": top_score >= 0.7 (high confidence, results are good)
- "query_more": 0.4 <= top_score < 0.7 (medium confidence, might need more context)
- "refine_query": top_score < 0.4 (low confidence, query may be off-topic)

**ProjectConfig**: name, storage_path, segment_size, segment_overlap, chunk_size, chunk_overlap, context_tokens, embedding_model, context_model, enable_faiss, enable_bm25, fusion_weights

## Settings (settings.py)

Single source of truth for:
- OPENAI_API_KEY (from env)
- STORAGE_PATH (from env, default ~/.contextual_rag)
- EMBEDDING_MODEL: text-embedding-3-small
- CONTEXT_MODEL: gpt-4.1
- SEGMENT_SIZE: 8000 tokens
- SEGMENT_OVERLAP: 800 tokens (10%)
- CHUNK_SIZE: 800 tokens
- CHUNK_OVERLAP: 80 tokens (10%)
- CONTEXT_TOKENS: 100-200
- DEFAULT_TOP_K: 20
- MAX_CONCURRENT_CALLS: 10
- PROMPT_PATH: prompts/context_prompt.txt
- CONFIDENCE_THRESHOLD: 0.7 (for suggested_action logic)

Use pydantic-settings with env file support.

## Storage

```
{STORAGE_PATH}/
└── {project_name}/
    ├── config.json       # ProjectConfig
    ├── chunks.json       # ContextualizedChunk list
    ├── faiss.index       # FAISS index (if enabled)
    ├── faiss_ids.json    # ID mapping
    └── bm25.pkl          # BM25 index (if enabled)
```

## Implementation Order

### Step 1: Settings + Models
- settings.py with pydantic-settings
- models.py with all Pydantic models
- Write unit tests first

### Step 2: Loader
- Load PDF (pypdf or LangChain)
- Load TXT, MD
- Return document text + metadata
- Tests with sample files in tests/fixtures/

### Step 3: Chunker
- Token counting with tiktoken
- Segment large docs into ~8000 token units with 800 token overlap
- Chunk segments into 800 token pieces with 80 token overlap
- No word breakage at boundaries (adjust to word boundaries)
- Unit tests

### Step 4: Context Generator
- Async LLM calls via LangChain
- Context is plain text output (no structured parsing needed)
- Prompt structure: segment first (cacheable), then chunk (variable) - enables OpenAI prompt caching
- Use LangChain `abatch()` for parallel processing (not OpenAI Batch API which has 24hr latency)
- Module-level ChatOpenAI cache to reuse instances (enables prompt caching)
- Timeout handling
- Rate limit handling with exponential backoff (via LangChain max_retries)
- Load prompt template from prompts/ directory
- Skip option for standard RAG mode
- Integration tests with real OpenAI calls

### Step 5: FAISS Store
- Build index from embeddings
- Save/load with pathlib
- Query top-k
- Optional (can be disabled)
- Integration tests

### Step 6: BM25 Store
- Build index from text
- Save/load (pickle)
- Query top-k
- Optional (can be disabled)
- Unit tests (no external API)

### Step 7: Retriever
- Reciprocal Rank Fusion
- Hybrid mode: FAISS + BM25 + fusion
- Semantic mode: FAISS only
- Lexical mode: BM25 only
- Warn and fallback if requested mode unavailable
- Return RetrievalResponse with agent decision hints
- Unit tests for fusion logic

### Step 8: Project
- Main interface class
- add_documents() → load + segment + chunk
- build() → context + indexes
- query() → retrieval (default top_k=20)
- as_retriever() → callable for Agno
- Integration tests

### Step 9: Manager
- create_project()
- list_projects()
- get_project()
- delete_project()
- Integration tests

### Step 10: Public API
- Clean exports in __init__.py
- Example scripts in examples/

## Resolved Decisions

1. **Chunk overlap**: 80 tokens (10%), no word breakage
2. **Segment overlap**: 800 tokens (10%), no word breakage
3. **Embedding model**: text-embedding-3-small
4. **Context model**: gpt-4.1
5. **Batch processing**: LangChain `abatch()` for parallel LLM calls (immediate results). Note: OpenAI Batch API is different - 24hr latency for 50% cost savings, not used here.
6. **LLM instance caching**: Module-level cache for ChatOpenAI to enable OpenAI prompt caching
7. **Hybrid fallback**: Warn and fallback to available index
8. **Adding documents**: Full rebuild (append later)
9. **Prompts**: Stored in prompts/ directory, tariff-domain focused

## Test Strategy

**Unit tests (tests/unit/)**:
- Mocks for external services
- Test chunker logic, fusion algorithm, config validation

**Integration tests (tests/integration/)**:
- Real OpenAI API calls
- Real file I/O
- End-to-end: ingest → build → query

**Test fixtures (tests/fixtures/)**:
- Sample PDF, TXT, MD files
- Expected outputs for validation

## Dependencies

**Core**:
- langchain + langchain-openai + langchain-community
- faiss-cpu
- rank-bm25
- pydantic + pydantic-settings
- structlog
- tiktoken (token counting)
- aiofiles (async file I/O)

**Document loading**:
- pypdf (PDF)
- python-docx (DOCX, optional)

**Dev**:
- pytest + pytest-asyncio
- python-dotenv