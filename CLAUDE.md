# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Konte** - Contextual RAG library that improves retrieval by prepending LLM-generated context to chunks before embedding and indexing. Implements hybrid retrieval (FAISS semantic + BM25 lexical) with reciprocal rank fusion.

## Commands

```bash
# Install dependencies
pip install -e .

# Run all tests
pytest

# Run unit tests only (70 tests, no API calls)
pytest tests/unit/

# Run integration tests (42 tests, requires OPENAI_API_KEY)
pytest tests/integration/

# Run single test file
pytest tests/unit/test_chunker.py

# Run single test
pytest tests/unit/test_chunker.py::test_segment_splitting -v
```

## Architecture

```
konte/
├── __init__.py      # Public API exports
├── models.py        # Pydantic models: Chunk, ContextualizedChunk, RetrievalResult, RetrievalResponse, ProjectConfig
├── loader.py        # Document loading (PDF, TXT, MD) - sync and async
├── chunker.py       # Segment (~8000 tokens) + chunk (800 tokens) with overlap
├── context.py       # Async LLM context generation (LangChain abatch, cached LLM instance)
├── project.py       # Main interface: Project class
├── manager.py       # Project CRUD: create, list, get, delete
├── config/
│   └── settings.py  # pydantic-settings, env vars, SSoT for config
├── stores/
│   ├── base.py      # Store protocol (common interface)
│   ├── faiss_store.py   # FAISS index: build, save, load, query
│   ├── bm25_store.py    # BM25 index: build, save, load, query
│   └── retriever.py     # Hybrid retrieval + reciprocal rank fusion
└── cli/
    ├── app.py       # Typer CLI commands
    └── __main__.py  # Module entry point

prompts/
└── context_prompt.txt   # Tariff-domain context generation prompt

docs/
└── AGENT_INTEGRATION_GUIDE.md  # LangChain/Agno integration guide

examples/
└── parallel_multi_project_retrieval.py  # Multi-project querying

tests/
├── unit/           # Mocks allowed here only
├── integration/    # Real API calls
├── e2e/            # End-to-end tests
└── fixtures/       # Test documents (sample.txt, sample.md)
```

## Public API

```python
from konte import (
    # Main class
    Project,

    # Manager functions
    create_project,
    list_projects,
    get_project,
    delete_project,
    project_exists,

    # Models
    Chunk,
    ContextualizedChunk,
    RetrievalResult,
    RetrievalResponse,
    ProjectConfig,

    # Settings
    settings,
)
```

## Usage Examples

### Create and Build Project
```python
project = Project.create("my_project")
project.add_documents([Path("doc.pdf"), Path("doc.txt")])
await project.build()  # async - generates context, builds indexes
project.save()
```

### Query Project
```python
response = project.query("search query", mode="hybrid", top_k=20)
# mode: "hybrid" (default), "semantic", "lexical"

for result in response.results:
    print(f"[{result.score:.2f}] {result.content}")

# Agent hints
print(response.suggested_action)  # "deliver", "query_more", "refine_query"
```

### Load Existing Project
```python
project = Project.open("my_project")
# or
project = get_project("my_project")
```

## System Flowcharts

- `high_level_system_flowchart.md` - Overview of ingestion → context → indexing → retrieval pipeline
- `low_level_system_flowchart.md` - Detailed implementation flow with token counting, error handling, score calculation

## Key Concepts

- **Segment**: ~8000 token portion of document, 10% overlap (800 tokens)
- **Chunk**: 800 token portion of segment, 10% overlap (80 tokens)
- **Contextual Chunk**: Original chunk + LLM-generated context (100-200 tokens) prepended
- **Hybrid Retrieval**: FAISS (semantic) + BM25 (lexical) combined via reciprocal rank fusion
- **Suggested Action**: Agent decision hints based on top_score (deliver ≥0.7, query_more ≥0.4, refine_query <0.4)
- **Batch Processing**: Uses LangChain `abatch()` (immediate results), not OpenAI Batch API (24hr latency)

## Storage Structure

```
{STORAGE_PATH}/
└── {project_name}/
    ├── config.json       # ProjectConfig
    ├── raw_chunks.json   # Raw chunks (before build)
    ├── segments.json     # Segment data
    ├── chunks.json       # ContextualizedChunk list (after build)
    ├── faiss.faiss       # FAISS index (LangChain format)
    ├── faiss.pkl         # FAISS docstore (LangChain format)
    ├── bm25.pkl          # BM25 index (if enabled)
    └── bm25_chunks.json  # Chunk data for BM25
```

## Configuration

All config via pydantic-settings in `settings.py`. Key settings:
- `OPENAI_API_KEY`: Required for embeddings and context generation
- `STORAGE_PATH`: Project storage (default: `~/.konte`)
- `EMBEDDING_MODEL`: text-embedding-3-small
- `CONTEXT_MODEL`: gpt-4.1
- `DEFAULT_TOP_K`: 20
- `SEGMENT_SIZE`: 8000 tokens
- `CHUNK_SIZE`: 800 tokens
- `MAX_CONCURRENT_CALLS`: 10

## Dependencies

- LangChain v0.3 for LLM/embeddings
- faiss-cpu for vector search
- rank-bm25 for lexical search
- pydantic + pydantic-settings for models/config
- structlog for logging
- tiktoken for token counting
- aiofiles for async file I/O
- pypdf for PDF loading
- typer + rich for CLI

## Development Guidelines

### Principles (in priority order)
1. KISS, YAGNI, DRY - priority over SOLID
2. Single Source of Truth (SSoT)
3. Separation of Concerns (SoC)
4. Single Responsibility Principle (SRP)

### Code Style
- **Pydantic V2** for all models - no dataclass
- **Async-first** - no threads, no mixing sync/async
- **Imports at top of file only** - never in middle of functions
- **pathlib for paths** - no hardcoded paths, use settings.py or .env
- **structlog for logging**
- **LangChain v0.3** with `.with_structured_output(PydanticModel)` for LLM outputs - no regex parsing

### Testing (TDD)
- Write tests first, run often, refactor to pass
- **Unit tests** (`tests/unit/`): Mocks and fixtures allowed here only
- **Integration tests** (`tests/integration/`): Real API calls, real dependencies - no mocks
- Test fixtures in `tests/fixtures/`
- No mock functions/classes in production code

### File Organization
- Debug/temp code: `debug/`
- Example code: `examples/` or `scripts/`
- Data check code: `data_check/`
- Prefer updating existing files over creating new ones
