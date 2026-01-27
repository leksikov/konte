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

# Run unit tests only (111 tests, no API calls)
pytest tests/unit/

# Run integration tests (53 tests, requires OPENAI_API_KEY)
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
├── loader.py        # Document loading (PDF, TXT, MD) - sync and async, with logging
├── chunker.py       # Segment (~8000 tokens) + chunk (800 tokens) with overlap, with logging
├── context.py       # Async LLM context generation (LangChain abatch, cached LLM instance)
├── generator.py     # RAG answer generation with LLM (BackendAI/OpenAI)
├── project.py       # Main interface: Project class (query, query_with_answer)
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

scripts/
├── build_knowledge_bases.py      # Build all projects from example_knowledge_base/
├── build_combined_project.py     # Merge all projects into unified index
└── build_baseline_project.py     # Build project without context (for baseline eval)

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
    GeneratedAnswer,

    # Generator
    generate_answer,

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

### Full RAG with Answer Generation
```python
# Query with LLM-generated answer
response, answer = await project.query_with_answer(
    query="What is the HS code for DDR5 RAM?",
    mode="hybrid",
    max_chunks=5,
)
print(answer.answer)       # LLM-generated answer
print(answer.model)        # Model used
print(answer.sources_used) # Number of chunks used
```

## System Flowcharts

- `high_level_system_flowchart.md` - Overview of ingestion → context → indexing → retrieval pipeline
- `low_level_system_flowchart.md` - Detailed implementation flow with token counting, error handling, score calculation

## Key Concepts

- **Segment**: ~8000 token portion of document, 10% overlap (800 tokens)
- **Chunk**: 800 token portion of segment, 10% overlap (80 tokens)
- **Contextual Chunk**: Original chunk + LLM-generated context (up to 300 tokens) prepended
- **Hybrid Retrieval**: FAISS (semantic) + BM25 (lexical) combined via reciprocal rank fusion
- **Suggested Action**: Agent decision hints based on top_score (deliver ≥0.7, query_more ≥0.4, refine_query <0.4)
- **Batch Processing**: Uses LangChain `abatch()` (immediate results), not OpenAI Batch API (24hr latency)
- **RAG Answer Generation**: Optional LLM answer generation from retrieved chunks via `query_with_answer()`
- **Custom Metadata**: Chunks support custom metadata dict for filtering (e.g., document_name, page_no)
- **Combined Projects**: Multiple projects can be merged into unified index via `scripts/build_combined_project.py`
- **Tokenizer**: Uses gpt-4.1 (o200k_base) encoding - ~30% more efficient for Korean text

## Logging & Observability

Structured logging via structlog provides visibility into the ingestion pipeline.

### Pipeline Log Events

| Stage | Log Event | Level | Key Fields |
|-------|-----------|-------|------------|
| **Loading** | `document_loading` | debug | path, file_type |
| | `document_loaded` | debug | path, content_length |
| | `pdf_pages_loaded` | debug | path, page_count |
| | `loading_document` | info | path |
| | `document_chunked` | info | path, num_chunks |
| | `documents_added` | info | total_chunks |
| **Chunking** | `segmentation_started` | debug | source, total_tokens |
| | `segment_created` | debug | source, segment_index, token_count |
| | `chunking_segment` | debug | source, segment_index, num_chunks |
| | `chunks_created` | debug | source, total_segments, total_chunks |
| **Context** | `context_generation_started` | info | total_segments, skip_context |
| | `generating_context_for_segment` | info | segment_index, total_segments, num_chunks |
| | `context_generation_complete` | info | num_chunks, skipped |
| **Indexing** | `faiss_index_built` | info | - |
| | `bm25_index_built` | info | - |
| | `project_build_complete` | info | - |

### Log Levels
- **info**: Major pipeline stages (document load, context generation, index build)
- **debug**: Granular events (individual segments, token counts, file reads)

### Example Output
```
2024-01-15 10:30:01 [info] loading_document path=/data/doc.pdf
2024-01-15 10:30:02 [info] document_chunked path=/data/doc.pdf num_chunks=55
2024-01-15 10:30:02 [info] context_generation_started total_segments=5 skip_context=False
2024-01-15 10:30:03 [info] generating_context_for_segment segment_index=0 total_segments=5 num_chunks=11
2024-01-15 10:30:05 [info] generating_context_for_segment segment_index=1 total_segments=5 num_chunks=11
...
2024-01-15 10:30:15 [info] context_generation_complete num_chunks=55 skipped=False
2024-01-15 10:30:16 [info] faiss_index_built
2024-01-15 10:30:16 [info] bm25_index_built
2024-01-15 10:30:16 [info] project_build_complete
```

Note: 1 document → 5 segments (~8000 tokens each) → ~11 chunks per segment (~800 tokens each) = 55 total chunks.

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
- `OPENAI_API_KEY`: Optional (required for OpenAI embeddings, Backend.AI is default)
- `STORAGE_PATH`: Project storage (default: `~/.konte`)
- `EMBEDDING_MODEL`: text-embedding-3-small
- `CONTEXT_MODEL`: gpt-4.1
- `DEFAULT_TOP_K`: 20
- `SEGMENT_SIZE`: 8000 tokens
- `CHUNK_SIZE`: 800 tokens
- `MAX_CONCURRENT_CALLS`: 1
- `BACKENDAI_ENDPOINT`: BackendAI vLLM endpoint (default: `https://qwen3vl.asia03.app.backend.ai/v1`)
- `BACKENDAI_MODEL_NAME`: Model for context/answer generation (default: `Qwen3-VL-8B-Instruct`)

## Available Knowledge Base Projects

| Project | Chunks | Description |
|---------|--------|-------------|
| `hs_machinery_electronics_guide` | 182 | HS Ch 84-85 machinery/electronics classification |
| `korea_tariff_schedule` | 687 | Official Korean tariff rates (legal document) |
| `tariff_terminology_bilingual` | 485 | Korean-English tariff terminology |
| `tariff_terminology_dictionary` | 6,270 | Comprehensive tariff terminology dictionary |
| `wco_hs_explanatory_notes` | 3,036 | WCO HS explanatory notes and GRI rules |
| `all_tariff_documents` | 10,660 | Combined index of all above projects |

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

## Evaluation

RAG evaluation using DeepEval with LLM-as-judge metrics. Evaluates answer correctness for diverse question types.

**Important**: This is a RAG (Retrieval-Augmented Generation) system for answering questions based on retrieved context. It is NOT a classification system.

### Structure
```
evaluation/
├── custom_llm.py                # BackendAI LLM wrapper for DeepEval
├── custom_metrics.py            # Custom AnswerCorrectness GEval metric
├── deepeval_synthesizer.py      # Generate diverse test cases with DeepEval Synthesizer
├── EVALUATION_REPORT.md         # Detailed experiment results
├── EVALUATION_GUIDE.md          # How to run evaluation pipeline
├── prompts/
│   ├── __init__.py              # Exports prompt configurations
│   └── eval_prompts.py          # GEval criteria for HS Code and Answer Correctness
├── data/
│   └── synthetic/
│       ├── deepeval_goldens_korean_100.json  # 100 diverse questions (DeepEval Synthesizer)
│       ├── synthetic_goldens_100.json        # 100 HS code questions (legacy)
│       └── archive/                          # Old datasets
├── experiments/
│   ├── llm_reranking.py         # LLM reranking experiments (binary filter)
│   ├── run_deepeval_full.py     # DeepEval evaluation (supports hs_code and answer types)
│   ├── run_baseline_eval.py     # Baseline evaluation (no context, no reranking)
│   └── results/                 # Experiment results JSON files
└── results/
    └── *.log                    # Evaluation logs
```

### Test Case Generation

Use DeepEval Synthesizer to generate diverse question types:
```bash
python -m evaluation.deepeval_synthesizer \
  --project wco_hs_explanatory_notes_korean \
  --output evaluation/data/synthetic/deepeval_goldens_korean_100.json \
  --num 100 --model gpt-4.1-mini
```

Evolution types for question diversity:
- **Reasoning**: Logical complexity questions
- **Multi-context**: Questions requiring multiple sources
- **Concretizing**: Specific detail questions
- **Constrained**: Questions with constraints
- **Comparative**: Comparison questions
- **Hypothetical**: "What if" scenarios
- **In-Breadth**: Scope expansion questions

### Contextual RAG vs Baseline

| Configuration | HS Code (100q) | Diverse RAG (100q) |
|---------------|----------------|---------------------|
| **Contextual + Reranking** | **94% (0.920)** | **94% (0.828)** |
| Baseline (no context, no rerank) | 85% (0.822) | 74% (0.613) |
| **Improvement** | **+9% (+0.098)** | **+20% (+0.215)** |

Key findings:
- Context generation provides +20% improvement for diverse questions
- HS code lookups benefit +9% from context
- Most impact on complex, multi-context questions

### Evaluation by Question Type

| Evolution Type | Pass Rate | Avg Score |
|----------------|-----------|-----------|
| Reasoning | 100.0% | 0.861 |
| Comparative | 96.9% | 0.841 |
| Constrained | 96.7% | 0.857 |
| In-Breadth | 95.7% | 0.852 |
| Multi-context | 92.9% | 0.814 |
| Concretizing | 91.2% | 0.809 |
| Hypothetical | 86.7% | 0.777 |
| **TOTAL** | **94.0%** | **0.828** |

### Two Evaluation Types

The system supports two evaluation types with switchable LLM judge prompts:

| Type | Dataset | Metric | Use Case |
|------|---------|--------|----------|
| `answer` | `deepeval_goldens_korean_100.json` | AnswerCorrectness | Diverse RAG questions (default) |
| `hs_code` | `synthetic_goldens_100.json` | HSCodeCorrectness | HS code lookup questions |

### Quick Start

```bash
# Run LLM reranking (generates answers)
python -m evaluation.experiments.llm_reranking \
  --project wco_hs_explanatory_notes_korean \
  --test-cases evaluation/data/synthetic/deepeval_goldens_korean_100.json \
  --method binary --initial-k 100 --final-k 15 --max-cases 0 \
  --output evaluation/experiments/results/llm_rerank_binary_deepeval_diverse.json

# Run DeepEval evaluation (answer correctness - default)
python -m evaluation.experiments.run_deepeval_full binary deepeval_diverse answer

# Or for HS code evaluation (legacy)
python -m evaluation.experiments.run_deepeval_full binary 100 hs_code
```

### GEval Metrics (LLM-as-Judge)

Two metrics available in `evaluation/prompts/eval_prompts.py`:

**AnswerCorrectness** (for diverse RAG questions):
- Evaluates if actual output contains key facts from expected output
- Checks semantic equivalence (same meaning, different wording OK)
- Verifies technical accuracy (codes, terms, numbers)
- Ignores format differences, language mixing, verbosity

**HSCodeCorrectness** (for HS code lookup questions):
- Focuses on HS code accuracy and semantic equivalence
- Normalizes format differences (제2523.21호 = 2523.21 = 제2523호의 21)
- Ignores language mixing and extra explanation

See `evaluation/EVALUATION_GUIDE.md` for detailed instructions.
