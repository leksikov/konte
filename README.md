# Konte

[![CI](https://github.com/leksikov/konte/actions/workflows/ci.yml/badge.svg)](https://github.com/leksikov/konte/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/leksikov/konte/blob/main/LICENSE)

**A contextual RAG library for Python — hybrid retrieval that finds the chunk your users are actually asking about.**

Your best chunk is often invisible to search, because chunking threw away the words that made it findable. Konte fixes that: before any chunk is embedded or indexed, an LLM that can see the entire surrounding section writes a short context and prepends it. The chunk keeps its meaning, and retrieval starts working on real questions.

## See the difference

Here is a chunk from a quarterly report, split the ordinary way:

```text
Revenue grew 3% over the previous quarter, driven primarily by
stronger demand in the memory segment.
```

Now search for **"Samsung Q3 2024 memory revenue"**. This chunk *is* the answer — and it contains none of those words. No company. No quarter. No year. Embeddings miss it; BM25 has nothing to match. This is the single biggest source of silent RAG failure.

Konte indexes the same chunk like this:

```text
This passage is from Samsung Electronics' Q3 2024 quarterly report,
in the Semiconductor Division section covering DRAM and NAND
performance. It follows the divisional revenue breakdown, where the
memory business is reported separately from foundry.

Revenue grew 3% over the previous quarter, driven primarily by
stronger demand in the memory segment.
```

Same chunk, same answer — now findable both semantically *and* lexically. You write no rules and no metadata schema; the context is generated at build time from the document itself.

## Does it actually work

Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) research reports up to a **49% reduction** in failed retrievals, and **67% with reranking**.

We measured the end-to-end effect on answer quality, not just retrieval, using DeepEval LLM-as-judge on a ~3,000-chunk technical corpus:

| Configuration | Exact lookup (100 q) | Diverse RAG (70 q) |
|---|---|---|
| **Konte** — contextual chunks + reranking | **97%** | **98.6%** |
| Baseline — no context, no reranking | 85% | 74% |

The gains concentrate exactly where naive RAG hurts most: **+25%** on complex, multi-context questions. Methodology and how to reproduce on your own corpus: [docs/EVALUATION.md](https://github.com/leksikov/konte/blob/main/docs/EVALUATION.md).

## What you get

| | |
|---|---|
| **Contextual indexing** | LLM-written context prepended to every chunk, for both the vector *and* keyword index |
| **Hybrid retrieval** | FAISS semantic + BM25 lexical, fused with reciprocal rank fusion — or either alone |
| **Full RAG, one call** | `query_with_answer()` retrieves and generates a grounded answer |
| **Agent-ready** | Every response carries `suggested_action` — `deliver`, `query_more`, or `refine_query` |
| **Reranking** | Optional, via any vLLM `/score` endpoint |
| **Filtering** | By source filename or arbitrary metadata, with AND/OR semantics |
| **Bring your own model** | Any OpenAI-compatible endpoint for context and answer generation |
| **Built to resume** | Per-segment checkpointing; an interrupted build picks up where it stopped |
| **Prefix-cache aware** | Prompts are ordered so vLLM/OpenAI KV caching actually kicks in — see [Performance](#performance-optimizations) |

Runs locally. Your documents stay on your machine, in your own index.

## Installation

Konte is pre-release and installs from source. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/leksikov/konte.git
cd konte
uv sync                   # library + CLI + dev tools, into .venv
source .venv/bin/activate # or prefix every command below with `uv run`
```

Or with pip:

```bash
pip install -e .          # library + CLI
```

Examples below assume an activated environment; with uv, run them as `uv run konte ...` instead.

**Optional extras** — the REST API server and web UI:

```bash
uv sync --extra api --extra ui      # or: pip install -e ".[api,ui]"
```

`uv sync` is exact: name every extra and group you want in one command, or pass `--inexact` to add without removing.

Then set your key:

```bash
export OPENAI_API_KEY=sk-...
```

OpenAI is the default, but context and answer generation work with **any** OpenAI-compatible endpoint — a local vLLM server, for example (see [Configuration](#configuration)). Embeddings use OpenAI whenever FAISS is enabled, so the key is optional only for BM25-only projects.

## Quick start

Index a set of reports and search them — create, add, build, query.

```python
import asyncio
from pathlib import Path
from konte import Project

async def main():
    project = Project.create("annual_reports")

    project.add_documents([
        Path("SAMSUNG_2024_Q3.pdf"),
        Path("TSMC_2024_ANNUAL.md"),
    ])

    # segment (~8k tokens) → chunk (800) → LLM context → FAISS + BM25
    await project.build()
    project.save()

    response = project.query("Samsung Q3 2024 memory revenue")

    for r in response.results[:3]:
        print(f"[{r.score:.2f}] {r.source}")
        print(f"        {r.content[:80]}...")

    print(f"\n{response.suggested_action} (top score {response.top_score:.2f})")

asyncio.run(main())
```

```text
[0.91] SAMSUNG_2024_Q3.pdf
        Revenue grew 3% over the previous quarter, driven primarily by...
[0.77] SAMSUNG_2024_Q3.pdf
        The memory segment recorded its fourth consecutive quarter of...
[0.52] TSMC_2024_ANNUAL.md
        Foundry demand from memory-adjacent customers remained stable...

deliver (top score 0.91)
```

Note the top hit — the chunk from the opening example, retrieved by a query that shares no words with it.

Build once; reopen instantly from then on:

```python
project = Project.open("annual_reports")
```

### Want an answer, not chunks

```python
response, answer = await project.query_with_answer(
    query="How did Samsung's memory business perform in Q3 2024?",
    max_chunks=5,
)

print(answer.answer)        # answered only from the retrieved chunks
print(answer.sources_used)  # 5
print(response.top_score)   # 0.91
```

### The same thing from the terminal

```bash
konte create annual_reports
konte add annual_reports SAMSUNG_2024_Q3.pdf TSMC_2024_ANNUAL.md
konte build annual_reports

konte query annual_reports "Samsung Q3 2024 memory revenue"   # chunks
konte ask   annual_reports "How did memory perform?" --show-sources  # answer

konte list                 # all projects
konte info annual_reports  # chunk counts, config, index status
konte delete annual_reports
```

Serve it instead of scripting it:

```bash
konte serve --host 0.0.0.0 --port 8000   # REST API   ([api] extra)
konte ui --port 7860                     # web UI     ([ui] extra)
```

Both open each project once and keep it in memory, so a request costs a search
rather than a reload. Set `PRELOAD_PROJECTS` to pay for that opening at startup
instead of on the first request, and `PROJECT_CACHE_SIZE` to bound how many
projects a server holds at once:

```bash
PRELOAD_PROJECTS=annual_reports,tariffs   # or "*" for every project
PROJECT_CACHE_SIZE=4
```

Embedding Konte in your own server? `get_shared_project()` is the same
process-wide cache the API uses. It hands out one instance per project, safe to
query from several threads and reopened by itself whenever the project is
rebuilt on disk. `get_project()` still returns a private instance, which is what
anything that adds documents, builds, or saves needs:

```python
from konte import get_shared_project

project = get_shared_project("annual_reports")   # shared, query-only
response = project.query("Samsung Q3 2024 memory revenue")
```

## Project Management

```python
from konte import (
    create_project,
    list_projects,
    get_project,
    delete_project,
    project_exists,
)

# Create new project
project = create_project("my_project")

# List all projects
projects = list_projects()  # Returns: ["my_project", ...]

# Check if project exists
if project_exists("my_project"):
    project = get_project("my_project")

# Delete project
delete_project("my_project")
```

## Retrieval Modes

```python
# Hybrid (default) - FAISS + BM25 with rank fusion
response = project.query("query", mode="hybrid")

# Semantic only - FAISS embeddings
response = project.query("query", mode="semantic")

# Lexical only - BM25 keyword matching
response = project.query("query", mode="lexical")
```

### BM25 Keyword Extraction

BM25 has no notion of stopwords or Korean particles, so a natural-language
question dilutes the terms that matter. Before a lexical or hybrid search,
Konte reduces the query to content words with one LLM call — a large recall
gain for Korean, and a round trip your caller waits for.

It is on by default, capped at `KEYWORD_EXTRACTION_TIMEOUT` with no retries,
and cached per query, so a repeated search costs nothing. Turn it off per call:

```python
# Pure index lookup: no LLM, no network
response = project.query("HS 8471 classification", use_keyword_extraction=False)
```

or for a whole deployment with `BM25_KEYWORD_EXTRACTION=false`. The same
control exists on `query_async()`, `query_with_answer()`, the
`use_keyword_extraction` field of the `/query` and `/ask` endpoints, the
`--no-keyword-extraction` CLI flag, and the UI checkbox. When extraction fails
or times out, retrieval falls back to whitespace tokenization with stopword
filtering rather than failing the query.

## Skip Context Generation

For standard RAG without LLM-generated context (faster, cheaper):

```python
await project.build(skip_context=True)
```

## Custom Context Prompts

The context-generation prompt is a plain text template with two placeholders: `{segment}` (the ~8000-token surrounding segment) and `{chunk}` (the chunk to contextualize). The default prompt (`konte/prompts/context_prompt.txt`) is domain-neutral and responds in the document's language.

Domain-specific prompts substantially improve retrieval for specialized corpora. Override per project:

```python
project = Project.create("my_project", context_prompt_path=Path("my_prompt.txt"))
```

or via the CLI:

```bash
konte create my_project --prompt my_prompt.txt
konte build my_project --prompt my_prompt.txt
```

or globally with the `PROMPT_PATH` environment variable. See [examples/prompts/](https://github.com/leksikov/konte/tree/main/examples/prompts/) for domain-specific prompt examples (Korean customs-tariff classification).

## RAG Answer Generation

`query_with_answer()` is the full RAG pipeline: retrieve chunks, then generate an LLM-grounded answer.

```python
import asyncio
from konte import Project

async def main():
    project = Project.open("my_project")

    # Full RAG pipeline: retrieval + LLM answer
    response, answer = await project.query_with_answer(
        query="What was the revenue growth in Q3?",
        mode="hybrid",
        max_chunks=5,
    )

    print(answer.answer)       # LLM-generated answer
    print(answer.model)        # Model used (e.g., "gpt-4.1-mini")
    print(answer.sources_used) # Number of chunks used

    # Retrieval metadata is also available
    print(response.top_score)        # How well the best result matched (0-1)
    print(response.suggested_action) # "deliver", "query_more", or "refine_query"

asyncio.run(main())
```

Answers are generated with OpenAI by default, or with your own OpenAI-compatible endpoint when `LLM_BASE_URL` and `LLM_MODEL` are configured.

### Custom Prompt Templates

Override the default answer prompt with `{context}` and `{question}` placeholders:

```python
custom_prompt = """Based on the following documents, answer the question.

Documents:
{context}

Question: {question}

Provide the answer with references to source documents.
Answer:"""

response, answer = await project.query_with_answer(
    query="What are the main risk factors?",
    prompt_template=custom_prompt,
    max_chunks=10,
)
```

### With Reranking

Combine answer generation with reranking for better retrieval quality. Reranking requires a vLLM server exposing a `/score` endpoint (e.g. serving Qwen3-Reranker-8B), configured via `RERANKER_BASE_URL`:

```python
response, answer = await project.query_with_answer(
    query="How do the two products compare?",
    rerank=True,           # Rerank via RERANKER_BASE_URL before answer generation
    rerank_initial_k=50,   # Retrieve 50 candidates, rerank to top_k
    max_chunks=5,
)
```

## Metadata Filtering

Filter retrieval results by source filename or custom metadata fields.

### source_filter — Substring Match

`source_filter` matches against the chunk's source filename (case-sensitive substring):

```python
# Only return chunks from Samsung documents
response = project.query("revenue growth", source_filter="SAMSUNG")

# Only return chunks from 2024 reports
response = project.query("revenue growth", source_filter="2024")

# Combine with retrieval mode
response = project.query("DRAM market", mode="semantic", source_filter="SAMSUNG_2024")
```

### metadata_filter — Equality Match

`metadata_filter` filters by exact values in chunk metadata (AND logic for multiple keys):

```python
# Filter by exact source filename
response = project.query(
    "semiconductor revenue",
    metadata_filter={"source": "TSMC_2024_ANNUAL.md"},
)

# Filter by multiple fields (AND logic)
response = project.query(
    "quarterly results",
    metadata_filter={"company": "SAMSUNG", "year": 2024},
)

# List values match any (OR within a key)
response = project.query(
    "chip production",
    metadata_filter={"company": ["SAMSUNG", "TSMC"]},
)
```

### With Answer Generation

Filters work with `query_with_answer()` too:

```python
response, answer = await project.query_with_answer(
    query="What was the revenue?",
    source_filter="SAMSUNG_2024",
    max_chunks=5,
)
```

See [examples/metadata_filtering.py](https://github.com/leksikov/konte/blob/main/examples/metadata_filtering.py) for a complete example.

## Agent Integration

Konte returns retrieval responses with decision hints for agent workflows:

```python
response = project.query("query")

# Suggested action based on confidence
print(response.suggested_action)
# "deliver" (score >= 0.7), "query_more" (0.4-0.7), or "refine_query" (< 0.4)

print(response.has_high_confidence)  # True if top_score >= 0.7
print(response.top_score)            # How well the best result matched (0-1)
print(response.score_spread)         # Same measure, best minus worst
```

`top_score` measures the query against the chunk — vector similarity, the
share of the query's terms the lexical index matched, or the reranker's own
score. It is comparable across queries, which is what makes a threshold on it
mean anything.

`RetrievalResult.score` is a different number: it is what ordered the results
inside one response. Rank fusion and lexical normalization both scale the best
result to 1.0, so `results[0].score` is 1.0 in hybrid and lexical mode whether
the corpus answered the question or merely returned its least bad chunk. Rank
with it; never threshold on it.

```python

# Use as a plain callable tool
retriever = project.as_retriever()
response = retriever("my query")
```

Konte integrates with LangChain and Agno agent frameworks. See the [Agent Integration Guide](https://github.com/leksikov/konte/blob/main/docs/AGENT_INTEGRATION_GUIDE.md) for:

- LangChain RAG chains and custom retrievers
- Agno tools and multi-project agents
- Confidence-based agent decisions
- Streaming responses

## RetrievalResponse Schema

All models are Pydantic V2:

```python
class RetrievalResponse:
    results: list[RetrievalResult]  # Ranked results
    query: str                      # Original query
    total_found: int                # Number of results
    top_score: float                # Best result's match against the query (0-1)
    score_spread: float             # Same measure, best minus worst
    has_high_confidence: bool       # top_score >= 0.7
    suggested_action: str           # "deliver", "query_more", "refine_query"

class RetrievalResult:
    content: str      # Original chunk text
    context: str      # LLM-generated context
    score: float      # Ranking score within this response (0-1), not a threshold
    source: str       # Source filename
    chunk_id: str     # Unique chunk identifier
    metadata: dict    # Additional metadata
```

## Multi-Project Retrieval

Query multiple knowledge bases in parallel:

```python
from konte import Project

projects = [
    Project.open("annual_reports"),
    Project.open("product_docs"),
    Project.open("support_tickets"),
]

# Query all projects
results = {}
for project in projects:
    results[project.config.name] = project.query("battery performance issues")

# Merge and rank
all_results = []
for name, response in results.items():
    for r in response.results:
        all_results.append((name, r))

all_results.sort(key=lambda x: x[1].score, reverse=True)
```

See [examples/parallel_multi_project_retrieval.py](https://github.com/leksikov/konte/blob/main/examples/parallel_multi_project_retrieval.py) for a complete example.

## Performance Optimizations

### vLLM/OpenAI Prefix Caching

Context generation is optimized for KV cache prefix caching:

```
Prompt structure: [SEGMENT ~8000 tokens] + [CHUNK ~800 tokens]
```

**How it works:**
1. All chunks within a segment share the same prefix (segment text)
2. Chunks are sent in parallel via `abatch(max_concurrency=len(chunks))`
3. First request computes and caches the segment prefix KV states
4. Subsequent chunk requests hit the cache - only compute the unique chunk suffix
5. Segments are processed sequentially to maximize cache efficiency

```
Segment A (10 chunks):
  Request 1: segment_A + chunk_1  → compute prefix, cache it
  Request 2: segment_A + chunk_2  → cache hit, compute only chunk_2
  Request 3: segment_A + chunk_3  → cache hit, compute only chunk_3
  ...
Then Segment B, etc.
```

Request order within a segment doesn't matter - whichever arrives first triggers caching.

### Other Optimizations

- **LLM Instance Caching**: Reuses ChatOpenAI instance across calls
- **Batch Processing**: Uses LangChain's `abatch()` for parallel LLM calls within segment
- **Build Checkpointing**: Each finished segment appends one line to a checkpoint
  log, so an interrupted build resumes at a segment boundary and the log costs
  one write per segment rather than one rewrite of everything so far
- **Shared Project Cache**: The API and UI open a project once per process
  instead of per request, and reopen it only after it is rebuilt
- **Deferred Corpus Parsing**: Opening a project reads its indexes; the stored
  chunks and segments are parsed only if something asks for them, which a
  query-serving process never does

## Logging

Structured logging via structlog provides visibility into the ingestion pipeline:

```
2024-01-15 10:30:01 [info] loading_document path=/data/doc.pdf
2024-01-15 10:30:02 [info] document_chunked path=/data/doc.pdf num_chunks=55
2024-01-15 10:30:02 [info] context_generation_started total_segments=5 skip_context=False
2024-01-15 10:30:03 [info] generating_context_for_segment segment_key=('doc.pdf', 0) total_segments=5 num_chunks=11
...
2024-01-15 10:30:15 [info] context_generation_complete num_chunks=55 skipped=False
2024-01-15 10:30:16 [info] faiss_index_built
2024-01-15 10:30:16 [info] project_build_complete
```

Pipeline: 1 document → 5 segments (~8000 tokens each) → ~11 chunks per segment (~800 tokens) = 55 chunks total. Use debug level for granular token counts.

## Index Options

```python
# FAISS only (no BM25)
project = Project.create("semantic_only", enable_bm25=False)

# BM25 only (no FAISS) - no embeddings needed
project = Project.create("lexical_only", enable_faiss=False)
```

## Configuration

Set via environment variables or `.env` file (see [.env.example](https://github.com/leksikov/konte/blob/main/.env.example)):

```bash
OPENAI_API_KEY=sk-...          # Required for embeddings (FAISS, the default)
STORAGE_PATH=~/.konte          # Project storage location
EMBEDDING_MODEL=text-embedding-3-small
CONTEXT_MODEL=gpt-4.1-mini     # Model for context/answer generation
DEFAULT_TOP_K=20
PROMPT_PATH=                   # Optional global context-prompt override

BM25_KEYWORD_EXTRACTION=true   # LLM keyword extraction before lexical search
KEYWORD_EXTRACTION_TIMEOUT=5.0 # Seconds before falling back to tokenization

# Optional: any OpenAI-compatible server (vLLM, Ollama, LM Studio, ...)
# (replaces OpenAI for context/answer generation only - not embeddings)
LLM_BASE_URL=https://your-endpoint/v1
LLM_MODEL=your-model-name

# Optional: reranker (vLLM-style /score endpoint), required only for rerank=True
RERANKER_BASE_URL=https://your-endpoint/v1
RERANKER_MODEL=Qwen3-Reranker-8B
```

## Examples

| Example | Demonstrates |
|---------|-------------|
| [basic_usage.py](https://github.com/leksikov/konte/blob/main/examples/basic_usage.py) | Project CRUD, document loading, building, querying, retrieval modes |
| [query_with_answer.py](https://github.com/leksikov/konte/blob/main/examples/query_with_answer.py) | Full RAG pipeline, custom prompt templates, GeneratedAnswer model |
| [metadata_filtering.py](https://github.com/leksikov/konte/blob/main/examples/metadata_filtering.py) | source_filter, metadata_filter, combining filters with modes |
| [async_reranking.py](https://github.com/leksikov/konte/blob/main/examples/async_reranking.py) | Async querying, reranking, comparing with/without reranking |
| [parallel_multi_project_retrieval.py](https://github.com/leksikov/konte/blob/main/examples/parallel_multi_project_retrieval.py) | Multi-project querying, result merging |
| [prompts/](https://github.com/leksikov/konte/tree/main/examples/prompts/) | Domain-specific context prompt templates |

## Evaluation

The numbers in [Does it actually work](#does-it-actually-work) come from an end-to-end DeepEval LLM-as-judge run on a customs-tariff classification case study — a private Korean corpus of ~3,000 chunks, scored on answer correctness rather than retrieval hit-rate, so the figures reflect what an end user would actually experience.

Two question sets were used: 100 exact-lookup questions (find one specific code) and 70 diverse RAG questions (reasoning, comparison, multi-context). Contextual retrieval helps most on the latter, where a single chunk is rarely sufficient.

Full results, methodology, ablations, and instructions for running the same evaluation against your own corpus: [docs/EVALUATION.md](https://github.com/leksikov/konte/blob/main/docs/EVALUATION.md).

## Architecture

```text
Documents (PDF / TXT / MD)
            |
            v
Segmenter (~8000-token segments, 10% overlap)
            |
            v
Chunker (800-token chunks, 10% overlap)
            |
            v
LLM context generation (100-200 tokens per chunk,
sees the full segment; optional via skip_context)
            |
            v
Contextualized chunks (context + content)
            |
     +------+------+
     |             |
     v             v
FAISS index    BM25 index
(semantic)     (lexical)
     |             |
     +------+------+
            |
            v
Hybrid retrieval (reciprocal rank fusion)
            |
            v
RetrievalResponse: scored chunks + suggested_action
(deliver / query_more / refine_query)
```

Detailed diagrams: [architecture overview](https://github.com/leksikov/konte/blob/main/docs/architecture_overview.md) · [detailed pipeline](https://github.com/leksikov/konte/blob/main/docs/architecture_detailed.md)

## Troubleshooting

**macOS: `OMP: Error #15` / libomp crash when importing FAISS**

FAISS and other libraries may both load OpenMP. Set:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Contributing

Contributions welcome — see [CONTRIBUTING.md](https://github.com/leksikov/konte/blob/main/CONTRIBUTING.md). Run `uv run pytest tests/unit` (no API keys needed) and `uv run ruff check .` before submitting a PR.

## License

[MIT](https://github.com/leksikov/konte/blob/main/LICENSE)
