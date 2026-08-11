"""Did the outputs change, not just the timings?

Speed numbers only mean something if both revisions still do the same thing.
This case records what each revision actually produces, at three levels:

``chunking``
    The segments and chunks a document is cut into - counts, boundaries, and a
    hash of every piece of text. Chunk boundaries decide what can ever be
    retrieved, so a change here would invalidate every stored index.

``retrieval``
    The top-k for a fixed query set, in each mode. Run twice: once with keyword
    extraction pinned to identical local tokenization, which isolates the
    ranking code, and once with each revision left to its own defaults, which
    shows what a caller would actually observe.

``answer``
    The generated answer for the same question, end to end.

Nothing here is a benchmark. It is the evidence that the benchmark is comparing
like with like.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib

from benchmarks.build import open_project
from benchmarks.corpus import available_real_projects, document_excerpt, real_project
from benchmarks.harness import Context, pin_keyword_extraction

DEFAULT_PROJECT = "wco_korean_feb2026"
CHUNKING_CHUNKS = 40
TOP_K = 5
MODES = ("lexical", "semantic", "hybrid")

QUERIES = [
    "기계류의 부분품은 어느 호에 분류되나요?",
    "소매용으로 포장된 세트의 분류 기준은 무엇인가요?",
    "플라스틱제 물품의 분류 원칙을 설명해 주세요.",
]
ANSWER_QUERIES = QUERIES[:2]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _chunking(ctx: Context) -> dict:
    """Fingerprint how a document is segmented and chunked."""
    chunker = importlib.import_module("konte.chunker")
    document = document_excerpt(CHUNKING_CHUNKS)
    text = document.read_text()

    chunks, segments = chunker.create_chunks(text, source=document.name)

    return {
        "document": document.name,
        "source_characters": len(text),
        "segment_count": len(segments),
        "chunk_count": len(chunks),
        "segments": [
            {
                "key": f"{key[0]}|{key[1]}" if isinstance(key, tuple) else str(key),
                "characters": len(value),
                "tokens": chunker.count_tokens(value),
                "sha": _digest(value),
            }
            for key, value in segments.items()
        ],
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "segment_idx": chunk.segment_idx,
                "chunk_idx": chunk.chunk_idx,
                "characters": len(chunk.content),
                "tokens": chunker.count_tokens(chunk.content),
                "sha": _digest(chunk.content),
                "head": chunk.content[:60],
            }
            for chunk in chunks
        ],
    }


def _top_k(project, query: str, mode: str) -> list[dict]:
    response = project.query(query, mode=mode, top_k=TOP_K)
    return [
        {
            "chunk_id": result.chunk_id,
            "score": round(result.score, 6),
            "source": result.source,
            "snippet": result.content[:140].replace("\n", " "),
        }
        for result in response.results
    ]


def _retrieval(project, pinned: bool) -> dict:
    if pinned:
        pin_keyword_extraction()
    return {query: {mode: _top_k(project, query, mode) for mode in MODES} for query in QUERIES}


def _answers(project) -> dict:
    results = {}
    for query in ANSWER_QUERIES:
        try:
            response, answer = asyncio.run(
                project.query_with_answer(query, mode="hybrid", max_chunks=5)
            )
            results[query] = {
                "answer": answer.answer,
                "model": answer.model,
                "sources_used": answer.sources_used,
                "retrieved": [result.chunk_id for result in response.results[:TOP_K]],
            }
        except Exception as error:  # an endpoint failure is a result, not a crash
            results[query] = {"error": f"{type(error).__name__}: {error}"}
    return results


def run(ctx: Context) -> dict:
    name = ctx.options.get("project", DEFAULT_PROJECT)
    result: dict = {"chunking": _chunking(ctx)}

    if name not in available_real_projects():
        result["retrieval"] = {"status": "skipped", "reason": f"no built project {name!r}"}
        return result

    storage, project_name = real_project(name, ctx.revision, reindex_lexical=True)
    result["project"] = name

    # Defaults first: once extraction is pinned it cannot be unpinned within
    # the process, and the default behaviour is what a caller actually sees.
    project = open_project(project_name, storage)
    result["retrieval_default_settings"] = _retrieval(project, pinned=False)

    if ctx.options.get("answers", "1") != "0":
        result["answers"] = _answers(project)

    result["retrieval_pinned_extraction"] = _retrieval(project, pinned=True)
    return result
