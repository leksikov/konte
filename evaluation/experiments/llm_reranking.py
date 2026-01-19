"""LLM-based reranking experiment.

Uses Qwen3 LLM to filter/rerank 100 candidate chunks down to top-15 for answer generation.

Two approaches:
1. Binary filtering: Ask LLM "Is this chunk relevant?" for each chunk
2. Batch scoring: Ask LLM to score relevance of multiple chunks at once

This is experimental - comparing against hybrid retrieval baseline (88% accuracy).
"""

import asyncio
import json
from pathlib import Path

import httpx
import structlog
from tqdm import tqdm

from konte import Project
from konte.config import settings
from konte.models import RetrievalResult

logger = structlog.get_logger()

# LLM endpoint for reranking (use BackendAI if configured, otherwise OpenAI)
if settings.use_backendai:
    LLM_ENDPOINT = settings.BACKENDAI_ENDPOINT
    LLM_MODEL = settings.BACKENDAI_MODEL_NAME
    LLM_API_KEY = settings.BACKENDAI_API_KEY or "placeholder"
else:
    LLM_ENDPOINT = "https://api.openai.com/v1"
    LLM_MODEL = settings.CONTEXT_MODEL
    LLM_API_KEY = settings.OPENAI_API_KEY

# Concurrency for LLM calls
LLM_CONCURRENCY = 10


BINARY_FILTER_PROMPT = """주어진 질문에 대해 이 문서가 관련성이 있는지 판단하세요.

질문: {query}

문서:
{document}

이 문서가 질문에 답하는 데 도움이 되는 정보를 포함하고 있나요?
"예" 또는 "아니오"로만 답하세요."""


BATCH_SCORE_PROMPT = """다음 질문에 대해 각 문서의 관련성을 0-10 점수로 평가하세요.

질문: {query}

문서들:
{documents}

각 문서 번호와 점수를 JSON 형식으로 출력하세요:
{{"scores": [{{"doc": 1, "score": 8}}, {{"doc": 2, "score": 3}}, ...]}}

점수 기준:
- 10: 질문에 직접 답하는 핵심 정보 포함
- 7-9: 관련 정보 포함, 답변에 도움됨
- 4-6: 일부 관련 정보 있음
- 1-3: 거의 관련 없음
- 0: 전혀 관련 없음"""


async def binary_filter_chunk(
    client: httpx.AsyncClient,
    query: str,
    chunk: RetrievalResult,
    idx: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool, float]:
    """Filter single chunk using binary relevance check.

    Returns (idx, is_relevant, original_score).
    """
    async with semaphore:
        try:
            # Use context (short summary) for faster processing
            doc_text = chunk.context[:500] if chunk.context else chunk.content[:500]

            prompt = BINARY_FILTER_PROMPT.format(
                query=query,
                document=doc_text,
            )

            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            }

            response = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
            )
            response.raise_for_status()
            data = response.json()

            answer = data["choices"][0]["message"]["content"].strip().lower()
            is_relevant = "예" in answer or "yes" in answer

            return (idx, is_relevant, chunk.score)

        except Exception as e:
            logger.warning("binary_filter_failed", idx=idx, error=str(e))
            # On error, keep the chunk (conservative)
            return (idx, True, chunk.score)


async def batch_score_chunks(
    client: httpx.AsyncClient,
    query: str,
    chunks: list[RetrievalResult],
    batch_start: int,
) -> list[tuple[int, float]]:
    """Score a batch of chunks using LLM.

    Returns list of (global_idx, score).
    """
    try:
        # Format documents with numbers
        doc_texts = []
        for i, chunk in enumerate(chunks):
            doc_text = chunk.context[:300] if chunk.context else chunk.content[:300]
            doc_texts.append(f"[문서 {i+1}]\n{doc_text}")

        documents = "\n\n".join(doc_texts)

        prompt = BATCH_SCORE_PROMPT.format(
            query=query,
            documents=documents,
        )

        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.0,
        }

        response = await client.post(
            f"{LLM_ENDPOINT}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
        )
        response.raise_for_status()
        data = response.json()

        answer = data["choices"][0]["message"]["content"].strip()

        # Parse JSON response
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', answer, re.DOTALL)
        if json_match:
            scores_data = json.loads(json_match.group())
            scores = scores_data.get("scores", [])

            results = []
            for item in scores:
                doc_num = item.get("doc", 0) - 1  # Convert to 0-indexed
                score = item.get("score", 0) / 10.0  # Normalize to 0-1
                if 0 <= doc_num < len(chunks):
                    global_idx = batch_start + doc_num
                    results.append((global_idx, score))

            return results
        else:
            # Fallback: return original order with default scores
            return [(batch_start + i, 0.5) for i in range(len(chunks))]

    except Exception as e:
        logger.warning("batch_score_failed", batch_start=batch_start, error=str(e))
        # Fallback: return original order
        return [(batch_start + i, 0.5) for i in range(len(chunks))]


async def rerank_with_binary_filter(
    query: str,
    chunks: list[RetrievalResult],
    top_k: int = 15,
    min_filtered: int = 5,
) -> list[RetrievalResult]:
    """Rerank using binary relevance filtering with fallback.

    Filters out irrelevant chunks. If fewer than min_filtered chunks pass,
    fills up to top_k with top hybrid-scored chunks.

    Args:
        query: Search query.
        chunks: Initial candidate chunks from hybrid retrieval.
        top_k: Maximum number of chunks to return.
        min_filtered: Minimum filtered chunks before triggering fallback.

    Returns:
        List of filtered chunks (filtered first, then hybrid fallback).
    """
    if not chunks:
        return []

    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
        tasks = [
            binary_filter_chunk(client, query, chunk, idx, semaphore)
            for idx, chunk in enumerate(chunks)
        ]

        results = await asyncio.gather(*tasks)

        # Separate relevant and non-relevant chunks
        relevant_indices = set()
        relevant = []
        for idx, is_relevant, score in results:
            if is_relevant:
                relevant.append((idx, score))
                relevant_indices.add(idx)

        # Sort relevant by original score
        relevant.sort(key=lambda x: x[1], reverse=True)

        # If fewer than min_filtered pass, fill with top hybrid chunks
        if len(relevant) < min_filtered:
            # Get non-relevant chunks sorted by hybrid score
            non_relevant = [
                (idx, score) for idx, is_relevant, score in results
                if not is_relevant
            ]
            non_relevant.sort(key=lambda x: x[1], reverse=True)

            # Fill up to top_k
            slots_to_fill = top_k - len(relevant)
            fallback = non_relevant[:slots_to_fill]

            logger.info(
                "binary_filter_fallback",
                relevant_count=len(relevant),
                fallback_count=len(fallback),
            )

            # Combine: filtered first, then fallback
            combined = relevant + fallback
        else:
            combined = relevant[:top_k]

        # Build final list
        reranked = [chunks[idx] for idx, _ in combined[:top_k]]

        logger.info(
            "binary_filter_complete",
            input_chunks=len(chunks),
            relevant_chunks=len(relevant),
            output_chunks=len(reranked),
            used_fallback=len(relevant) < min_filtered,
        )

        return reranked


async def rerank_with_batch_scoring(
    query: str,
    chunks: list[RetrievalResult],
    top_k: int = 15,
    batch_size: int = 10,
) -> list[RetrievalResult]:
    """Rerank using LLM batch scoring.

    Scores chunks in batches, returns top_k by LLM score.
    """
    if not chunks:
        return []

    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        all_scores = []

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            batch_scores = await batch_score_chunks(client, query, batch, batch_start)
            all_scores.extend(batch_scores)

        # Sort by LLM score
        all_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        reranked = [chunks[idx] for idx, _ in all_scores[:top_k]]

        logger.info(
            "batch_score_complete",
            input_chunks=len(chunks),
            output_chunks=len(reranked),
            top_score=all_scores[0][1] if all_scores else 0.0,
        )

        return reranked


async def evaluate_llm_reranking(
    project: Project,
    test_cases: list[dict],
    method: str = "binary",  # "binary" or "batch"
    initial_k: int = 100,
    final_k: int = 15,
    max_cases: int | None = None,
) -> dict:
    """Evaluate LLM-based reranking on test cases.

    Args:
        project: Konte project to query.
        test_cases: List of test cases with 'input' and 'expected_output'.
        method: "binary" for binary filtering, "batch" for batch scoring.
        initial_k: Number of candidates to retrieve before reranking.
        final_k: Number of chunks after reranking (sent to answer LLM).
        max_cases: Limit number of test cases (for quick testing).

    Returns:
        Dict with evaluation results.
    """
    from konte.generator import generate_answer

    cases = test_cases[:max_cases] if max_cases else test_cases
    results = []

    rerank_fn = rerank_with_binary_filter if method == "binary" else rerank_with_batch_scoring

    for tc in tqdm(cases, desc=f"LLM Rerank ({method})"):
        query = tc["input"]
        expected = tc.get("expected_output", "")

        try:
            # Step 1: Get initial candidates (hybrid)
            response = project.query(query, mode="hybrid", top_k=initial_k)

            # Step 2: LLM rerank to final_k
            reranked = await rerank_fn(query, response.results, top_k=final_k)

            # Step 3: Generate answer with reranked chunks
            # Create a fake response with reranked results
            from konte.models import RetrievalResponse
            reranked_response = RetrievalResponse(
                results=reranked,
                query=query,
                total_found=len(reranked),
                top_score=reranked[0].score if reranked else 0.0,
                score_spread=0.0,
                has_high_confidence=True,
                suggested_action="deliver",
            )

            answer = await generate_answer(
                question=query,
                retrieval_response=reranked_response,
                max_chunks=final_k,
            )

            results.append({
                "input": query,
                "expected_output": expected,
                "actual_output": answer.answer,
                "retrieval_context": [r.content for r in reranked],
                "initial_chunks": len(response.results),
                "reranked_chunks": len(reranked),
                "status": "success",
            })

        except Exception as e:
            logger.error("evaluation_failed", query=query[:50], error=str(e))
            results.append({
                "input": query,
                "expected_output": expected,
                "actual_output": f"Error: {e}",
                "retrieval_context": [],
                "status": "error",
            })

    # Summary
    success = sum(1 for r in results if r["status"] == "success")

    return {
        "method": method,
        "initial_k": initial_k,
        "final_k": final_k,
        "total_cases": len(cases),
        "success": success,
        "results": results,
    }


async def main():
    """Run LLM reranking experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Reranking Experiment")
    parser.add_argument("--project", default="wco_hs_explanatory_notes_korean")
    parser.add_argument("--test-cases", type=Path,
                        default=Path("evaluation/data/synthetic/synthetic_goldens_korean_v2.json"))
    parser.add_argument("--method", choices=["binary", "batch"], default="binary")
    parser.add_argument("--initial-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=10, help="Limit cases for quick test")
    parser.add_argument("--output", type=Path, default=Path("evaluation/experiments/llm_rerank_results.json"))
    args = parser.parse_args()

    # Load project
    print(f"Loading project: {args.project}")
    project = Project.open(args.project)
    print(f"Project loaded with {len(project._chunks)} chunks")

    # Load test cases
    print(f"Loading test cases from {args.test_cases}")
    with open(args.test_cases) as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    # Run evaluation
    print(f"\nRunning LLM reranking experiment (method={args.method})")
    print(f"  Initial candidates: {args.initial_k}")
    print(f"  Final chunks: {args.final_k}")
    print(f"  Max cases: {args.max_cases or 'all'}")

    results = await evaluate_llm_reranking(
        project=project,
        test_cases=test_cases,
        method=args.method,
        initial_k=args.initial_k,
        final_k=args.final_k,
        max_cases=args.max_cases,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")
    print(f"Success: {results['success']}/{results['total_cases']}")


if __name__ == "__main__":
    asyncio.run(main())
