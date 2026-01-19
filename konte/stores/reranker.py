"""Reranker module using Qwen3-Reranker-8B via vLLM."""

import asyncio

import httpx
import structlog

from konte.models import ContextualizedChunk

logger = structlog.get_logger()

# Default reranker endpoints
RERANKER_BASE_URL = "https://qwen_reranker.asia03.app.backend.ai/v1"
RERANKER_ENDPOINT = f"{RERANKER_BASE_URL}/rerank"
SCORE_ENDPOINT = f"{RERANKER_BASE_URL}/score"
RERANKER_MODEL = "Qwen3-Reranker-8B"

# Batch size for legacy /rerank endpoint (kept for reference)
RERANK_BATCH_SIZE = 10

# Concurrency for score-based reranking
SCORE_CONCURRENCY = 20


# Max chars for reranking - balance between length bias and including answer
MAX_RERANK_CHARS = 1200


async def _score_single_chunk(
    client: httpx.AsyncClient,
    query: str,
    chunk: ContextualizedChunk,
    idx: int,
    model: str,
    semaphore: asyncio.Semaphore,
    max_chars: int = MAX_RERANK_CHARS,
) -> tuple[int, float]:
    """Score a single (query, document) pair using /score endpoint.

    Combines context (summary) + truncated raw content for best results:
    - Context provides key terms (HS codes, products)
    - Raw content provides actual document text
    - Total length capped to avoid length bias
    """
    async with semaphore:
        try:
            # Combine: context (summary) + raw content, capped at max_chars
            context = chunk.context or ""
            raw = chunk.chunk.content
            # Context first (has key terms), then raw content
            doc_text = f"{context} {raw}"[:max_chars]
            payload = {
                "model": model,
                "text_1": query,
                "text_2": doc_text,
            }
            response = await client.post(
                SCORE_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            score = data["data"][0]["score"]
            return (idx, score)
        except Exception as e:
            logger.warning("score_chunk_failed", idx=idx, error=str(e))
            return (idx, 0.0)


async def rerank_chunks_with_score(
    query: str,
    chunks: list[tuple[ContextualizedChunk, float]],
    top_k: int | None = None,
    model: str = RERANKER_MODEL,
    concurrency: int = SCORE_CONCURRENCY,
) -> list[tuple[ContextualizedChunk, float]]:
    """Rerank chunks using /score endpoint for each (query, doc) pair.

    This approach gives consistent scores compared to batch /rerank endpoint.

    Args:
        query: The search query.
        chunks: List of (chunk, score) tuples from initial retrieval.
        top_k: Number of top results to return. Defaults to len(chunks).
        model: Reranker model name.
        concurrency: Max concurrent score requests.

    Returns:
        Reranked list of (chunk, relevance_score) tuples.
    """
    if not chunks:
        return []

    k = top_k or len(chunks)
    semaphore = asyncio.Semaphore(concurrency)

    try:
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            # Score all chunks concurrently (with semaphore limit)
            tasks = [
                _score_single_chunk(client, query, chunk, idx, model, semaphore)
                for idx, (chunk, _) in enumerate(chunks)
            ]
            all_scores = await asyncio.gather(*tasks)

            # Sort by score descending
            sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)

            # Build reranked list
            reranked = [(chunks[idx][0], score) for idx, score in sorted_scores[:k]]

            logger.info(
                "rerank_with_score_complete",
                query_len=len(query),
                input_chunks=len(chunks),
                output_chunks=len(reranked),
                top_score=reranked[0][1] if reranked else 0.0,
            )

            return reranked

    except Exception as e:
        logger.error("rerank_with_score_failed", error=str(e))
        # Fallback to original order
        return chunks[:k]


async def rerank_chunks(
    query: str,
    chunks: list[tuple[ContextualizedChunk, float]],
    top_k: int | None = None,
    endpoint: str = RERANKER_ENDPOINT,
    model: str = RERANKER_MODEL,
    batch_size: int = RERANK_BATCH_SIZE,
    instruction: str | None = None,
) -> list[tuple[ContextualizedChunk, float]]:
    """Rerank chunks using Qwen3-Reranker-8B.

    Uses batched reranking to avoid model issues with large document sets.

    Args:
        query: The search query.
        chunks: List of (chunk, score) tuples from initial retrieval.
        top_k: Number of top results to return. Defaults to len(chunks).
        endpoint: Reranker API endpoint.
        model: Reranker model name.
        batch_size: Max documents per rerank call (default 10).
        instruction: Custom instruction for the reranker task.

    Returns:
        Reranked list of (chunk, relevance_score) tuples.
    """
    if not chunks:
        return []

    k = top_k or len(chunks)
    inst = instruction or DEFAULT_INSTRUCTION

    try:
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            # Process in batches
            all_scores: list[tuple[int, float]] = []  # (original_idx, score)

            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                # Extract documents for this batch
                documents = [chunk.contextualized_content for chunk, _ in batch_chunks]

                # Include instruction in payload
                payload = {
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "instruction": inst,
                }

                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                # Map batch results back to original indices
                for result in data.get("results", []):
                    batch_idx = result["index"]
                    original_idx = batch_start + batch_idx
                    score = result["relevance_score"]
                    all_scores.append((original_idx, score))

            # Sort by score descending
            all_scores.sort(key=lambda x: x[1], reverse=True)

            # Build reranked list
            reranked = [(chunks[idx][0], score) for idx, score in all_scores[:k]]

            logger.info(
                "rerank_complete",
                query_len=len(query),
                input_chunks=len(chunks),
                output_chunks=len(reranked),
                num_batches=(len(chunks) + batch_size - 1) // batch_size,
                top_score=reranked[0][1] if reranked else 0.0,
            )

            return reranked

    except Exception as e:
        logger.error("rerank_failed", error=str(e))
        # Fallback to original order
        return chunks[:k]
