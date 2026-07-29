"""Reranker module using a vLLM /score endpoint (e.g. Qwen3-Reranker-8B)."""

import asyncio

import httpx
import structlog

from konte.config.settings import settings
from konte.models import ContextualizedChunk

logger = structlog.get_logger()

# Concurrency for score-based reranking
SCORE_CONCURRENCY = 20


# Max chars for reranking - balance between length bias and including answer
MAX_RERANK_CHARS = 1200


def _resolve_score_endpoint() -> str:
    """Return the /score endpoint URL from settings.

    Raises:
        ValueError: If RERANKER_BASE_URL is not configured.
    """
    if not settings.RERANKER_BASE_URL:
        raise ValueError(
            "Reranking requires a reranker endpoint. Set RERANKER_BASE_URL "
            "(e.g. RERANKER_BASE_URL=https://your-vllm-endpoint/v1) to a vLLM "
            "server exposing a /score endpoint, or query with rerank=False."
        )
    return f"{settings.RERANKER_BASE_URL.rstrip('/')}/score"


async def _score_single_chunk(
    client: httpx.AsyncClient,
    query: str,
    chunk: ContextualizedChunk,
    idx: int,
    model: str,
    semaphore: asyncio.Semaphore,
    score_endpoint: str,
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
                score_endpoint,
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
    model: str | None = None,
    concurrency: int = SCORE_CONCURRENCY,
) -> list[tuple[ContextualizedChunk, float]]:
    """Rerank chunks using /score endpoint for each (query, doc) pair.

    This approach gives consistent scores compared to batch /rerank endpoint.

    Args:
        query: The search query.
        chunks: List of (chunk, score) tuples from initial retrieval.
        top_k: Number of top results to return. Defaults to len(chunks).
        model: Reranker model name. Defaults to settings.RERANKER_MODEL.
        concurrency: Max concurrent score requests.

    Returns:
        Reranked list of (chunk, relevance_score) tuples.

    Raises:
        ValueError: If RERANKER_BASE_URL is not configured.
    """
    if not chunks:
        return []

    score_endpoint = _resolve_score_endpoint()
    reranker_model = model or settings.RERANKER_MODEL
    k = top_k or len(chunks)
    semaphore = asyncio.Semaphore(concurrency)

    try:
        async with httpx.AsyncClient(
            timeout=60.0, verify=settings.RERANKER_VERIFY_SSL
        ) as client:
            # Score all chunks concurrently (with semaphore limit)
            tasks = [
                _score_single_chunk(
                    client, query, chunk, idx, reranker_model, semaphore, score_endpoint
                )
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
