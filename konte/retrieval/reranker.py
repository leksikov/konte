"""Reranker module using a vLLM /score endpoint."""

import asyncio
from typing import NamedTuple

import httpx
import structlog

from konte.domain.models import ContextualizedChunk
from konte.runtime.settings import settings

logger = structlog.get_logger()

# Concurrency for score-based reranking
SCORE_CONCURRENCY = 20

# Max chars for reranking - balance between length bias and including answer
MAX_RERANK_CHARS = 1200


class RerankOutcome(NamedTuple):
    """Reranked pairs, and whether the reranker is what scored them.

    `scored` is False when every request failed and `results` still carries the
    retrieval's own ranking scores, which no caller may read as relevance.
    """

    results: list[tuple[ContextualizedChunk, float]]
    scored: bool


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
) -> tuple[int, float | None]:
    """Score a single (query, document) pair using /score endpoint.

    The generated context leads because it carries the identifiers and topic
    terms the raw text often omits; the raw content follows so the reranker sees
    the real wording. The pair is truncated because these models score shorter
    documents higher regardless of relevance.

    Returns:
        (idx, score), with score None if the request failed.
    """
    async with semaphore:
        try:
            doc_text = f"{chunk.context or ''} {chunk.chunk.content}"[:max_chars]
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
            return (idx, None)


async def rerank_chunks_with_score(
    query: str,
    chunks: list[tuple[ContextualizedChunk, float]],
    top_k: int | None = None,
    model: str | None = None,
    concurrency: int = SCORE_CONCURRENCY,
) -> RerankOutcome:
    """Rerank chunks using /score endpoint for each (query, doc) pair.

    This approach gives consistent scores compared to batch /rerank endpoint.

    Args:
        query: The search query.
        chunks: List of (chunk, score) tuples from initial retrieval.
        top_k: Number of top results to return. Defaults to len(chunks).
        model: Reranker model name. Defaults to settings.RERANKER_MODEL,
            which must then be configured.
        concurrency: Max concurrent score requests.

    Returns:
        RerankOutcome carrying the reranked (chunk, relevance_score) tuples, on
        whatever scale the endpoint scores. If every score request fails
        (unreachable endpoint, TLS error, wrong model), the original retrieval
        order and scores come back with `scored` False and an error is logged;
        partial failures sort the failed chunks last at 0.0 and still count as
        scored.

    Raises:
        ValueError: If RERANKER_BASE_URL or the reranker model is not configured.
    """
    if not chunks:
        return RerankOutcome([], False)

    score_endpoint = _resolve_score_endpoint()
    reranker_model = model or settings.RERANKER_MODEL
    if not reranker_model:
        raise ValueError(
            "Reranking requires a model name. Set RERANKER_MODEL to the model "
            "your /score endpoint serves, or pass model= explicitly."
        )
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

            # Every request failed: misconfigured endpoint/model or TLS issue.
            # Raise so the handler below falls back to the original scores
            # instead of returning fabricated all-zero rankings.
            if all(score is None for _, score in all_scores):
                raise RuntimeError(
                    f"all {len(all_scores)} rerank score requests to "
                    f"{score_endpoint} failed; check RERANKER_BASE_URL, "
                    "RERANKER_MODEL, and RERANKER_VERIFY_SSL"
                )

            # Sorting a failed request as 0.0 would rank it above every real
            # result a logit-scaled reranker put below zero.
            sorted_scores = sorted(
                ((idx, score) for idx, score in all_scores if score is not None),
                key=lambda pair: pair[1],
                reverse=True,
            )
            sorted_scores += [(idx, 0.0) for idx, score in all_scores if score is None]

            # Build reranked list
            reranked = [(chunks[idx][0], score) for idx, score in sorted_scores[:k]]

            logger.info(
                "rerank_with_score_complete",
                query_len=len(query),
                input_chunks=len(chunks),
                output_chunks=len(reranked),
                top_score=reranked[0][1] if reranked else 0.0,
            )

            return RerankOutcome(reranked, True)

    except Exception as e:
        logger.error("rerank_with_score_failed", error=str(e))
        # Fallback to original order
        return RerankOutcome(chunks[:k], False)
