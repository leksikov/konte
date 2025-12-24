"""Hybrid retriever with reciprocal rank fusion."""

from typing import Literal

import structlog

from konte.config import settings
from konte.models import ContextualizedChunk, RetrievalResponse, RetrievalResult
from konte.stores.bm25_store import BM25Store
from konte.stores.faiss_store import FAISSStore
from konte.stores.reranker import rerank_chunks_with_score

logger = structlog.get_logger()

RetrievalMode = Literal["hybrid", "semantic", "lexical"]


def reciprocal_rank_fusion(
    results_list: list[list[tuple[ContextualizedChunk, float]]],
    k: int = 60,
) -> list[tuple[ContextualizedChunk, float]]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each result list

    Args:
        results_list: List of result lists, each containing (chunk, score) tuples.
        k: Constant to prevent high ranks from dominating (default 60).

    Returns:
        Combined list of (chunk, score) tuples sorted by RRF score.
    """
    # Map chunk_id to (chunk, rrf_score)
    rrf_scores: dict[str, tuple[ContextualizedChunk, float]] = {}

    for results in results_list:
        for rank, (chunk, _) in enumerate(results):
            chunk_id = chunk.chunk.chunk_id
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed

            if chunk_id in rrf_scores:
                existing_chunk, existing_score = rrf_scores[chunk_id]
                rrf_scores[chunk_id] = (existing_chunk, existing_score + rrf_score)
            else:
                rrf_scores[chunk_id] = (chunk, rrf_score)

    # Sort by RRF score descending
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Normalize scores to 0-1 range
    if sorted_results:
        max_score = sorted_results[0][1]
        if max_score > 0:
            sorted_results = [
                (chunk, score / max_score) for chunk, score in sorted_results
            ]

    return sorted_results


def _determine_suggested_action(top_score: float) -> Literal["deliver", "query_more", "refine_query"]:
    """Determine suggested action based on top score.

    Args:
        top_score: The highest retrieval score.

    Returns:
        Suggested action for the agent.
    """
    if top_score >= settings.CONFIDENCE_THRESHOLD:
        return "deliver"
    elif top_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
        return "query_more"
    else:
        return "refine_query"


def _build_retrieval_response(
    query: str,
    results: list[tuple[ContextualizedChunk, float]],
    top_k: int,
) -> RetrievalResponse:
    """Build RetrievalResponse from retrieval results.

    Args:
        query: The original query.
        results: List of (chunk, score) tuples.
        top_k: Number of results to include.

    Returns:
        RetrievalResponse with agent decision hints.
    """
    results = results[:top_k]

    retrieval_results = [
        RetrievalResult(
            content=chunk.chunk.content,
            context=chunk.context,
            score=score,
            source=chunk.chunk.source,
            chunk_id=chunk.chunk.chunk_id,
            metadata=chunk.chunk.metadata,
        )
        for chunk, score in results
    ]

    top_score = results[0][1] if results else 0.0
    bottom_score = results[-1][1] if results else 0.0
    score_spread = top_score - bottom_score

    return RetrievalResponse(
        results=retrieval_results,
        query=query,
        total_found=len(results),
        top_score=top_score,
        score_spread=score_spread,
        has_high_confidence=top_score >= settings.CONFIDENCE_THRESHOLD,
        suggested_action=_determine_suggested_action(top_score),
    )


class Retriever:
    """Hybrid retriever combining FAISS and BM25 search."""

    def __init__(
        self,
        faiss_store: FAISSStore | None = None,
        bm25_store: BM25Store | None = None,
    ):
        """Initialize retriever with stores.

        Args:
            faiss_store: FAISS store for semantic search.
            bm25_store: BM25 store for lexical search.
        """
        self._faiss = faiss_store
        self._bm25 = bm25_store

    def retrieve_semantic(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve using FAISS semantic search only.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            RetrievalResponse with results.
        """
        k = top_k or settings.DEFAULT_TOP_K

        if self._faiss is None or self._faiss.is_empty:
            logger.warning("semantic_retrieval_no_index")
            return _build_retrieval_response(query, [], k)

        results = self._faiss.query(query, top_k=k)
        return _build_retrieval_response(query, results, k)

    def retrieve_lexical(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve using BM25 lexical search only.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            RetrievalResponse with results.
        """
        k = top_k or settings.DEFAULT_TOP_K

        if self._bm25 is None or self._bm25.is_empty:
            logger.warning("lexical_retrieval_no_index")
            return _build_retrieval_response(query, [], k)

        results = self._bm25.query(query, top_k=k)
        return _build_retrieval_response(query, results, k)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve using both FAISS and BM25 with rank fusion.

        Falls back to single index if only one is available.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            RetrievalResponse with results.
        """
        k = top_k or settings.DEFAULT_TOP_K

        has_faiss = self._faiss is not None and not self._faiss.is_empty
        has_bm25 = self._bm25 is not None and not self._bm25.is_empty

        if not has_faiss and not has_bm25:
            logger.warning("hybrid_retrieval_no_indexes")
            return _build_retrieval_response(query, [], k)

        if not has_faiss:
            logger.warning("hybrid_fallback_to_lexical")
            return self.retrieve_lexical(query, top_k=k)

        if not has_bm25:
            logger.warning("hybrid_fallback_to_semantic")
            return self.retrieve_semantic(query, top_k=k)

        # Get results from both indexes (more than top_k to allow for fusion)
        fetch_k = k * 2
        faiss_results = self._faiss.query(query, top_k=fetch_k)
        bm25_results = self._bm25.query(query, top_k=fetch_k)

        # Fuse results
        fused = reciprocal_rank_fusion([faiss_results, bm25_results])

        return _build_retrieval_response(query, fused, k)

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve documents using specified mode.

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            RetrievalResponse with results and agent hints.
        """
        if mode == "semantic":
            return self.retrieve_semantic(query, top_k=top_k)
        elif mode == "lexical":
            return self.retrieve_lexical(query, top_k=top_k)
        else:
            return self.retrieve_hybrid(query, top_k=top_k)

    async def retrieve_with_rerank(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        initial_k: int = 50,
    ) -> RetrievalResponse:
        """Retrieve with reranking using Qwen3-Reranker-8B.

        First retrieves initial_k candidates, then reranks to get top_k.

        Args:
            query: Query string.
            mode: Initial retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Final number of results after reranking.
            initial_k: Number of candidates to retrieve before reranking.

        Returns:
            RetrievalResponse with reranked results.
        """
        k = top_k or settings.DEFAULT_TOP_K

        # Get initial candidates
        if mode == "semantic":
            initial_results = self._get_semantic_results(query, initial_k)
        elif mode == "lexical":
            initial_results = self._get_lexical_results(query, initial_k)
        else:
            initial_results = self._get_hybrid_results(query, initial_k)

        if not initial_results:
            return _build_retrieval_response(query, [], k)

        # Rerank candidates using score endpoint (reliable individual scoring)
        reranked = await rerank_chunks_with_score(query, initial_results, top_k=k)

        return _build_retrieval_response(query, reranked, k)

    def _get_semantic_results(
        self, query: str, top_k: int
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Get raw semantic results."""
        if self._faiss is None or self._faiss.is_empty:
            return []
        return self._faiss.query(query, top_k=top_k)

    def _get_lexical_results(
        self, query: str, top_k: int
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Get raw lexical results."""
        if self._bm25 is None or self._bm25.is_empty:
            return []
        return self._bm25.query(query, top_k=top_k)

    def _get_hybrid_results(
        self, query: str, top_k: int
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Get raw hybrid results with rank fusion."""
        has_faiss = self._faiss is not None and not self._faiss.is_empty
        has_bm25 = self._bm25 is not None and not self._bm25.is_empty

        if not has_faiss and not has_bm25:
            return []

        if not has_faiss:
            return self._get_lexical_results(query, top_k)

        if not has_bm25:
            return self._get_semantic_results(query, top_k)

        # Get from both and fuse
        fetch_k = top_k * 2
        faiss_results = self._faiss.query(query, top_k=fetch_k)
        bm25_results = self._bm25.query(query, top_k=fetch_k)

        fused = reciprocal_rank_fusion([faiss_results, bm25_results])
        return fused[:top_k]
