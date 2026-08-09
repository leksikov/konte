"""Hybrid retriever with reciprocal rank fusion."""

import random

import structlog

from konte.config import settings
from konte.models import (
    ContextualizedChunk,
    MetadataFilter,
    RetrievalMode,
    RetrievalResponse,
    RetrievalResult,
    SuggestedAction,
)
from konte.query_processor import extract_search_keywords
from konte.stores.bm25_store import BM25Store
from konte.stores.faiss_store import FAISSStore
from konte.stores.reranker import rerank_chunks_with_score

logger = structlog.get_logger()

ScoredChunks = list[tuple[ContextualizedChunk, float]]

RRF_K = 60
_FUSION_CANDIDATE_MULTIPLIER = 2
_INJECTED_EVIDENCE_SCORE = 0.95


def reciprocal_rank_fusion(
    results_list: list[ScoredChunks],
    k: int = RRF_K,
) -> ScoredChunks:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each result list

    Args:
        results_list: List of result lists, each containing (chunk, score) tuples.
        k: Constant to prevent high ranks from dominating (default 60).

    Returns:
        Combined list of (chunk, score) tuples sorted by RRF score, rescaled so
        the best result scores 1.0.
    """
    fused: dict[str, tuple[ContextualizedChunk, float]] = {}

    for results in results_list:
        for rank, (chunk, _) in enumerate(results):
            chunk_id = chunk.chunk.chunk_id
            contribution = 1.0 / (k + rank + 1)  # rank is 0-indexed
            existing = fused.get(chunk_id)
            if existing is None:
                fused[chunk_id] = (chunk, contribution)
            else:
                fused[chunk_id] = (existing[0], existing[1] + contribution)

    ranked = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    if not ranked:
        return []

    top_score = ranked[0][1]
    if top_score <= 0:
        return ranked
    return [(chunk, score / top_score) for chunk, score in ranked]


def _determine_suggested_action(top_score: float) -> SuggestedAction:
    """Determine suggested action based on top score.

    Args:
        top_score: The highest retrieval score.

    Returns:
        Suggested action for the agent.
    """
    if top_score >= settings.CONFIDENCE_THRESHOLD:
        return "deliver"
    if top_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
        return "query_more"
    return "refine_query"


def _build_retrieval_response(
    query: str,
    results: ScoredChunks,
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

    top_score = results[0][1] if results else 0.0
    bottom_score = results[-1][1] if results else 0.0

    return RetrievalResponse(
        results=[
            RetrievalResult(
                content=chunk.chunk.content,
                context=chunk.context,
                score=score,
                source=chunk.chunk.source,
                chunk_id=chunk.chunk.chunk_id,
                metadata=chunk.chunk.metadata,
            )
            for chunk, score in results
        ],
        query=query,
        total_found=len(results),
        top_score=top_score,
        score_spread=top_score - bottom_score,
        has_high_confidence=top_score >= settings.CONFIDENCE_THRESHOLD,
        suggested_action=_determine_suggested_action(top_score),
    )


def _inject_evidence_result(
    response: RetrievalResponse,
    evidence: str,
    position: int | None = None,
) -> RetrievalResponse:
    """Inject evidence at specified position for ablation study.

    Args:
        response: Original retrieval response.
        evidence: Evidence text to inject.
        position: Position to insert (0=top, None=random).

    Returns:
        New RetrievalResponse with evidence injected.
    """
    evidence_result = RetrievalResult(
        content=evidence,
        context="[Ablation study: injected evidence]",
        score=_INJECTED_EVIDENCE_SCORE,
        source="ablation_study",
        chunk_id="INJECTED_EVIDENCE",
        metadata={"injected": True},
    )

    if position is None:
        insert_at = random.randint(0, len(response.results))
    else:
        insert_at = min(position, len(response.results))

    new_results = list(response.results)
    new_results.insert(insert_at, evidence_result)

    return RetrievalResponse(
        results=new_results,
        query=response.query,
        total_found=len(new_results),
        top_score=new_results[0].score if new_results else 0.0,
        score_spread=response.score_spread,
        has_high_confidence=True,
        suggested_action="deliver",
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

    @property
    def _has_semantic(self) -> bool:
        """True when a vector index is attached and non-empty."""
        return self._faiss is not None and not self._faiss.is_empty

    @property
    def _has_lexical(self) -> bool:
        """True when a lexical index is attached and non-empty."""
        return self._bm25 is not None and not self._bm25.is_empty

    def retrieve_semantic(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResponse:
        """Retrieve using FAISS semantic search only.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with results.
        """
        k = top_k or settings.DEFAULT_TOP_K
        if not self._has_semantic:
            logger.warning("semantic_retrieval_no_index")

        results = self._semantic_results(query, k, metadata_filter, source_filter)
        return _build_retrieval_response(query, results, k)

    def retrieve_lexical(
        self,
        query: str,
        top_k: int | None = None,
        use_keyword_extraction: bool = True,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResponse:
        """Retrieve using BM25 lexical search only.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            use_keyword_extraction: If True, extract keywords for better Korean BM25.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with results.
        """
        k = top_k or settings.DEFAULT_TOP_K
        if not self._has_lexical:
            logger.warning("lexical_retrieval_no_index")

        results = self._lexical_results(
            query, k, use_keyword_extraction, metadata_filter, source_filter
        )
        return _build_retrieval_response(query, results, k)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int | None = None,
        use_keyword_extraction: bool = True,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResponse:
        """Retrieve using both FAISS and BM25 with rank fusion.

        Falls back to single index if only one is available.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            use_keyword_extraction: If True, extract keywords for better Korean BM25.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with results.
        """
        k = top_k or settings.DEFAULT_TOP_K

        if not self._has_semantic and not self._has_lexical:
            logger.warning("hybrid_retrieval_no_indexes")
        elif not self._has_semantic:
            logger.warning("hybrid_fallback_to_lexical")
        elif not self._has_lexical:
            logger.warning("hybrid_fallback_to_semantic")

        results = self._hybrid_results(
            query, k, use_keyword_extraction, metadata_filter, source_filter
        )
        return _build_retrieval_response(query, results, k)

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
        inject_evidence: str | None = None,
        inject_position: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve documents using specified mode.

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.
            inject_evidence: For ablation study - inject this text.
            inject_position: Position to inject (0=top, None=random).

        Returns:
            RetrievalResponse with results and agent hints.
        """
        if mode == "semantic":
            response = self.retrieve_semantic(
                query, top_k=top_k, metadata_filter=metadata_filter, source_filter=source_filter
            )
        elif mode == "lexical":
            response = self.retrieve_lexical(
                query, top_k=top_k, metadata_filter=metadata_filter, source_filter=source_filter
            )
        else:
            response = self.retrieve_hybrid(
                query, top_k=top_k, metadata_filter=metadata_filter, source_filter=source_filter
            )

        if inject_evidence:
            response = _inject_evidence_result(response, inject_evidence, inject_position)

        return response

    async def retrieve_with_rerank(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        initial_k: int = 50,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResponse:
        """Retrieve with reranking via the configured reranker endpoint.

        Requires settings.RERANKER_BASE_URL (see also RERANKER_MODEL).
        First retrieves initial_k candidates, then reranks to get top_k.

        Args:
            query: Query string.
            mode: Initial retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Final number of results after reranking.
            initial_k: Number of candidates to retrieve before reranking.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with reranked results.
        """
        k = top_k or settings.DEFAULT_TOP_K

        initial_results = self._results_for_mode(
            query,
            mode,
            initial_k,
            # Deliberate: reranked hybrid has always fed BM25 the raw query.
            # Enabling extraction here would shift every reranked hybrid result.
            use_keyword_extraction=mode != "hybrid",
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )
        if not initial_results:
            return _build_retrieval_response(query, [], k)

        reranked = await rerank_chunks_with_score(query, initial_results, top_k=k)
        return _build_retrieval_response(query, reranked, k)

    def _results_for_mode(
        self,
        query: str,
        mode: RetrievalMode,
        top_k: int,
        use_keyword_extraction: bool,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> ScoredChunks:
        """Rank with one mode without wrapping the outcome in a response."""
        if mode == "semantic":
            return self._semantic_results(query, top_k, metadata_filter, source_filter)
        if mode == "lexical":
            return self._lexical_results(
                query, top_k, use_keyword_extraction, metadata_filter, source_filter
            )
        return self._hybrid_results(
            query, top_k, use_keyword_extraction, metadata_filter, source_filter
        )

    def _semantic_results(
        self,
        query: str,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> ScoredChunks:
        """Rank with the vector index alone; empty when it is unavailable."""
        if self._faiss is None or self._faiss.is_empty:
            return []
        return self._faiss.query(
            query, top_k=top_k, metadata_filter=metadata_filter, source_filter=source_filter
        )

    def _lexical_results(
        self,
        query: str,
        top_k: int,
        use_keyword_extraction: bool = True,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> ScoredChunks:
        """Rank with the lexical index alone; empty when it is unavailable.

        BM25 has no notion of stopwords or Korean particles, so a raw
        natural-language question dilutes every term that matters. Keyword
        extraction reduces the query to content words first.
        """
        if self._bm25 is None or self._bm25.is_empty:
            return []

        search_query = query
        if use_keyword_extraction:
            keywords = extract_search_keywords(query)
            search_query = " ".join(keywords)
            logger.debug(
                "bm25_keyword_extraction",
                original_query=query,
                keywords=keywords,
                search_query=search_query,
            )

        return self._bm25.query(
            search_query, top_k=top_k, metadata_filter=metadata_filter, source_filter=source_filter
        )

    def _hybrid_results(
        self,
        query: str,
        top_k: int,
        use_keyword_extraction: bool = True,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> ScoredChunks:
        """Fuse both indexes, degrading to whichever one is available."""
        if not self._has_semantic:
            return self._lexical_results(
                query, top_k, use_keyword_extraction, metadata_filter, source_filter
            )
        if not self._has_lexical:
            return self._semantic_results(query, top_k, metadata_filter, source_filter)

        fetch_k = top_k * _FUSION_CANDIDATE_MULTIPLIER
        return reciprocal_rank_fusion(
            [
                self._semantic_results(query, fetch_k, metadata_filter, source_filter),
                self._lexical_results(
                    query, fetch_k, use_keyword_extraction, metadata_filter, source_filter
                ),
            ]
        )
