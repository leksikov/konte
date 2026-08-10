"""Hybrid retriever with reciprocal rank fusion."""

import random
from collections.abc import Sequence
from typing import NamedTuple

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
from konte.query_processor import extract_search_keywords, extract_search_keywords_async
from konte.stores.bm25_store import BM25Store
from konte.stores.faiss_store import FAISSStore
from konte.stores.reranker import rerank_chunks_with_score

logger = structlog.get_logger()

ScoredChunks = list[tuple[ContextualizedChunk, float]]

RRF_K = 60
_FUSION_CANDIDATE_MULTIPLIER = 2
_INJECTED_EVIDENCE_SCORE = 0.95


class _Ranked(NamedTuple):
    """A ranked list, and how well each of its chunks matched, by chunk id.

    Fusion and lexical normalization both scale their winner to 1.0, so a
    ranking score cannot say whether a chunk answered the query or was merely
    the least bad. `absolute` can: vector similarity, lexical coverage, or the
    reranker's score. Only it may reach the confidence hints.
    """

    results: ScoredChunks
    absolute: dict[str, float]


class _Queries(NamedTuple):
    """The text each index is asked to rank against.

    `semantic` is the query exactly as the caller wrote it — an embedding model
    reads a natural-language question better than a bag of keywords. `lexical`
    is the same string unless keyword extraction reduced it for BM25.

    Resolving both up front lifts the one network-bound step out of the ranking
    helpers, leaving those pure and the async entry points an await apart.
    """

    semantic: str
    lexical: str


def _resolve_keyword_extraction(override: bool | None) -> bool:
    """Settle whether BM25 searches extracted keywords or the raw query."""
    if override is None:
        return settings.BM25_KEYWORD_EXTRACTION
    return override


def _lexical_query(query: str, keywords: Sequence[str]) -> str:
    """Assemble the string BM25 will tokenize from an extraction result.

    An extraction that keeps nothing — an empty list from the model, or a
    question made entirely of stopwords reaching the fallback — would search
    for the empty string, scoring every chunk zero. The original query stands in.
    """
    search_query = " ".join(keywords)
    logger.debug(
        "bm25_keyword_extraction",
        original_query=query,
        keywords=keywords,
        search_query=search_query,
    )
    return search_query or query


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
        the best result scores 1.0. Ranking only: the rescaling erases how well
        anything matched. See _Ranked.
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


def _by_chunk_id(results: ScoredChunks) -> dict[str, float]:
    """Read ranking scores as absolute ones, for the indexes where they are."""
    return {chunk.chunk.chunk_id: score for chunk, score in results}


def _merge_absolute(*readings: dict[str, float]) -> dict[str, float]:
    """Keep the strongest reading per chunk, so fusion cannot penalize agreement."""
    merged: dict[str, float] = {}
    for reading in readings:
        for chunk_id, score in reading.items():
            if score > merged.get(chunk_id, 0.0):
                merged[chunk_id] = score
    return merged


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
    ranked: _Ranked,
    top_k: int,
) -> RetrievalResponse:
    """Build RetrievalResponse from retrieval results.

    The hints read `ranked.absolute`, never the ranking scores: those are
    rescaled per response, so a threshold on them fires on everything.

    Args:
        query: The original query.
        ranked: Ranked results and their absolute match readings.
        top_k: Number of results to include.

    Returns:
        RetrievalResponse with agent decision hints.
    """
    results = ranked.results[:top_k]
    measured = [ranked.absolute.get(chunk.chunk.chunk_id, 0.0) for chunk, _ in results]

    top_score = max(measured, default=0.0)
    bottom_score = min(measured, default=0.0)

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
        top_score=_INJECTED_EVIDENCE_SCORE,
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
        return self._respond(
            _Queries(query, query),
            "semantic",
            top_k,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )

    def retrieve_lexical(
        self,
        query: str,
        top_k: int | None = None,
        use_keyword_extraction: bool | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResponse:
        """Retrieve using BM25 lexical search only.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            use_keyword_extraction: Extract keywords before BM25 search, at the
                cost of one LLM call. None follows settings.BM25_KEYWORD_EXTRACTION.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with results.
        """
        return self._respond(
            self._queries(query, "lexical", use_keyword_extraction),
            "lexical",
            top_k,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int | None = None,
        use_keyword_extraction: bool | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResponse:
        """Retrieve using both FAISS and BM25 with rank fusion.

        Falls back to single index if only one is available.

        Args:
            query: Query string.
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            use_keyword_extraction: Extract keywords before BM25 search, at the
                cost of one LLM call. None follows settings.BM25_KEYWORD_EXTRACTION.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with results.
        """
        return self._respond(
            self._queries(query, "hybrid", use_keyword_extraction),
            "hybrid",
            top_k,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        use_keyword_extraction: bool | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
        inject_evidence: str | None = None,
        inject_position: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve documents using specified mode.

        Blocks on keyword extraction when that is enabled; retrieve_async is
        the variant that does not.

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            use_keyword_extraction: Extract keywords before BM25 search, at the
                cost of one LLM call. None follows settings.BM25_KEYWORD_EXTRACTION.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.
            inject_evidence: For ablation study - inject this text.
            inject_position: Position to inject (0=top, None=random).

        Returns:
            RetrievalResponse with results and agent hints.
        """
        return self._respond(
            self._queries(query, mode, use_keyword_extraction),
            mode,
            top_k,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
            inject_evidence=inject_evidence,
            inject_position=inject_position,
        )

    async def retrieve_async(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        use_keyword_extraction: bool | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
        inject_evidence: str | None = None,
        inject_position: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve documents using specified mode, without blocking the loop.

        Identical to retrieve() except that keyword extraction is awaited;
        ranking is an in-memory index lookup with nothing to await.

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            use_keyword_extraction: Extract keywords before BM25 search, at the
                cost of one LLM call. None follows settings.BM25_KEYWORD_EXTRACTION.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.
            inject_evidence: For ablation study - inject this text.
            inject_position: Position to inject (0=top, None=random).

        Returns:
            RetrievalResponse with results and agent hints.
        """
        return self._respond(
            await self._queries_async(query, mode, use_keyword_extraction),
            mode,
            top_k,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
            inject_evidence=inject_evidence,
            inject_position=inject_position,
        )

    async def retrieve_with_rerank(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        initial_k: int = 50,
        use_keyword_extraction: bool | None = None,
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
            use_keyword_extraction: Extract keywords for better Korean BM25, at
                the cost of one LLM call. Unset means off for hybrid (see
                below) and settings.BM25_KEYWORD_EXTRACTION otherwise.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with reranked results.
        """
        k = top_k or settings.DEFAULT_TOP_K

        # Deliberate: reranked hybrid has always fed BM25 the raw query, so an
        # unset flag stays off here instead of following the configured
        # default. Enabling it would shift every reranked hybrid result.
        if use_keyword_extraction is None and mode == "hybrid":
            use_keyword_extraction = False

        queries = await self._queries_async(query, mode, use_keyword_extraction)
        initial = self._ranked(
            queries,
            mode,
            initial_k,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )
        if not initial.results:
            return _build_retrieval_response(query, initial, k)

        # An unreachable reranker hands back the retrieval's own ranking scores,
        # which would read as a perfect match; keep the initial readings.
        outcome = await rerank_chunks_with_score(query, initial.results, top_k=k)
        absolute = _by_chunk_id(outcome.results) if outcome.scored else initial.absolute
        return _build_retrieval_response(query, _Ranked(outcome.results, absolute), k)

    def _queries(
        self,
        query: str,
        mode: RetrievalMode,
        use_keyword_extraction: bool | None,
    ) -> _Queries:
        """Resolve the per-index query text, extracting keywords when asked to."""
        if not self._extraction_applies(mode, use_keyword_extraction):
            return _Queries(query, query)
        return _Queries(query, _lexical_query(query, extract_search_keywords(query)))

    async def _queries_async(
        self,
        query: str,
        mode: RetrievalMode,
        use_keyword_extraction: bool | None,
    ) -> _Queries:
        """Async twin of _queries; the extraction call is the only difference."""
        if not self._extraction_applies(mode, use_keyword_extraction):
            return _Queries(query, query)
        return _Queries(query, _lexical_query(query, await extract_search_keywords_async(query)))

    def _extraction_applies(self, mode: RetrievalMode, override: bool | None) -> bool:
        """True when extraction would change what this retrieval actually reads.

        Semantic mode never reads the lexical query, and a project without a
        lexical index degrades to semantic whatever the mode asked for; neither
        should pay for a keyword call whose result is discarded.
        """
        return (
            mode != "semantic"
            and self._has_lexical
            and _resolve_keyword_extraction(override)
        )

    def _respond(
        self,
        queries: _Queries,
        mode: RetrievalMode,
        top_k: int | None,
        *,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
        inject_evidence: str | None = None,
        inject_position: int | None = None,
    ) -> RetrievalResponse:
        """Rank against already-resolved queries and wrap the outcome."""
        k = top_k or settings.DEFAULT_TOP_K
        ranked = self._ranked(
            queries,
            mode,
            k,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )
        response = _build_retrieval_response(queries.semantic, ranked, k)

        if inject_evidence:
            return _inject_evidence_result(response, inject_evidence, inject_position)
        return response

    def _ranked(
        self,
        queries: _Queries,
        mode: RetrievalMode,
        top_k: int,
        *,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> _Ranked:
        """Rank with one mode without wrapping the outcome in a response."""
        self._warn_missing_indexes(mode)

        if mode == "semantic":
            return self._semantic_results(
                queries.semantic, top_k, metadata_filter, source_filter
            )
        if mode == "lexical":
            return self._lexical_results(queries.lexical, top_k, metadata_filter, source_filter)
        return self._hybrid_results(queries, top_k, metadata_filter, source_filter)

    def _warn_missing_indexes(self, mode: RetrievalMode) -> None:
        """Report the indexes this mode asked for and did not get."""
        if mode == "semantic":
            if not self._has_semantic:
                logger.warning("semantic_retrieval_no_index")
        elif mode == "lexical":
            if not self._has_lexical:
                logger.warning("lexical_retrieval_no_index")
        elif not self._has_semantic and not self._has_lexical:
            logger.warning("hybrid_retrieval_no_indexes")
        elif not self._has_semantic:
            logger.warning("hybrid_fallback_to_lexical")
        elif not self._has_lexical:
            logger.warning("hybrid_fallback_to_semantic")

    def _semantic_results(
        self,
        query: str,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> _Ranked:
        """Rank with the vector index alone; empty when it is unavailable.

        Similarity compares the query to the chunk, not the chunk to its
        rivals, so it is already the absolute reading.
        """
        if self._faiss is None or self._faiss.is_empty:
            return _Ranked([], {})
        results = self._faiss.query(
            query, top_k=top_k, metadata_filter=metadata_filter, source_filter=source_filter
        )
        return _Ranked(results, _by_chunk_id(results))

    def _lexical_results(
        self,
        search_query: str,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> _Ranked:
        """Rank with the lexical index alone; empty when it is unavailable.

        BM25 has no notion of stopwords or Korean particles, so a raw
        natural-language question dilutes every term that matters. The caller
        passes the already-reduced query; see _Queries.

        Normalization hands the winner 1.0 whether it matched the whole query
        or a single stopword, so coverage is the absolute reading instead.
        """
        if self._bm25 is None or self._bm25.is_empty:
            return _Ranked([], {})
        return _Ranked(
            *self._bm25.query_with_coverage(
                search_query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                source_filter=source_filter,
            )
        )

    def _hybrid_results(
        self,
        queries: _Queries,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> _Ranked:
        """Fuse both indexes, degrading to whichever one is available.

        Fusion consumes rank positions only, so each index's reading is carried
        around it rather than through it.
        """
        if not self._has_semantic:
            return self._lexical_results(
                queries.lexical, top_k, metadata_filter, source_filter
            )
        if not self._has_lexical:
            return self._semantic_results(
                queries.semantic, top_k, metadata_filter, source_filter
            )

        fetch_k = top_k * _FUSION_CANDIDATE_MULTIPLIER
        semantic = self._semantic_results(
            queries.semantic, fetch_k, metadata_filter, source_filter
        )
        lexical = self._lexical_results(queries.lexical, fetch_k, metadata_filter, source_filter)
        return _Ranked(
            reciprocal_rank_fusion([semantic.results, lexical.results]),
            _merge_absolute(semantic.absolute, lexical.absolute),
        )
