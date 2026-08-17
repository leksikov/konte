"""Hybrid retriever with reciprocal rank fusion."""

import structlog

from konte.domain.models import RetrievalMode, RetrievalRequest, RetrievalResponse
from konte.index.bm25_store import BM25Store
from konte.index.faiss_store import FAISSStore
from konte.retrieval.fusion import (
    Ranked,
    by_chunk_id,
    drop_redundant,
    merge_absolute,
    reciprocal_rank_fusion,
)
from konte.retrieval.query_processor import Queries, resolve_queries, resolve_queries_async
from konte.retrieval.reranker import rerank_chunks_with_score
from konte.retrieval.response import build_response, inject_evidence
from konte.runtime.settings import settings

logger = structlog.get_logger()

# Candidates ranked per result returned, so dropping redundant ones still fills top_k.
_CANDIDATE_MULTIPLIER = 2


class Retriever:
    """Hybrid retriever combining FAISS and BM25 search.

    Two entry points, both taking the whole request: retrieve() blocks on
    whatever the request needs, aretrieve() awaits it.
    """

    def __init__(
        self,
        faiss_store: FAISSStore | None = None,
        bm25_store: BM25Store | None = None,
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ):
        """Initialize retriever with stores.

        Args:
            faiss_store: FAISS store for semantic search.
            bm25_store: BM25 store for lexical search.
            semantic_weight: How much a vector rank counts in hybrid fusion.
            lexical_weight: How much a lexical rank counts in hybrid fusion.
        """
        self._faiss = faiss_store
        self._bm25 = bm25_store
        self._fusion_weights = (semantic_weight, lexical_weight)

    @property
    def _has_semantic(self) -> bool:
        """True when a vector index is attached and non-empty."""
        return self._faiss is not None and not self._faiss.is_empty

    @property
    def _has_lexical(self) -> bool:
        """True when a lexical index is attached and non-empty."""
        return self._bm25 is not None and not self._bm25.is_empty

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve documents for one request.

        Blocks on keyword extraction when that is enabled; aretrieve is the
        variant that does not.

        Args:
            request: What this retrieval is asked for.

        Returns:
            RetrievalResponse with results and agent hints.

        Raises:
            ValueError: If the request asks to rerank, which is one network
                request per candidate and so has no blocking form.
        """
        if request.rerank:
            raise ValueError("Reranking sends a request per candidate; await aretrieve() instead.")
        return self._respond(request, resolve_queries(request, self._has_lexical))

    async def aretrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve documents for one request, without blocking the loop.

        Identical to retrieve() except that keyword extraction is awaited and
        reranking is available; ranking itself is an in-memory index lookup
        with nothing to await.

        Args:
            request: What this retrieval is asked for.

        Returns:
            RetrievalResponse with results and agent hints.
        """
        queries = await resolve_queries_async(request, self._has_lexical)
        if not request.rerank:
            return self._respond(request, queries)
        return await self._reranked(request, queries)

    def _respond(self, request: RetrievalRequest, queries: Queries) -> RetrievalResponse:
        """Rank against already-resolved queries and wrap the outcome."""
        top_k = self._top_k(request)
        return self._finish(request, self._ranked(request, queries, top_k), top_k)

    async def _reranked(self, request: RetrievalRequest, queries: Queries) -> RetrievalResponse:
        """Re-score the leading candidates through the configured reranker.

        Requires settings.RERANKER_BASE_URL (see also RERANKER_MODEL).
        """
        top_k = self._top_k(request)
        initial = self._ranked(request, queries, request.rerank_initial_k)
        if not initial.results:
            return self._finish(request, initial, top_k)

        # An unreachable reranker hands back the retrieval's own ranking scores,
        # which would read as a perfect match; keep the initial readings.
        outcome = await rerank_chunks_with_score(request.query, initial.results, top_k=top_k)
        absolute = by_chunk_id(outcome.results) if outcome.scored else initial.absolute
        return self._finish(request, Ranked(outcome.results, absolute), top_k)

    def _finish(
        self,
        request: RetrievalRequest,
        ranked: Ranked,
        top_k: int,
    ) -> RetrievalResponse:
        """Wrap a ranking as a response, injecting ablation evidence if asked."""
        response = build_response(request.query, ranked, top_k)
        if request.inject_evidence:
            return inject_evidence(response, request.inject_evidence, request.inject_position)
        return response

    @staticmethod
    def _top_k(request: RetrievalRequest) -> int:
        """How many results this request asked for."""
        return request.top_k or settings.DEFAULT_TOP_K

    def _ranked(self, request: RetrievalRequest, queries: Queries, top_k: int) -> Ranked:
        """Rank in the request's mode without wrapping the outcome in a response.

        Over-fetching leaves room to drop redundant results without shortening
        the response; fusion already draws on that many, a single index does not.
        """
        self._warn_missing_indexes(request.mode)

        if request.mode == "semantic":
            candidates = self._semantic_results(
                queries.semantic, top_k * _CANDIDATE_MULTIPLIER, request
            )
        elif request.mode == "lexical":
            candidates = self._lexical_results(
                queries.lexical, top_k * _CANDIDATE_MULTIPLIER, request
            )
        else:
            candidates = self._hybrid_results(queries, top_k, request)

        return Ranked(drop_redundant(candidates.results, top_k), candidates.absolute)

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

    def _semantic_results(self, query: str, top_k: int, request: RetrievalRequest) -> Ranked:
        """Rank with the vector index alone; empty when it is unavailable.

        Similarity compares the query to the chunk, not the chunk to its
        rivals, so it is already the absolute reading.
        """
        if self._faiss is None or self._faiss.is_empty:
            return Ranked([], {})
        results = self._faiss.query(
            query,
            top_k=top_k,
            metadata_filter=request.metadata_filter,
            source_filter=request.source_filter,
        )
        return Ranked(results, by_chunk_id(results))

    def _lexical_results(
        self,
        search_query: str,
        top_k: int,
        request: RetrievalRequest,
    ) -> Ranked:
        """Rank with the lexical index alone; empty when it is unavailable.

        BM25 has no notion of stopwords or Korean particles, so a raw
        natural-language question dilutes every term that matters. The caller
        passes the already-reduced query; see Queries.

        Normalization hands the winner 1.0 whether it matched the whole query
        or a single stopword, so coverage is the absolute reading instead.
        """
        if self._bm25 is None or self._bm25.is_empty:
            return Ranked([], {})
        return Ranked(
            *self._bm25.query_with_coverage(
                search_query,
                top_k=top_k,
                metadata_filter=request.metadata_filter,
                source_filter=request.source_filter,
            )
        )

    def _hybrid_results(self, queries: Queries, top_k: int, request: RetrievalRequest) -> Ranked:
        """Fuse both indexes, degrading to whichever one is available.

        Fusion consumes rank positions only, so each index's reading is carried
        around it rather than through it.
        """
        fetch_k = top_k * _CANDIDATE_MULTIPLIER

        if not self._has_semantic:
            return self._lexical_results(queries.lexical, fetch_k, request)
        if not self._has_lexical:
            return self._semantic_results(queries.semantic, fetch_k, request)

        semantic = self._semantic_results(queries.semantic, fetch_k, request)
        lexical = self._lexical_results(queries.lexical, fetch_k, request)
        return Ranked(
            reciprocal_rank_fusion(
                [semantic.results, lexical.results], weights=self._fusion_weights
            ),
            merge_absolute(semantic.absolute, lexical.absolute),
        )
