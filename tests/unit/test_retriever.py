"""Unit tests for retriever module."""

from unittest.mock import AsyncMock, patch

import pytest

from konte.models import Chunk, ContextualizedChunk
from konte.stores.reranker import RerankOutcome


@pytest.fixture
def sample_chunks():
    """Create sample contextualized chunks for testing."""
    chunks = []
    for i in range(5):
        chunk = Chunk(
            chunk_id=f"test_s0_c{i}",
            content=f"Content for chunk {i}",
            source="test.txt",
            segment_idx=0,
            chunk_idx=i,
        )
        chunks.append(ContextualizedChunk(chunk=chunk, context=f"Context {i}"))
    return chunks


@pytest.mark.unit
class TestReciprocalRankFusion:
    """Test reciprocal rank fusion algorithm."""

    def test_single_list(self, sample_chunks):
        """Test fusion with single result list."""
        from konte.stores.retriever import reciprocal_rank_fusion

        results = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        fused = reciprocal_rank_fusion([results])

        assert len(fused) == 2
        # Order should be preserved
        assert fused[0][0].chunk.chunk_id == "test_s0_c0"
        assert fused[1][0].chunk.chunk_id == "test_s0_c1"

    def test_two_lists_same_order(self, sample_chunks):
        """Test fusion when both lists have same order."""
        from konte.stores.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[0], 0.95), (sample_chunks[1], 0.85)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 2
        assert fused[0][0].chunk.chunk_id == "test_s0_c0"

    def test_two_lists_different_order(self, sample_chunks):
        """Test fusion combines results from different orderings."""
        from konte.stores.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[1], 0.95), (sample_chunks[0], 0.85)]

        fused = reciprocal_rank_fusion([list1, list2])

        # Both should appear in fused results
        chunk_ids = [r[0].chunk.chunk_id for r in fused]
        assert "test_s0_c0" in chunk_ids
        assert "test_s0_c1" in chunk_ids

    def test_disjoint_lists(self, sample_chunks):
        """Test fusion of lists with no overlap."""
        from konte.stores.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[2], 0.95), (sample_chunks[3], 0.85)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 4

    def test_scores_normalized(self, sample_chunks):
        """Test that fused scores are normalized to 0-1."""
        from konte.stores.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9)]
        list2 = [(sample_chunks[0], 0.9)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert 0.0 <= fused[0][1] <= 1.0

    def test_empty_lists(self):
        """Test fusion with empty lists."""
        from konte.stores.retriever import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_equal_weights_match_unweighted(self, sample_chunks):
        """Test a shared weight cancels in the rescaling, whatever its value."""
        from konte.stores.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[1], 0.95), (sample_chunks[2], 0.85)]

        unweighted = reciprocal_rank_fusion([list1, list2])
        weighted = reciprocal_rank_fusion([list1, list2], weights=(0.5, 0.5))

        assert [(c.chunk.chunk_id, s) for c, s in weighted] == [
            (c.chunk.chunk_id, s) for c, s in unweighted
        ]

    def test_weights_shift_the_winner(self, sample_chunks):
        """Test the heavier list's top result wins a head-to-head disagreement."""
        from konte.stores.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[1], 0.95), (sample_chunks[0], 0.85)]

        first_heavy = reciprocal_rank_fusion([list1, list2], weights=(0.8, 0.2))
        second_heavy = reciprocal_rank_fusion([list1, list2], weights=(0.2, 0.8))

        assert first_heavy[0][0].chunk.chunk_id == "test_s0_c0"
        assert second_heavy[0][0].chunk.chunk_id == "test_s0_c1"

    def test_weights_must_cover_every_list(self, sample_chunks):
        """Test a short weight vector is rejected, not silently truncated."""
        from konte.stores.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9)]
        list2 = [(sample_chunks[1], 0.9)]

        with pytest.raises(ValueError):
            reciprocal_rank_fusion([list1, list2], weights=(1.0,))


@pytest.mark.unit
class TestDetermineSuggestedAction:
    """Test suggested action determination."""

    def test_high_score_delivers(self):
        """Test that high scores suggest 'deliver'."""
        from konte.stores.retriever import _determine_suggested_action

        assert _determine_suggested_action(0.9) == "deliver"
        assert _determine_suggested_action(0.7) == "deliver"

    def test_medium_score_query_more(self):
        """Test that medium scores suggest 'query_more'."""
        from konte.stores.retriever import _determine_suggested_action

        assert _determine_suggested_action(0.6) == "query_more"
        assert _determine_suggested_action(0.4) == "query_more"

    def test_low_score_refine_query(self):
        """Test that low scores suggest 'refine_query'."""
        from konte.stores.retriever import _determine_suggested_action

        assert _determine_suggested_action(0.3) == "refine_query"
        assert _determine_suggested_action(0.0) == "refine_query"


def _ranked(results):
    """Wrap results whose ranking scores happen to be absolute, as FAISS's are."""
    from konte.stores.retriever import _by_chunk_id, _Ranked

    return _Ranked(results, _by_chunk_id(results))


@pytest.mark.unit
class TestBuildRetrievalResponse:
    """Test building RetrievalResponse."""

    def test_builds_response_with_results(self, sample_chunks):
        """Test building response from results."""
        from konte.stores.retriever import _build_retrieval_response

        results = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.7),
            (sample_chunks[2], 0.5),
        ]
        response = _build_retrieval_response("test query", _ranked(results), top_k=3)

        assert response.query == "test query"
        assert response.total_found == 3
        assert len(response.results) == 3
        assert response.top_score == 0.9
        assert response.score_spread == 0.4  # 0.9 - 0.5
        assert response.has_high_confidence is True
        assert response.suggested_action == "deliver"

    def test_builds_response_empty_results(self):
        """Test building response with no results."""
        from konte.stores.retriever import _build_retrieval_response, _Ranked

        response = _build_retrieval_response("test query", _Ranked([], {}), top_k=10)

        assert response.total_found == 0
        assert response.top_score == 0.0
        assert response.suggested_action == "refine_query"

    def test_respects_top_k(self, sample_chunks):
        """Test that top_k limits results."""
        from konte.stores.retriever import _build_retrieval_response

        results = [(c, 0.9 - i * 0.1) for i, c in enumerate(sample_chunks)]
        response = _build_retrieval_response("test", _ranked(results), top_k=2)

        assert len(response.results) == 2

    def test_confidence_ignores_the_ranking_scores(self, sample_chunks):
        """Test the hints read the absolute measurements, not the ordering."""
        from konte.stores.retriever import _build_retrieval_response, _Ranked

        # A fused ranking: the winner reads 1.0 however little it matched.
        results = [(sample_chunks[0], 1.0), (sample_chunks[1], 0.6)]
        measured = {"test_s0_c0": 0.31, "test_s0_c1": 0.22}

        response = _build_retrieval_response("test", _Ranked(results, measured), top_k=2)

        assert response.results[0].score == 1.0  # ordering is untouched
        assert response.top_score == pytest.approx(0.31)
        assert response.score_spread == pytest.approx(0.09)
        assert response.has_high_confidence is False
        assert response.suggested_action == "refine_query"

    def test_unmeasured_chunk_scores_zero(self, sample_chunks):
        """Test a chunk with no absolute reading cannot inflate confidence."""
        from konte.stores.retriever import _build_retrieval_response, _Ranked

        results = [(sample_chunks[0], 1.0)]
        response = _build_retrieval_response("test", _Ranked(results, {}), top_k=1)

        assert response.top_score == 0.0
        assert response.suggested_action == "refine_query"


@pytest.mark.unit
class TestRetrieverModes:
    """Test retriever with different modes (using mocked stores)."""

    def test_retrieve_semantic_empty_store(self):
        """Test semantic retrieval with no FAISS store."""
        from konte.stores.retriever import Retriever

        retriever = Retriever(faiss_store=None, bm25_store=None)
        response = retriever.retrieve_semantic("test query")

        assert response.total_found == 0

    def test_retrieve_lexical_empty_store(self):
        """Test lexical retrieval with no BM25 store."""
        from konte.stores.retriever import Retriever

        retriever = Retriever(faiss_store=None, bm25_store=None)
        response = retriever.retrieve_lexical("test query")

        assert response.total_found == 0

    def test_retrieve_hybrid_no_stores(self):
        """Test hybrid retrieval with no stores."""
        from konte.stores.retriever import Retriever

        retriever = Retriever(faiss_store=None, bm25_store=None)
        response = retriever.retrieve_hybrid("test query")

        assert response.total_found == 0

    def test_retrieve_default_mode_is_hybrid(self):
        """Test that default mode is hybrid."""
        from konte.stores.retriever import Retriever

        retriever = Retriever()
        response = retriever.retrieve("test query")

        # Should complete without error, even with empty stores
        assert response.query == "test query"


class _StubFAISS:
    """A vector index returning fixed similarities."""

    is_empty = False

    def __init__(self, results):
        self._results = results

    def query(self, query, top_k=None, metadata_filter=None, source_filter=None):
        return self._results[:top_k]


@pytest.mark.unit
class TestConfidenceIsNotTheRanking:
    """Test that rescaled and normalized rankings cannot fabricate confidence.

    Fusion and BM25 normalization both scale their winner to 1.0, which had
    every hybrid and lexical response claiming high confidence.
    """

    def _corpus(self):
        contents = [
            "The quarterly revenue grew by twelve percent year over year.",
            "Employee headcount remained flat across all regions.",
            "Cash flow from operations totalled two billion dollars.",
        ]
        return [
            ContextualizedChunk(
                chunk=Chunk(
                    chunk_id=f"c{i}",
                    content=content,
                    source="report.md",
                    segment_idx=0,
                    chunk_idx=i,
                ),
                context="",
            )
            for i, content in enumerate(contents)
        ]

    def _hybrid(self, similarity):
        from konte.stores.bm25_store import BM25Store
        from konte.stores.retriever import Retriever

        corpus = self._corpus()
        bm25 = BM25Store()
        bm25.build_index(corpus)
        faiss = _StubFAISS([(corpus[0], similarity), (corpus[1], similarity - 0.05)])
        return Retriever(faiss_store=faiss, bm25_store=bm25), corpus

    def test_weak_hybrid_evidence_does_not_deliver(self):
        """Test a poor semantic match no longer reads as a perfect one."""
        retriever, _ = self._hybrid(0.31)

        response = retriever.retrieve_hybrid(
            "gibberish xyzzy", top_k=3, use_keyword_extraction=False
        )

        assert response.results[0].score == 1.0  # the ranking is untouched
        assert response.top_score == pytest.approx(0.31)
        assert response.has_high_confidence is False
        assert response.suggested_action == "refine_query"

    def test_strong_hybrid_evidence_still_delivers(self):
        """Test confidence was not simply suppressed everywhere.

        The query shares no term with the corpus, leaving the semantic
        reading on its own.
        """
        retriever, _ = self._hybrid(0.88)

        response = retriever.retrieve_hybrid(
            "gibberish xyzzy", top_k=3, use_keyword_extraction=False
        )

        assert response.top_score == pytest.approx(0.88)
        assert response.suggested_action == "deliver"

    def test_hybrid_agrees_with_semantic_on_the_same_evidence(self):
        """Test fusing a second index cannot upgrade the same evidence to deliver."""
        retriever, _ = self._hybrid(0.55)

        hybrid = retriever.retrieve_hybrid("gibberish xyzzy", top_k=3, use_keyword_extraction=False)
        semantic = retriever.retrieve_semantic("gibberish xyzzy", top_k=3)

        assert hybrid.top_score == pytest.approx(semantic.top_score)
        assert hybrid.suggested_action == semantic.suggested_action

    def test_lexical_only_evidence_is_measured_by_coverage(self):
        """Test a chunk only BM25 found is still measured, not assumed perfect."""
        from konte.stores.bm25_store import BM25Store
        from konte.stores.retriever import Retriever

        corpus = self._corpus()
        bm25 = BM25Store()
        bm25.build_index(corpus)
        # The vector index knows nothing about the chunk BM25 ranks first.
        faiss = _StubFAISS([(corpus[2], 0.2)])
        retriever = Retriever(faiss_store=faiss, bm25_store=bm25)

        response = retriever.retrieve_hybrid(
            "quarterly revenue", top_k=3, use_keyword_extraction=False
        )

        assert response.top_score == pytest.approx(1.0)
        assert response.suggested_action == "deliver"

    def test_lexical_mode_separates_a_match_from_a_miss(self):
        """Test the lexical-only path stops answering 1.0 to everything."""
        from konte.stores.bm25_store import BM25Store
        from konte.stores.retriever import Retriever

        bm25 = BM25Store()
        bm25.build_index(self._corpus())
        retriever = Retriever(faiss_store=None, bm25_store=bm25)

        matched = retriever.retrieve_lexical(
            "quarterly revenue", top_k=3, use_keyword_extraction=False
        )
        absent = retriever.retrieve_lexical(
            "quarterly zirconium tungsten", top_k=3, use_keyword_extraction=False
        )

        assert matched.results[0].score == absent.results[0].score == 1.0
        assert matched.suggested_action == "deliver"
        assert absent.top_score < matched.top_score
        assert absent.suggested_action == "refine_query"

    async def test_unreachable_reranker_does_not_read_as_relevance(self):
        """Test a reranker outage falls back to the retrieval's own measurements."""
        from konte.stores.reranker import RerankOutcome
        from konte.stores.retriever import Retriever

        corpus = self._corpus()
        faiss = _StubFAISS([(corpus[0], 0.31), (corpus[1], 0.28)])
        retriever = Retriever(faiss_store=faiss, bm25_store=None)

        with patch(
            "konte.stores.retriever.rerank_chunks_with_score", new_callable=AsyncMock
        ) as rerank:
            # Every request failed, so these are the retrieval's own scores.
            rerank.return_value = RerankOutcome([(corpus[0], 1.0), (corpus[1], 0.9)], False)
            response = await retriever.retrieve_with_rerank("query", mode="semantic", top_k=2)

        assert response.top_score == pytest.approx(0.31)
        assert response.suggested_action == "refine_query"

    async def test_reranker_score_is_the_confidence_when_it_ran(self):
        """Test the cross-encoder's score replaces the retrieval's reading."""
        from konte.stores.reranker import RerankOutcome
        from konte.stores.retriever import Retriever

        corpus = self._corpus()
        faiss = _StubFAISS([(corpus[0], 0.31), (corpus[1], 0.28)])
        retriever = Retriever(faiss_store=faiss, bm25_store=None)

        with patch(
            "konte.stores.retriever.rerank_chunks_with_score", new_callable=AsyncMock
        ) as rerank:
            rerank.return_value = RerankOutcome([(corpus[1], 0.93), (corpus[0], 0.12)], True)
            response = await retriever.retrieve_with_rerank("query", mode="semantic", top_k=2)

        assert response.top_score == pytest.approx(0.93)
        assert response.suggested_action == "deliver"


@pytest.mark.unit
class TestFusionWeights:
    """Test that the configured weights decide which index wins a disagreement."""

    def _hybrid(self, semantic_weight, lexical_weight):
        from konte.stores.bm25_store import BM25Store
        from konte.stores.retriever import Retriever

        contents = [
            "Operating cash reserves increased modestly.",
            "Employee headcount remained flat across all regions.",
            "Cash flow from operations totalled two billion dollars.",
            "The quarterly revenue grew by twelve percent year over year.",
        ]
        corpus = [
            ContextualizedChunk(
                chunk=Chunk(
                    chunk_id=f"c{i}",
                    content=content,
                    source="report.md",
                    segment_idx=0,
                    chunk_idx=i,
                ),
                context="",
            )
            for i, content in enumerate(contents)
        ]
        bm25 = BM25Store()
        bm25.build_index(corpus)
        # Nothing BM25 ranks highly, so the two indexes disagree.
        faiss = _StubFAISS([(corpus[3], 0.9), (corpus[1], 0.85)])
        return Retriever(
            faiss_store=faiss,
            bm25_store=bm25,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

    def _top_chunk_id(self, retriever):
        response = retriever.retrieve_hybrid("cash flow", top_k=1, use_keyword_extraction=False)
        return response.results[0].chunk_id

    def test_semantic_weight_wins(self):
        """Test the vector index's top result leads when it is weighted heavier."""
        assert self._top_chunk_id(self._hybrid(0.9, 0.1)) == "c3"

    def test_lexical_weight_wins(self):
        """Test the lexical index's top result leads when it is weighted heavier."""
        assert self._top_chunk_id(self._hybrid(0.1, 0.9)) == "c2"


@pytest.fixture
def lexical_retriever(sample_chunks):
    """Build a retriever over a real, BM25-only index."""
    from konte.stores.bm25_store import BM25Store
    from konte.stores.retriever import Retriever

    store = BM25Store()
    store.build_index(sample_chunks)
    return Retriever(faiss_store=None, bm25_store=store)


@pytest.fixture
def extractor():
    """Patch both extraction entry points and yield the sync one."""
    with (
        patch("konte.stores.retriever.extract_search_keywords") as sync_extract,
        patch(
            "konte.stores.retriever.extract_search_keywords_async",
            new_callable=AsyncMock,
        ) as async_extract,
    ):
        sync_extract.return_value = ["chunk", "content"]
        async_extract.return_value = ["chunk", "content"]
        sync_extract.async_variant = async_extract
        yield sync_extract


def _keyword_extraction(enabled: bool):
    """Override the configured default for the duration of a test."""
    return patch("konte.stores.retriever.settings.BM25_KEYWORD_EXTRACTION", enabled)


@pytest.mark.unit
class TestKeywordExtractionControl:
    """Test who decides whether a retrieval pays for an LLM call."""

    def test_caller_can_turn_extraction_off(self, lexical_retriever, extractor):
        """Test the flag reaches BM25 through the mode-dispatching entry point."""
        lexical_retriever.retrieve("test query", use_keyword_extraction=False)

        extractor.assert_not_called()

    def test_caller_can_turn_extraction_on(self, lexical_retriever, extractor):
        """Test an explicit True is honoured."""
        lexical_retriever.retrieve("test query", use_keyword_extraction=True)

        extractor.assert_called_once_with("test query")

    def test_unset_follows_configuration(self, lexical_retriever, extractor):
        """Test a deployment can disable extraction without touching callers."""
        with _keyword_extraction(False):
            lexical_retriever.retrieve("test query")

        extractor.assert_not_called()

    def test_explicit_flag_overrides_configuration(self, lexical_retriever, extractor):
        """Test the caller still wins over a disabled default."""
        with _keyword_extraction(False):
            lexical_retriever.retrieve("test query", use_keyword_extraction=True)

        extractor.assert_called_once()

    def test_semantic_mode_never_extracts(self, lexical_retriever, extractor):
        """Test a mode that ignores BM25 does not pay for its preprocessing."""
        lexical_retriever.retrieve("test query", mode="semantic", use_keyword_extraction=True)

        extractor.assert_not_called()

    def test_no_lexical_index_never_extracts(self, extractor):
        """Test a semantic-only project skips a call whose result it discards."""
        from konte.stores.retriever import Retriever

        Retriever(faiss_store=None, bm25_store=None).retrieve(
            "test query", use_keyword_extraction=True
        )

        extractor.assert_not_called()

    def test_lexical_index_searches_the_keywords(self, lexical_retriever, extractor):
        """Test the extracted keywords are what BM25 actually receives."""
        with patch.object(
            lexical_retriever._bm25,
            "query_with_coverage",
            wraps=lexical_retriever._bm25.query_with_coverage,
        ) as bm25_query:
            lexical_retriever.retrieve(
                "which chunk has the content?", mode="lexical", use_keyword_extraction=True
            )

        assert bm25_query.call_args.args[0] == "chunk content"

    def test_empty_extraction_keeps_the_raw_query(self, lexical_retriever, extractor):
        """Test an empty extraction does not blank out the search.

        An empty BM25 query scores every chunk zero and silently returns
        whichever ones the corpus lists first.
        """
        extractor.return_value = []

        with patch.object(
            lexical_retriever._bm25,
            "query_with_coverage",
            wraps=lexical_retriever._bm25.query_with_coverage,
        ) as bm25_query:
            lexical_retriever.retrieve("what is it", mode="lexical", use_keyword_extraction=True)

        assert bm25_query.call_args.args[0] == "what is it"


@pytest.mark.unit
class TestAsyncRetrieval:
    """Test that the async entry points stay off the blocking extractor."""

    async def test_retrieve_async_awaits_extraction(self, lexical_retriever, extractor):
        """Test async retrieval never calls the sync, loop-blocking variant."""
        await lexical_retriever.retrieve_async("test query", use_keyword_extraction=True)

        extractor.async_variant.assert_awaited_once_with("test query")
        extractor.assert_not_called()

    async def test_retrieve_async_matches_retrieve(self, lexical_retriever, extractor):
        """Test both entry points rank identically for the same query."""
        sync = lexical_retriever.retrieve("test query", use_keyword_extraction=False)
        result = await lexical_retriever.retrieve_async("test query", use_keyword_extraction=False)

        assert [r.chunk_id for r in result.results] == [r.chunk_id for r in sync.results]

    async def test_rerank_hybrid_keeps_the_raw_query(self, lexical_retriever, extractor):
        """Test the documented carve-out survives the configured default."""
        with patch(
            "konte.stores.retriever.rerank_chunks_with_score", new_callable=AsyncMock
        ) as rerank:
            rerank.return_value = RerankOutcome([], False)
            await lexical_retriever.retrieve_with_rerank("test query", mode="hybrid")

        extractor.async_variant.assert_not_awaited()

    async def test_rerank_hybrid_honours_an_explicit_flag(self, lexical_retriever, extractor):
        """Test the carve-out is a default, not a veto on the caller."""
        with patch(
            "konte.stores.retriever.rerank_chunks_with_score", new_callable=AsyncMock
        ) as rerank:
            rerank.return_value = RerankOutcome([], False)
            await lexical_retriever.retrieve_with_rerank(
                "test query", mode="hybrid", use_keyword_extraction=True
            )

        extractor.async_variant.assert_awaited_once_with("test query")
