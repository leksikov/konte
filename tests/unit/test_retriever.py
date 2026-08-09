"""Unit tests for retriever module."""

from unittest.mock import AsyncMock, patch

import pytest

from konte.models import Chunk, ContextualizedChunk


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
        response = _build_retrieval_response("test query", results, top_k=3)

        assert response.query == "test query"
        assert response.total_found == 3
        assert len(response.results) == 3
        assert response.top_score == 0.9
        assert response.score_spread == 0.4  # 0.9 - 0.5
        assert response.has_high_confidence is True
        assert response.suggested_action == "deliver"

    def test_builds_response_empty_results(self):
        """Test building response with no results."""
        from konte.stores.retriever import _build_retrieval_response

        response = _build_retrieval_response("test query", [], top_k=10)

        assert response.total_found == 0
        assert response.top_score == 0.0
        assert response.suggested_action == "refine_query"

    def test_respects_top_k(self, sample_chunks):
        """Test that top_k limits results."""
        from konte.stores.retriever import _build_retrieval_response

        results = [(c, 0.9 - i * 0.1) for i, c in enumerate(sample_chunks)]
        response = _build_retrieval_response("test", results, top_k=2)

        assert len(response.results) == 2


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
            lexical_retriever._bm25, "query", wraps=lexical_retriever._bm25.query
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
            lexical_retriever._bm25, "query", wraps=lexical_retriever._bm25.query
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
            rerank.return_value = []
            await lexical_retriever.retrieve_with_rerank("test query", mode="hybrid")

        extractor.async_variant.assert_not_awaited()

    async def test_rerank_hybrid_honours_an_explicit_flag(self, lexical_retriever, extractor):
        """Test the carve-out is a default, not a veto on the caller."""
        with patch(
            "konte.stores.retriever.rerank_chunks_with_score", new_callable=AsyncMock
        ) as rerank:
            rerank.return_value = []
            await lexical_retriever.retrieve_with_rerank(
                "test query", mode="hybrid", use_keyword_extraction=True
            )

        extractor.async_variant.assert_awaited_once_with("test query")
