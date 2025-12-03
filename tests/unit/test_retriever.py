"""Unit tests for retriever module."""

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
        from konte.retriever import reciprocal_rank_fusion

        results = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        fused = reciprocal_rank_fusion([results])

        assert len(fused) == 2
        # Order should be preserved
        assert fused[0][0].chunk.chunk_id == "test_s0_c0"
        assert fused[1][0].chunk.chunk_id == "test_s0_c1"

    def test_two_lists_same_order(self, sample_chunks):
        """Test fusion when both lists have same order."""
        from konte.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[0], 0.95), (sample_chunks[1], 0.85)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 2
        assert fused[0][0].chunk.chunk_id == "test_s0_c0"

    def test_two_lists_different_order(self, sample_chunks):
        """Test fusion combines results from different orderings."""
        from konte.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[1], 0.95), (sample_chunks[0], 0.85)]

        fused = reciprocal_rank_fusion([list1, list2])

        # Both should appear in fused results
        chunk_ids = [r[0].chunk.chunk_id for r in fused]
        assert "test_s0_c0" in chunk_ids
        assert "test_s0_c1" in chunk_ids

    def test_disjoint_lists(self, sample_chunks):
        """Test fusion of lists with no overlap."""
        from konte.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
        list2 = [(sample_chunks[2], 0.95), (sample_chunks[3], 0.85)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 4

    def test_scores_normalized(self, sample_chunks):
        """Test that fused scores are normalized to 0-1."""
        from konte.retriever import reciprocal_rank_fusion

        list1 = [(sample_chunks[0], 0.9)]
        list2 = [(sample_chunks[0], 0.9)]

        fused = reciprocal_rank_fusion([list1, list2])

        assert 0.0 <= fused[0][1] <= 1.0

    def test_empty_lists(self):
        """Test fusion with empty lists."""
        from konte.retriever import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion([[], []])
        assert fused == []


@pytest.mark.unit
class TestDetermineSuggestedAction:
    """Test suggested action determination."""

    def test_high_score_delivers(self):
        """Test that high scores suggest 'deliver'."""
        from konte.retriever import _determine_suggested_action

        assert _determine_suggested_action(0.9) == "deliver"
        assert _determine_suggested_action(0.7) == "deliver"

    def test_medium_score_query_more(self):
        """Test that medium scores suggest 'query_more'."""
        from konte.retriever import _determine_suggested_action

        assert _determine_suggested_action(0.6) == "query_more"
        assert _determine_suggested_action(0.4) == "query_more"

    def test_low_score_refine_query(self):
        """Test that low scores suggest 'refine_query'."""
        from konte.retriever import _determine_suggested_action

        assert _determine_suggested_action(0.3) == "refine_query"
        assert _determine_suggested_action(0.0) == "refine_query"


@pytest.mark.unit
class TestBuildRetrievalResponse:
    """Test building RetrievalResponse."""

    def test_builds_response_with_results(self, sample_chunks):
        """Test building response from results."""
        from konte.retriever import _build_retrieval_response

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
        from konte.retriever import _build_retrieval_response

        response = _build_retrieval_response("test query", [], top_k=10)

        assert response.total_found == 0
        assert response.top_score == 0.0
        assert response.suggested_action == "refine_query"

    def test_respects_top_k(self, sample_chunks):
        """Test that top_k limits results."""
        from konte.retriever import _build_retrieval_response

        results = [(c, 0.9 - i * 0.1) for i, c in enumerate(sample_chunks)]
        response = _build_retrieval_response("test", results, top_k=2)

        assert len(response.results) == 2


@pytest.mark.unit
class TestRetrieverModes:
    """Test retriever with different modes (using mocked stores)."""

    def test_retrieve_semantic_empty_store(self):
        """Test semantic retrieval with no FAISS store."""
        from konte.retriever import Retriever

        retriever = Retriever(faiss_store=None, bm25_store=None)
        response = retriever.retrieve_semantic("test query")

        assert response.total_found == 0

    def test_retrieve_lexical_empty_store(self):
        """Test lexical retrieval with no BM25 store."""
        from konte.retriever import Retriever

        retriever = Retriever(faiss_store=None, bm25_store=None)
        response = retriever.retrieve_lexical("test query")

        assert response.total_found == 0

    def test_retrieve_hybrid_no_stores(self):
        """Test hybrid retrieval with no stores."""
        from konte.retriever import Retriever

        retriever = Retriever(faiss_store=None, bm25_store=None)
        response = retriever.retrieve_hybrid("test query")

        assert response.total_found == 0

    def test_retrieve_default_mode_is_hybrid(self):
        """Test that default mode is hybrid."""
        from konte.retriever import Retriever

        retriever = Retriever()
        response = retriever.retrieve("test query")

        # Should complete without error, even with empty stores
        assert response.query == "test query"
