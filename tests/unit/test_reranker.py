"""Unit tests for reranker module."""

from unittest.mock import patch

import pytest

from konte.config.settings import settings
from konte.models import Chunk, ContextualizedChunk
from konte.stores.reranker import _resolve_score_endpoint, rerank_chunks_with_score


@pytest.fixture
def sample_chunks():
    """Create sample (chunk, score) tuples for testing."""
    chunks = []
    for i in range(3):
        chunk = Chunk(
            chunk_id=f"test_s0_c{i}",
            content=f"Content for chunk {i}",
            source="test.txt",
            segment_idx=0,
            chunk_idx=i,
        )
        chunks.append((ContextualizedChunk(chunk=chunk, context=f"Context {i}"), 0.9 - i * 0.1))
    return chunks


@pytest.mark.unit
class TestRerankerConfiguration:
    """Test reranker endpoint configuration."""

    def test_resolve_endpoint_unconfigured_raises(self, monkeypatch):
        """Missing RERANKER_BASE_URL raises a clear error."""
        monkeypatch.setattr(settings, "RERANKER_BASE_URL", None)

        with pytest.raises(ValueError, match="RERANKER_BASE_URL"):
            _resolve_score_endpoint()

    def test_resolve_endpoint_builds_score_url(self, monkeypatch):
        """Configured base URL resolves to its /score endpoint."""
        monkeypatch.setattr(settings, "RERANKER_BASE_URL", "https://example.com/v1")
        assert _resolve_score_endpoint() == "https://example.com/v1/score"

        monkeypatch.setattr(settings, "RERANKER_BASE_URL", "https://example.com/v1/")
        assert _resolve_score_endpoint() == "https://example.com/v1/score"

    async def test_rerank_unconfigured_raises(self, sample_chunks, monkeypatch):
        """Reranking without RERANKER_BASE_URL raises instead of silently failing."""
        monkeypatch.setattr(settings, "RERANKER_BASE_URL", None)

        with pytest.raises(ValueError, match="RERANKER_BASE_URL"):
            await rerank_chunks_with_score("query", sample_chunks)

    async def test_rerank_empty_chunks_returns_empty(self, monkeypatch):
        """Empty input returns empty output without touching the endpoint."""
        monkeypatch.setattr(settings, "RERANKER_BASE_URL", None)

        assert await rerank_chunks_with_score("query", []) == []


@pytest.mark.unit
class TestRerankerFailureModes:
    """Test failure handling: never fabricate all-zero rankings."""

    async def test_all_requests_failed_falls_back_to_original(self, sample_chunks, monkeypatch):
        """If every score request fails, original order and scores are kept."""
        monkeypatch.setattr(settings, "RERANKER_BASE_URL", "https://example.com/v1")

        async def fail_all(client, query, chunk, idx, model, semaphore, endpoint, max_chars=0):
            return (idx, None)

        with patch("konte.stores.reranker._score_single_chunk", side_effect=fail_all):
            result = await rerank_chunks_with_score("query", sample_chunks)

        assert result == sample_chunks  # original order, original retrieval scores
        assert all(score > 0.0 for _, score in result)

    async def test_partial_failure_scores_failed_chunk_zero(self, sample_chunks, monkeypatch):
        """A single failed request scores 0.0; the rest rerank normally."""
        monkeypatch.setattr(settings, "RERANKER_BASE_URL", "https://example.com/v1")

        async def fail_first(client, query, chunk, idx, model, semaphore, endpoint, max_chars=0):
            if idx == 0:
                return (idx, None)
            return (idx, 0.5 + idx * 0.1)

        with patch("konte.stores.reranker._score_single_chunk", side_effect=fail_first):
            result = await rerank_chunks_with_score("query", sample_chunks)

        assert len(result) == 3
        # Failed chunk sorts last with score 0.0; others keep their scores
        assert result[-1][0].chunk.chunk_id == "test_s0_c0"
        assert result[-1][1] == 0.0
        assert result[0][1] == pytest.approx(0.7)
