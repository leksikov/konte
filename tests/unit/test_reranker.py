"""Unit tests for reranker module."""

import pytest

from konte.config.settings import settings
from konte.models import Chunk, ContextualizedChunk


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
        from konte.stores.reranker import _resolve_score_endpoint

        monkeypatch.setattr(settings, "RERANKER_BASE_URL", None)

        with pytest.raises(ValueError, match="RERANKER_BASE_URL"):
            _resolve_score_endpoint()

    def test_resolve_endpoint_builds_score_url(self, monkeypatch):
        """Configured base URL resolves to its /score endpoint."""
        from konte.stores.reranker import _resolve_score_endpoint

        monkeypatch.setattr(settings, "RERANKER_BASE_URL", "https://example.com/v1")
        assert _resolve_score_endpoint() == "https://example.com/v1/score"

        monkeypatch.setattr(settings, "RERANKER_BASE_URL", "https://example.com/v1/")
        assert _resolve_score_endpoint() == "https://example.com/v1/score"

    async def test_rerank_unconfigured_raises(self, sample_chunks, monkeypatch):
        """Reranking without RERANKER_BASE_URL raises instead of silently failing."""
        from konte.stores.reranker import rerank_chunks_with_score

        monkeypatch.setattr(settings, "RERANKER_BASE_URL", None)

        with pytest.raises(ValueError, match="RERANKER_BASE_URL"):
            await rerank_chunks_with_score("query", sample_chunks)

    async def test_rerank_empty_chunks_returns_empty(self, monkeypatch):
        """Empty input returns empty output without touching the endpoint."""
        from konte.stores.reranker import rerank_chunks_with_score

        monkeypatch.setattr(settings, "RERANKER_BASE_URL", None)

        assert await rerank_chunks_with_score("query", []) == []
