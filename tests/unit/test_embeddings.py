"""Unit tests for the process-wide query embedding cache."""

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from konte.index.embeddings import clear_query_embedding_cache, query_vector

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 4


class _CountingEmbeddings(Embeddings):
    """Deterministic vectors, and a tally of how often they were asked for."""

    def __init__(self, fail: bool = False):
        self.calls: list[str] = []
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query, recording the request."""
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("endpoint unavailable")
        vector = [0.0] * _DIMENSIONS
        for token in text.lower().split():
            vector[sum(ord(character) for character in token) % _DIMENSIONS] += 1.0
        return vector


@pytest.fixture
def embeddings():
    """A client that never leaves the process and counts what it was asked."""
    return _CountingEmbeddings()


@pytest.mark.unit
class TestQueryVectorCache:
    """Test that a query costs one embedding request however often it is asked."""

    def test_repeated_query_embeds_once(self, embeddings):
        """Test that the same query goes on the wire once, not once per search."""
        vectors = [query_vector(embeddings, _MODEL, "working capital") for _ in range(8)]

        assert embeddings.calls == ["working capital"]
        assert all(vector is vectors[0] for vector in vectors)

    def test_distinct_queries_each_embed(self, embeddings):
        """Test that the cache answers only for the query it holds."""
        first = query_vector(embeddings, _MODEL, "working capital")
        second = query_vector(embeddings, _MODEL, "HS 8471")

        assert embeddings.calls == ["working capital", "HS 8471"]
        assert not np.array_equal(first, second)

    def test_separate_clients_share_the_cache(self, embeddings):
        """Test that one question fanned out across projects embeds once."""
        others = [_CountingEmbeddings() for _ in range(2)]
        query_vector(embeddings, _MODEL, "memory chip classification")
        for other in others:
            query_vector(other, _MODEL, "memory chip classification")

        assert embeddings.calls == ["memory chip classification"]
        assert all(other.calls == [] for other in others)

    def test_model_is_part_of_the_key(self, embeddings):
        """Test that one model's vectors are not served for another's."""
        query_vector(embeddings, _MODEL, "working capital")
        query_vector(embeddings, "text-embedding-3-large", "working capital")

        assert embeddings.calls == ["working capital", "working capital"]

    def test_clearing_forgets_everything(self, embeddings):
        """Test that a cleared cache asks again rather than serving stale vectors."""
        query_vector(embeddings, _MODEL, "working capital")
        clear_query_embedding_cache()
        query_vector(embeddings, _MODEL, "working capital")

        assert embeddings.calls == ["working capital", "working capital"]

    def test_a_failed_request_is_not_remembered(self):
        """Test that a query whose request failed is asked again, not cached empty."""
        failing = _CountingEmbeddings(fail=True)
        with pytest.raises(RuntimeError):
            query_vector(failing, _MODEL, "working capital")

        working = _CountingEmbeddings()
        vector = query_vector(working, _MODEL, "working capital")

        assert working.calls == ["working capital"]
        assert vector.shape == (1, _DIMENSIONS)


@pytest.mark.unit
class TestQueryVectorShape:
    """Test that what the cache hands back is what FAISS searches with."""

    def test_vector_is_a_float32_row(self, embeddings):
        """Test that a hit needs no conversion before the search."""
        vector = query_vector(embeddings, _MODEL, "working capital")

        assert vector.shape == (1, _DIMENSIONS)
        assert vector.dtype == np.float32

    def test_vector_cannot_be_written_through(self, embeddings):
        """Test that the shared entry is protected from a caller mutating it."""
        vector = query_vector(embeddings, _MODEL, "working capital")

        with pytest.raises(ValueError):
            vector[0][0] = 99.0
