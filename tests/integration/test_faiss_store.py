"""Integration tests for FAISS store (requires OPENAI_API_KEY)."""

import os

import pytest

from konte.models import Chunk, ContextualizedChunk

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture
def sample_chunks():
    """Create sample contextualized chunks for testing."""
    chunks = []
    contents = [
        ("Tariff code 8542.31 applies to electronic integrated circuits.",
         "This section covers semiconductor devices under Chapter 85."),
        ("Import duty rate is 5% for MFN countries.",
         "This discusses duty rates for electronics from WTO member nations."),
        ("Anti-dumping duties may apply to certain imports.",
         "This explains special duties imposed on unfairly priced goods."),
        ("The Harmonized System uses 6-digit codes internationally.",
         "This describes the international tariff classification standard."),
        ("Customs valuation follows WTO Agreement rules.",
         "This covers how import values are determined for duty calculation."),
    ]
    for i, (content, context) in enumerate(contents):
        chunk = Chunk(
            chunk_id=f"test_s0_c{i}",
            content=content,
            source="test.txt",
            segment_idx=0,
            chunk_idx=i,
        )
        chunks.append(ContextualizedChunk(chunk=chunk, context=context))
    return chunks


@pytest.mark.integration
class TestFAISSStoreBuild:
    """Test FAISS store build functionality."""

    def test_build_index_creates_index(self, sample_chunks):
        """Test that build_index creates a searchable index."""
        from konte.faiss_store import FAISSStore

        store = FAISSStore()
        store.build_index(sample_chunks)

        assert not store.is_empty

    def test_build_empty_chunks(self):
        """Test building index with empty chunks list."""
        from konte.faiss_store import FAISSStore

        store = FAISSStore()
        store.build_index([])

        assert store.is_empty


@pytest.mark.integration
class TestFAISSStoreQuery:
    """Test FAISS store query functionality."""

    def test_query_returns_results(self, sample_chunks):
        """Test that query returns relevant results."""
        from konte.faiss_store import FAISSStore

        store = FAISSStore()
        store.build_index(sample_chunks)

        results = store.query("What is the tariff code for electronics?", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r[0], ContextualizedChunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_query_scores_in_range(self, sample_chunks):
        """Test that query scores are between 0 and 1."""
        from konte.faiss_store import FAISSStore

        store = FAISSStore()
        store.build_index(sample_chunks)

        results = store.query("tariff classification", top_k=5)

        for chunk, score in results:
            assert 0.0 <= score <= 1.0

    def test_query_returns_relevant_first(self, sample_chunks):
        """Test that most relevant result comes first."""
        from konte.faiss_store import FAISSStore

        store = FAISSStore()
        store.build_index(sample_chunks)

        results = store.query("electronic integrated circuits tariff code", top_k=3)

        # First result should be about tariff codes/electronics
        top_content = results[0][0].chunk.content.lower()
        assert "8542" in top_content or "electronic" in top_content

    def test_query_empty_index(self):
        """Test querying empty index returns empty list."""
        from konte.faiss_store import FAISSStore

        store = FAISSStore()
        results = store.query("test query")

        assert results == []


@pytest.mark.integration
class TestFAISSStorePersistence:
    """Test FAISS store save/load functionality."""

    def test_save_and_load(self, sample_chunks, tmp_path):
        """Test that saved index can be loaded and queried."""
        from konte.faiss_store import FAISSStore

        # Build and save
        store1 = FAISSStore()
        store1.build_index(sample_chunks)
        store1.save(tmp_path)

        # Load into new store
        store2 = FAISSStore()
        store2.load(tmp_path)

        assert not store2.is_empty

        # Query should work
        results = store2.query("import duty rate", top_k=3)
        assert len(results) == 3

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading from nonexistent path raises error."""
        from konte.faiss_store import FAISSStore

        store = FAISSStore()
        with pytest.raises(FileNotFoundError):
            store.load(tmp_path / "nonexistent")

    def test_saved_chunks_preserved(self, sample_chunks, tmp_path):
        """Test that chunk data is preserved after save/load."""
        from konte.faiss_store import FAISSStore

        store1 = FAISSStore()
        store1.build_index(sample_chunks)
        store1.save(tmp_path)

        store2 = FAISSStore()
        store2.load(tmp_path)

        results = store2.query("test", top_k=1)
        chunk = results[0][0]

        # Verify chunk structure preserved
        assert chunk.chunk.chunk_id is not None
        assert chunk.chunk.source == "test.txt"
        assert chunk.context != ""
