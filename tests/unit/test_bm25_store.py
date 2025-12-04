"""Unit tests for BM25 store (no external API required)."""

import pytest

from konte.models import Chunk, ContextualizedChunk


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


@pytest.mark.unit
class TestBM25StoreBuild:
    """Test BM25 store build functionality."""

    def test_build_index_creates_index(self, sample_chunks):
        """Test that build_index creates a searchable index."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        assert not store.is_empty

    def test_build_empty_chunks(self):
        """Test building index with empty chunks list."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index([])

        assert store.is_empty


@pytest.mark.unit
class TestBM25StoreQuery:
    """Test BM25 store query functionality."""

    def test_query_returns_results(self, sample_chunks):
        """Test that query returns relevant results."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.query("tariff code electronics", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r[0], ContextualizedChunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_query_scores_in_range(self, sample_chunks):
        """Test that query scores are between 0 and 1."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.query("tariff classification", top_k=5)

        for chunk, score in results:
            assert 0.0 <= score <= 1.0

    def test_query_exact_match_scores_high(self, sample_chunks):
        """Test that exact keyword matches score higher."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.query("8542.31", top_k=3)

        # The chunk with exact tariff code should be first
        top_content = results[0][0].chunk.content.lower()
        assert "8542.31" in top_content

    def test_query_empty_index(self):
        """Test querying empty index returns empty list."""
        from konte.stores import BM25Store

        store = BM25Store()
        results = store.query("test query")

        assert results == []

    def test_query_no_matches(self, sample_chunks):
        """Test query with no matching terms still returns results (ranked by BM25)."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        # Query with terms not in corpus
        results = store.query("xyz123 completely unrelated", top_k=3)

        # BM25 will return results with 0 scores
        assert len(results) == 3


@pytest.mark.unit
class TestBM25StorePersistence:
    """Test BM25 store save/load functionality."""

    def test_save_and_load(self, sample_chunks, tmp_path):
        """Test that saved index can be loaded and queried."""
        from konte.stores import BM25Store

        # Build and save
        store1 = BM25Store()
        store1.build_index(sample_chunks)
        store1.save(tmp_path)

        # Load into new store
        store2 = BM25Store()
        store2.load(tmp_path)

        assert not store2.is_empty

        # Query should work
        results = store2.query("import duty rate", top_k=3)
        assert len(results) == 3

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading from nonexistent path raises error."""
        from konte.stores import BM25Store

        store = BM25Store()
        with pytest.raises(FileNotFoundError):
            store.load(tmp_path / "nonexistent")

    def test_saved_chunks_preserved(self, sample_chunks, tmp_path):
        """Test that chunk data is preserved after save/load."""
        from konte.stores import BM25Store

        store1 = BM25Store()
        store1.build_index(sample_chunks)
        store1.save(tmp_path)

        store2 = BM25Store()
        store2.load(tmp_path)

        results = store2.query("test", top_k=1)
        chunk = results[0][0]

        # Verify chunk structure preserved
        assert chunk.chunk.chunk_id is not None
        assert chunk.chunk.source == "test.txt"

    def test_query_results_same_after_reload(self, sample_chunks, tmp_path):
        """Test that query results are consistent before and after reload."""
        from konte.stores import BM25Store

        store1 = BM25Store()
        store1.build_index(sample_chunks)

        # Query before save
        results1 = store1.query("electronic integrated circuits", top_k=3)

        store1.save(tmp_path)

        store2 = BM25Store()
        store2.load(tmp_path)

        # Query after load
        results2 = store2.query("electronic integrated circuits", top_k=3)

        # Same chunks should be returned
        ids1 = [r[0].chunk.chunk_id for r in results1]
        ids2 = [r[0].chunk.chunk_id for r in results2]
        assert ids1 == ids2
