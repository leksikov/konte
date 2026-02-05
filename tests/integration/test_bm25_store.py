"""Integration tests for BM25 store list-value metadata filtering."""

import pytest

from konte.models import Chunk, ContextualizedChunk


@pytest.fixture
def metadata_chunks():
    """Create chunks with custom metadata for list-value filter testing."""
    data = [
        ("ADOBE_2022_10K.md", "Adobe revenue grew 15% in FY2022.", {"company": "ADOBE", "year": "2022"}),
        ("ADOBE_2023_10K.md", "Adobe cloud growth analysis for 2023.", {"company": "ADOBE", "year": "2023"}),
        ("3M_2022_10K.md", "3M industrial segment results in 2022.", {"company": "3M", "year": "2022"}),
        ("3M_2023_10K.md", "3M performance and segment breakdown.", {"company": "3M", "year": "2023"}),
        ("JOHNSON_2022_10K.md", "Johnson pharmaceutical division report.", {"company": "JOHNSON", "year": "2022"}),
    ]
    chunks = []
    for i, (source, content, meta) in enumerate(data):
        chunk = Chunk(
            chunk_id=f"{source}_s0_c{i}",
            content=content,
            source=source,
            segment_idx=0,
            chunk_idx=i,
            metadata=meta,
        )
        chunks.append(ContextualizedChunk(chunk=chunk, context=""))
    return chunks


@pytest.mark.integration
class TestBM25StoreListValueFilter:
    """Integration tests for BM25 store list-value metadata_filter."""

    def test_metadata_filter_list_values(self, metadata_chunks):
        """Test filtering with list returns chunks matching any value."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(metadata_chunks)

        results = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"company": ["ADOBE", "3M"]},
        )

        assert len(results) == 4
        companies = {r[0].chunk.metadata["company"] for r in results}
        assert companies == {"ADOBE", "3M"}

    def test_metadata_filter_list_single_item(self, metadata_chunks):
        """Test list with single item behaves same as scalar."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(metadata_chunks)

        results_list = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"year": ["2022"]},
        )
        results_scalar = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"year": "2022"},
        )

        ids_list = sorted(r[0].chunk.chunk_id for r in results_list)
        ids_scalar = sorted(r[0].chunk.chunk_id for r in results_scalar)
        assert ids_list == ids_scalar

    def test_metadata_filter_mixed_list_and_scalar(self, metadata_chunks):
        """Test combining list and scalar filters (AND logic)."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(metadata_chunks)

        results = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"company": ["ADOBE", "3M"], "year": "2022"},
        )

        assert len(results) == 2
        for chunk, _ in results:
            assert chunk.chunk.metadata["company"] in ["ADOBE", "3M"]
            assert chunk.chunk.metadata["year"] == "2022"

    def test_metadata_filter_list_no_match(self, metadata_chunks):
        """Test list filter with no matching values returns empty."""
        from konte.stores import BM25Store

        store = BM25Store()
        store.build_index(metadata_chunks)

        results = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"company": ["NONEXISTENT", "UNKNOWN"]},
        )

        assert results == []
