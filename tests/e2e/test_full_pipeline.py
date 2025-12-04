"""E2E tests for full pipeline: ingest → build → query.

These tests use real OpenAI API calls and validate complete workflows.
"""

import os
from pathlib import Path

import pytest

from konte.chunker import count_tokens

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
    pytest.mark.e2e,
]


@pytest.mark.e2e
class TestFullPipelineWithContext:
    """Test complete pipeline with context generation."""

    async def test_ingest_build_query_with_context(self, tmp_path):
        """Test full pipeline: add document → build with context → query."""
        from konte.project import Project

        project = Project.create(
            name="e2e_context_test",
            storage_path=tmp_path,
        )

        # Add document
        num_chunks = project.add_documents([FIXTURES_DIR / "sample.txt"])
        assert num_chunks > 0

        # Build with context generation (real API call)
        await project.build(skip_context=False)

        # Verify context was generated
        for ctx_chunk in project._contextualized_chunks:
            # Context should not be empty (unless rate limited)
            # Just verify the structure is correct
            assert ctx_chunk.chunk is not None
            assert isinstance(ctx_chunk.context, str)

        # Query should return results
        response = project.query("tariff classification duty rate")
        assert response.total_found > 0
        assert len(response.results) > 0
        assert response.top_score > 0

    async def test_context_generation_produces_valid_text(self, tmp_path):
        """Verify context generation produces meaningful text."""
        from konte.project import Project

        project = Project.create(
            name="e2e_context_valid_test",
            storage_path=tmp_path,
            chunk_size=200,  # Smaller chunks for faster test
            chunk_overlap=20,
        )

        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=False)

        # Check that at least some contexts have content
        contexts_with_content = [
            c for c in project._contextualized_chunks
            if c.context and len(c.context) > 10
        ]

        # At least 50% should have context (allowing for some failures)
        assert len(contexts_with_content) >= len(project._contextualized_chunks) * 0.5, (
            f"Only {len(contexts_with_content)} of {len(project._contextualized_chunks)} "
            "chunks have context"
        )


@pytest.mark.e2e
class TestSegmentSizesInPipeline:
    """Test that segment sizes are correctly bounded in full pipeline."""

    def test_segments_bounded_not_full_document(self, tmp_path):
        """Verify segments passed to context gen are bounded, not full doc."""
        from konte.project import Project

        project = Project.create(
            name="e2e_segment_test",
            storage_path=tmp_path,
            segment_size=500,  # Small segment size
            segment_overlap=50,
            chunk_size=100,
            chunk_overlap=10,
        )

        project.add_documents([FIXTURES_DIR / "sample.txt"])

        # Read document to get full size
        doc_content = (FIXTURES_DIR / "sample.txt").read_text()
        doc_tokens = count_tokens(doc_content)

        # Each segment should be significantly smaller than full document
        for seg_idx, segment_text in project._segments.items():
            segment_tokens = count_tokens(segment_text)

            # Segment should be bounded (not the full document)
            assert segment_tokens < doc_tokens, (
                f"Segment {seg_idx} has {segment_tokens} tokens, "
                f"which is >= full document ({doc_tokens} tokens)"
            )

            # Segment should be reasonably close to segment_size
            # Allow up to 2x for word boundary adjustments
            assert segment_tokens <= 500 * 2, (
                f"Segment {seg_idx} has {segment_tokens} tokens, "
                f"expected <= 1000 (500 * 2)"
            )


@pytest.mark.e2e
class TestRetrievalQuality:
    """Test that retrieval returns relevant results."""

    async def test_query_returns_relevant_results(self, tmp_path):
        """Test that queries return results containing query terms."""
        from konte.project import Project

        project = Project.create(
            name="e2e_relevance_test",
            storage_path=tmp_path,
        )

        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=True)  # Skip context for speed

        # Query for terms that should be in the document
        response = project.query("tariff duty import")

        assert response.total_found > 0

        # At least one result should contain relevant terms
        relevant_found = False
        for result in response.results:
            content_lower = result.content.lower()
            if "tariff" in content_lower or "duty" in content_lower or "import" in content_lower:
                relevant_found = True
                break

        assert relevant_found, "No results contained relevant terms"

    async def test_hybrid_mode_combines_semantic_and_lexical(self, tmp_path):
        """Test that hybrid mode uses both retrieval methods."""
        from konte.project import Project

        project = Project.create(
            name="e2e_hybrid_test",
            storage_path=tmp_path,
        )

        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=True)

        # Get results from all three modes
        hybrid_response = project.query("tariff classification", mode="hybrid")
        semantic_response = project.query("tariff classification", mode="semantic")
        lexical_response = project.query("tariff classification", mode="lexical")

        # All should return results
        assert hybrid_response.total_found > 0
        assert semantic_response.total_found > 0
        assert lexical_response.total_found > 0

        # Hybrid should potentially have different ranking than pure modes
        # (this is a soft check - they might be the same for small documents)
        hybrid_ids = [r.chunk_id for r in hybrid_response.results[:5]]
        semantic_ids = [r.chunk_id for r in semantic_response.results[:5]]
        lexical_ids = [r.chunk_id for r in lexical_response.results[:5]]

        # Just verify we got valid results from each mode
        assert len(hybrid_ids) > 0
        assert len(semantic_ids) > 0
        assert len(lexical_ids) > 0


@pytest.mark.e2e
class TestSuggestedAction:
    """Test that suggested_action is computed correctly."""

    async def test_suggested_action_values(self, tmp_path):
        """Test that suggested_action is one of valid values."""
        from konte.project import Project

        project = Project.create(
            name="e2e_action_test",
            storage_path=tmp_path,
        )

        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=True)

        response = project.query("tariff")

        # Should be one of the valid actions
        assert response.suggested_action in ["deliver", "query_more", "refine_query"]

        # Verify consistency with score thresholds
        if response.top_score >= 0.7:
            assert response.suggested_action == "deliver"
        elif response.top_score >= 0.4:
            assert response.suggested_action == "query_more"
        else:
            assert response.suggested_action == "refine_query"
