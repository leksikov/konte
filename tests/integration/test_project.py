"""Integration tests for Project class (requires OPENAI_API_KEY)."""

import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture
def sample_files():
    """Get list of sample fixture files."""
    return [
        FIXTURES_DIR / "sample.txt",
        FIXTURES_DIR / "sample.md",
    ]


@pytest.mark.integration
class TestProjectCreate:
    """Test project creation."""

    def test_create_project(self, tmp_path):
        """Test creating a new project."""
        from konte.project import Project

        project = Project.create(
            name="test_project",
            storage_path=tmp_path,
        )

        assert project.config.name == "test_project"
        assert project.config.storage_path == tmp_path

    def test_create_with_custom_config(self, tmp_path):
        """Test creating project with custom configuration."""
        from konte.project import Project

        project = Project.create(
            name="custom_project",
            storage_path=tmp_path,
            chunk_size=400,
            enable_bm25=False,
        )

        assert project.config.chunk_size == 400
        assert project.config.enable_bm25 is False


@pytest.mark.integration
class TestProjectAddDocuments:
    """Test adding documents to project."""

    def test_add_documents(self, tmp_path, sample_files):
        """Test adding documents creates chunks."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        num_chunks = project.add_documents(sample_files)

        assert num_chunks > 0

    def test_add_single_document(self, tmp_path):
        """Test adding a single document."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        num_chunks = project.add_documents([FIXTURES_DIR / "sample.txt"])

        assert num_chunks > 0


@pytest.mark.integration
class TestProjectBuild:
    """Test project build with real API calls."""

    async def test_build_with_context(self, tmp_path, sample_files):
        """Test building project with context generation."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        await project.build(skip_context=False)

        # Should have retriever ready
        response = project.query("tariff classification")
        assert response.total_found > 0

    async def test_build_skip_context(self, tmp_path):
        """Test building project without context generation."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        await project.build(skip_context=True)

        response = project.query("tariff")
        assert response.total_found > 0

    async def test_build_faiss_only(self, tmp_path):
        """Test building with FAISS only."""
        from konte.project import Project

        project = Project.create(
            name="test",
            storage_path=tmp_path,
            enable_bm25=False,
        )
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        await project.build(skip_context=True)

        response = project.query("tariff", mode="semantic")
        assert response.total_found > 0

    async def test_build_bm25_only(self, tmp_path):
        """Test building with BM25 only."""
        from konte.project import Project

        project = Project.create(
            name="test",
            storage_path=tmp_path,
            enable_faiss=False,
        )
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        await project.build(skip_context=True)

        response = project.query("tariff", mode="lexical")
        assert response.total_found > 0


@pytest.mark.integration
class TestProjectQuery:
    """Test project querying."""

    async def test_query_hybrid(self, tmp_path):
        """Test hybrid query mode."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=True)

        response = project.query("Harmonized System classification", mode="hybrid")

        assert response.total_found > 0
        assert len(response.results) > 0

    async def test_query_returns_relevant_results(self, tmp_path):
        """Test that queries return relevant results."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=True)

        response = project.query("import duty rate percentage")

        # Should find results about duty rates
        has_relevant = any(
            "duty" in r.content.lower() or "rate" in r.content.lower()
            for r in response.results
        )
        assert has_relevant


@pytest.mark.integration
class TestProjectPersistence:
    """Test project save/load."""

    async def test_save_and_load(self, tmp_path):
        """Test saving and loading project."""
        from konte.project import Project

        # Create and build
        project1 = Project.create(name="persist_test", storage_path=tmp_path)
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=True)
        project1.save()

        # Load into new instance
        project2 = Project.open(name="persist_test", storage_path=tmp_path)

        # Query should work
        response = project2.query("tariff")
        assert response.total_found > 0

    async def test_load_preserves_config(self, tmp_path):
        """Test that loading preserves configuration."""
        from konte.project import Project

        project1 = Project.create(
            name="config_test",
            storage_path=tmp_path,
            chunk_size=400,
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=True)
        project1.save()

        project2 = Project.open(name="config_test", storage_path=tmp_path)

        assert project2.config.chunk_size == 400

    def test_open_nonexistent_raises(self, tmp_path):
        """Test that opening nonexistent project raises error."""
        from konte.project import Project

        with pytest.raises(FileNotFoundError):
            Project.open(name="nonexistent", storage_path=tmp_path)


@pytest.mark.integration
class TestProjectAsRetriever:
    """Test as_retriever for Agno integration."""

    async def test_as_retriever_returns_callable(self, tmp_path):
        """Test that as_retriever returns a callable."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=True)

        retriever = project.as_retriever()

        assert callable(retriever)

    async def test_retriever_callable_works(self, tmp_path):
        """Test that the retriever callable returns results."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=True)

        retriever = project.as_retriever()
        response = retriever("tariff classification")

        assert response.total_found > 0


@pytest.mark.integration
class TestProjectSegmentStorage:
    """Test that segments are stored correctly (not full document)."""

    def test_segments_stored_not_full_document(self, tmp_path):
        """Verify segments map contains segment text, not full document."""
        from konte.project import Project
        from konte.chunker import count_tokens

        project = Project.create(
            name="segment_test",
            storage_path=tmp_path,
            segment_size=200,  # Small segment size for testing
            segment_overlap=20,
        )
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        # Check that segments are properly sized (not full document)
        # Read the document to know full size
        doc_content = (FIXTURES_DIR / "sample.txt").read_text()
        doc_tokens = count_tokens(doc_content)

        for seg_idx, segment_text in project._segments.items():
            segment_tokens = count_tokens(segment_text)
            # Segment should be around segment_size, not the full document
            # Allow 2x margin for word boundary adjustments
            assert segment_tokens <= 200 * 2, (
                f"Segment {seg_idx} has {segment_tokens} tokens, "
                f"expected <= 400 (200 * 2)"
            )
            # Segment should be smaller than full document
            assert segment_tokens < doc_tokens, (
                f"Segment {seg_idx} has {segment_tokens} tokens, "
                f"which is >= full document ({doc_tokens} tokens)"
            )

    def test_each_chunk_maps_to_valid_segment(self, tmp_path):
        """Verify each chunk's segment_idx maps to correct segment."""
        from konte.project import Project

        project = Project.create(
            name="chunk_map_test",
            storage_path=tmp_path,
            segment_size=200,
            segment_overlap=20,
            chunk_size=50,
            chunk_overlap=5,
        )
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        # Each chunk should have a corresponding segment
        for chunk in project._chunks:
            assert chunk.segment_idx in project._segments, (
                f"Chunk {chunk.chunk_id} has segment_idx {chunk.segment_idx} "
                f"not in segments map"
            )
            # Chunk content should appear in its segment
            segment_text = project._segments[chunk.segment_idx]
            # Due to overlap, chunk might be in adjacent segments too,
            # but should be findable in declared segment
            assert chunk.content[:50] in segment_text or segment_text[:50] in chunk.content, (
                f"Chunk content not found in segment {chunk.segment_idx}"
            )
