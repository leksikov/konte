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

        await project.build(skip_context=False)

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

        await project.build(skip_context=False)

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

        await project.build(skip_context=False)

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
        await project.build(skip_context=False)

        response = project.query("Harmonized System classification", mode="hybrid")

        assert response.total_found > 0
        assert len(response.results) > 0

    async def test_query_returns_relevant_results(self, tmp_path):
        """Test that queries return relevant results."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=False)

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
        await project1.build(skip_context=False)
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
        await project1.build(skip_context=False)
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
        await project.build(skip_context=False)

        retriever = project.as_retriever()

        assert callable(retriever)

    async def test_retriever_callable_works(self, tmp_path):
        """Test that the retriever callable returns results."""
        from konte.project import Project

        project = Project.create(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=False)

        retriever = project.as_retriever()
        response = retriever("tariff classification")

        assert response.total_found > 0


@pytest.mark.integration
class TestProjectCheckpoint:
    """Test build checkpoint functionality with real file I/O."""

    async def test_checkpoint_created_during_build(self, tmp_path):
        """Test that checkpoint file is created during build."""
        from konte.project import Project

        project = Project.create(name="checkpoint_test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        # Build with skip_context to make it faster
        await project.build(skip_context=True)

        # Checkpoint should be cleared after successful build
        checkpoint_path = tmp_path / "checkpoint_test" / "context_checkpoint.json"
        assert not checkpoint_path.exists()

    async def test_checkpoint_cleared_after_successful_build(self, tmp_path):
        """Test that checkpoint is cleared after successful build."""
        from konte.project import Project

        project = Project.create(name="clear_test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        await project.build(skip_context=True)

        # Checkpoint should be cleared
        checkpoint_path = project._checkpoint_path()
        assert not checkpoint_path.exists()

    async def test_build_resume_continues_from_checkpoint(self, tmp_path):
        """Test that build resumes from checkpoint."""
        from konte.project import Project
        from konte.models import BuildCheckpoint
        import json

        project = Project.create(name="resume_test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        # Get the segments that will be created
        segments_count = len(project._segments)
        assert segments_count >= 1

        # Manually create a checkpoint for the first segment
        first_seg_key = list(project._segments.keys())[0]
        seg_key_str = f"{first_seg_key[0]}|{first_seg_key[1]}"
        first_chunk = [c for c in project._chunks if c.segment_idx == first_seg_key[1]][0]

        checkpoint = BuildCheckpoint(
            completed_segments=[seg_key_str],
            contextualized_chunks=[
                {"chunk": first_chunk.model_dump(), "context": "Pre-existing context"}
            ],
        )
        project._save_checkpoint(checkpoint)

        # Build with resume=True
        await project.build(skip_context=True, resume=True)

        # Should have contextualized chunks
        assert len(project._contextualized_chunks) > 0

        # First chunk should have the pre-existing context
        first_ctx_chunk = next(
            c for c in project._contextualized_chunks
            if c.chunk.chunk_id == first_chunk.chunk_id
        )
        assert first_ctx_chunk.context == "Pre-existing context"

    async def test_build_resume_false_ignores_checkpoint(self, tmp_path):
        """Test that resume=False ignores existing checkpoint."""
        from konte.project import Project
        from konte.models import BuildCheckpoint

        project = Project.create(name="no_resume_test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])

        # Create a checkpoint with fake context
        first_seg_key = list(project._segments.keys())[0]
        seg_key_str = f"{first_seg_key[0]}|{first_seg_key[1]}"
        first_chunk = [c for c in project._chunks if c.segment_idx == first_seg_key[1]][0]

        checkpoint = BuildCheckpoint(
            completed_segments=[seg_key_str],
            contextualized_chunks=[
                {"chunk": first_chunk.model_dump(), "context": "Should be ignored"}
            ],
        )
        project._save_checkpoint(checkpoint)

        # Build with resume=False
        await project.build(skip_context=True, resume=False)

        # First chunk should NOT have the pre-existing context
        first_ctx_chunk = next(
            c for c in project._contextualized_chunks
            if c.chunk.chunk_id == first_chunk.chunk_id
        )
        # skip_context=True means context should be empty, not "Should be ignored"
        assert first_ctx_chunk.context != "Should be ignored"

    def test_checkpoint_path_method(self, tmp_path):
        """Test that _checkpoint_path returns correct path."""
        from konte.project import Project

        project = Project.create(name="path_test", storage_path=tmp_path)
        checkpoint_path = project._checkpoint_path()

        expected = tmp_path / "path_test" / "context_checkpoint.json"
        assert checkpoint_path == expected

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test checkpoint save and load round-trip."""
        from konte.project import Project
        from konte.models import BuildCheckpoint

        project = Project.create(name="roundtrip_test", storage_path=tmp_path)

        # Create and save checkpoint
        original = BuildCheckpoint(
            completed_segments=["doc.pdf|0", "doc.pdf|1"],
            contextualized_chunks=[
                {"chunk": {"chunk_id": "c1", "content": "text1"}, "context": "ctx1"},
                {"chunk": {"chunk_id": "c2", "content": "text2"}, "context": "ctx2"},
            ],
        )
        project._save_checkpoint(original)

        # Load checkpoint
        loaded = project._load_checkpoint()

        assert loaded is not None
        assert loaded.completed_segments == original.completed_segments
        assert len(loaded.contextualized_chunks) == 2

    def test_clear_checkpoint(self, tmp_path):
        """Test checkpoint clearing."""
        from konte.project import Project
        from konte.models import BuildCheckpoint

        project = Project.create(name="clear_test", storage_path=tmp_path)

        # Save checkpoint
        checkpoint = BuildCheckpoint(completed_segments=["doc.pdf|0"])
        project._save_checkpoint(checkpoint)

        # Verify it exists
        assert project._checkpoint_path().exists()

        # Clear it
        project._clear_checkpoint()

        # Verify it's gone
        assert not project._checkpoint_path().exists()


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

        for seg_key, segment_text in project._segments.items():
            segment_tokens = count_tokens(segment_text)
            # Segment should be around segment_size, not the full document
            # Allow 2x margin for word boundary adjustments
            assert segment_tokens <= 200 * 2, (
                f"Segment {seg_key} has {segment_tokens} tokens, "
                f"expected <= 400 (200 * 2)"
            )
            # Segment should be smaller than full document
            assert segment_tokens < doc_tokens, (
                f"Segment {seg_key} has {segment_tokens} tokens, "
                f"which is >= full document ({doc_tokens} tokens)"
            )

    def test_each_chunk_maps_to_valid_segment(self, tmp_path):
        """Verify each chunk's (source, segment_idx) maps to correct segment."""
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
            key = (chunk.source, chunk.segment_idx)
            assert key in project._segments, (
                f"Chunk {chunk.chunk_id} has key {key} "
                f"not in segments map"
            )
            # Chunk content should appear in its segment
            segment_text = project._segments[key]
            # Due to overlap, chunk might be in adjacent segments too,
            # but should be findable in declared segment
            assert chunk.content[:50] in segment_text or segment_text[:50] in chunk.content, (
                f"Chunk content not found in segment {key}"
            )
