"""E2E tests for project persistence: save → reload → query.

These tests validate that projects can be saved and loaded correctly,
with all indexes and data preserved.
"""

import os
from pathlib import Path

import pytest

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
class TestSaveAndReload:
    """Test project save and reload functionality."""

    async def test_save_reload_query_same_results(self, tmp_path):
        """Test that query returns same results after save/reload."""
        from konte.project import Project

        # Create and build project
        project1 = Project.create(
            name="e2e_persist_test",
            storage_path=tmp_path,
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)

        # Query before save
        query = "tariff classification"
        response1 = project1.query(query)

        # Save
        project1.save()

        # Reload into new instance
        project2 = Project.open(name="e2e_persist_test", storage_path=tmp_path)

        # Query after reload
        response2 = project2.query(query)

        # Results should be identical
        assert response1.total_found == response2.total_found
        assert response1.top_score == pytest.approx(response2.top_score, rel=1e-5)
        assert len(response1.results) == len(response2.results)

        # Check result ordering is preserved
        for r1, r2 in zip(response1.results, response2.results):
            assert r1.chunk_id == r2.chunk_id
            assert r1.score == pytest.approx(r2.score, rel=1e-5)
            assert r1.content == r2.content

    async def test_reload_preserves_config(self, tmp_path):
        """Test that configuration is preserved after reload."""
        from konte.project import Project

        # Create with custom config
        project1 = Project.create(
            name="e2e_config_persist",
            storage_path=tmp_path,
            chunk_size=500,
            chunk_overlap=50,
            segment_size=2000,
            segment_overlap=200,
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)
        project1.save()

        # Reload
        project2 = Project.open(name="e2e_config_persist", storage_path=tmp_path)

        # Verify config preserved
        assert project2.config.chunk_size == 500
        assert project2.config.chunk_overlap == 50
        assert project2.config.segment_size == 2000
        assert project2.config.segment_overlap == 200

    async def test_reload_preserves_chunks(self, tmp_path):
        """Test that all chunks are preserved after reload."""
        from konte.project import Project

        # Create and build
        project1 = Project.create(
            name="e2e_chunks_persist",
            storage_path=tmp_path,
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)

        num_chunks_before = len(project1._contextualized_chunks)
        chunk_ids_before = {c.chunk.chunk_id for c in project1._contextualized_chunks}

        project1.save()

        # Reload
        project2 = Project.open(name="e2e_chunks_persist", storage_path=tmp_path)

        num_chunks_after = len(project2._contextualized_chunks)
        chunk_ids_after = {c.chunk.chunk_id for c in project2._contextualized_chunks}

        # Same number of chunks
        assert num_chunks_before == num_chunks_after

        # Same chunk IDs
        assert chunk_ids_before == chunk_ids_after


@pytest.mark.e2e
class TestIndexPersistence:
    """Test that indexes are correctly persisted."""

    async def test_faiss_index_persisted(self, tmp_path):
        """Test that FAISS index is saved and loaded correctly."""
        from konte.project import Project

        project1 = Project.create(
            name="e2e_faiss_persist",
            storage_path=tmp_path,
            enable_bm25=False,  # FAISS only
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)

        # Verify FAISS index file exists after save
        project1.save()
        faiss_path = tmp_path / "e2e_faiss_persist" / "faiss.faiss"
        assert faiss_path.exists(), "FAISS index file not created"

        # Reload and verify semantic search works
        project2 = Project.open(name="e2e_faiss_persist", storage_path=tmp_path)
        response = project2.query("tariff", mode="semantic")
        assert response.total_found > 0

    async def test_bm25_index_persisted(self, tmp_path):
        """Test that BM25 index is saved and loaded correctly."""
        from konte.project import Project

        project1 = Project.create(
            name="e2e_bm25_persist",
            storage_path=tmp_path,
            enable_faiss=False,  # BM25 only
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)

        # Verify BM25 index file exists after save
        project1.save()
        bm25_path = tmp_path / "e2e_bm25_persist" / "bm25.pkl"
        assert bm25_path.exists(), "BM25 index file not created"

        # Reload and verify lexical search works
        project2 = Project.open(name="e2e_bm25_persist", storage_path=tmp_path)
        response = project2.query("tariff", mode="lexical")
        assert response.total_found > 0

    async def test_both_indexes_persisted(self, tmp_path):
        """Test that both indexes are saved and loaded for hybrid mode."""
        from konte.project import Project

        project1 = Project.create(
            name="e2e_hybrid_persist",
            storage_path=tmp_path,
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)
        project1.save()

        # Both files should exist
        project_dir = tmp_path / "e2e_hybrid_persist"
        assert (project_dir / "faiss.faiss").exists()
        assert (project_dir / "bm25.pkl").exists()
        assert (project_dir / "chunks.json").exists()
        assert (project_dir / "config.json").exists()

        # Reload and verify hybrid search works
        project2 = Project.open(name="e2e_hybrid_persist", storage_path=tmp_path)
        response = project2.query("tariff", mode="hybrid")
        assert response.total_found > 0


@pytest.mark.e2e
class TestContextPersistence:
    """Test that context is preserved across save/reload."""

    async def test_context_preserved_after_reload(self, tmp_path):
        """Test that generated context is saved and loaded."""
        from konte.project import Project

        project1 = Project.create(
            name="e2e_context_persist",
            storage_path=tmp_path,
            chunk_size=200,  # Smaller for faster test
            chunk_overlap=20,
        )
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)  # Generate context

        # Get context for first chunk
        if project1._contextualized_chunks:
            context_before = project1._contextualized_chunks[0].context

        project1.save()

        # Reload
        project2 = Project.open(name="e2e_context_persist", storage_path=tmp_path)

        # Context should be preserved
        if project2._contextualized_chunks:
            context_after = project2._contextualized_chunks[0].context
            assert context_before == context_after
