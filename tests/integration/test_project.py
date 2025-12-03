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
