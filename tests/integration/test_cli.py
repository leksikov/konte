"""Integration tests for CLI module (requires OPENAI_API_KEY)."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture
def sample_doc(tmp_path):
    """Create a sample document for testing."""
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        """
        Tariff Code 8542.31 - Electronic Integrated Circuits

        This section covers semiconductor devices and integrated circuits.
        The import duty rate is 5% for MFN (Most Favored Nation) countries.

        Classification rules:
        1. Processors and controllers
        2. Memories (RAM, ROM, EPROM)
        3. Amplifiers and converters

        Anti-dumping duties may apply to imports from certain countries.
        The Harmonized System uses 6-digit codes for international classification.
        """
    )
    return doc_path


@pytest.mark.integration
class TestCLIAddDocuments:
    """Test CLI add command with real document loading."""

    def test_add_document_creates_chunks(self, tmp_path, sample_doc):
        """Test that adding a document creates chunks."""
        from konte.cli.app import app

        # Create project
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        # Add document
        result = runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Added 1 document(s)" in result.stdout
        assert "Total chunks:" in result.stdout

    def test_add_multiple_documents(self, tmp_path, sample_doc):
        """Test adding multiple documents."""
        from konte.cli.app import app

        # Create another doc
        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("Additional document content for testing.")

        # Create project
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        # Add documents
        result = runner.invoke(app, ["add", "test_project", str(sample_doc), str(doc2), "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Added 2 document(s)" in result.stdout


@pytest.mark.integration
class TestCLIBuildIndexes:
    """Test CLI build command with real API calls."""

    def test_build_with_skip_context(self, tmp_path, sample_doc):
        """Test building indexes without context generation."""
        from konte.cli.app import app

        # Create and add document
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])

        # Build with skip-context (fast, no LLM calls)
        result = runner.invoke(app, ["build", "test_project", "--skip-context", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Build complete" in result.stdout

    def test_build_faiss_only(self, tmp_path, sample_doc):
        """Test building FAISS index only."""
        from konte.cli.app import app

        # Create and add document
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])

        # Build FAISS only
        result = runner.invoke(app, ["build", "test_project", "--faiss-only", "--skip-context", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "FAISS index: enabled" in result.stdout
        assert "BM25 index: disabled" in result.stdout

    def test_build_bm25_only(self, tmp_path, sample_doc):
        """Test building BM25 index only."""
        from konte.cli.app import app

        # Create and add document
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])

        # Build BM25 only
        result = runner.invoke(app, ["build", "test_project", "--bm25-only", "--skip-context", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "FAISS index: disabled" in result.stdout
        assert "BM25 index: enabled" in result.stdout


@pytest.mark.integration
class TestCLIQuery:
    """Test CLI query command with real retrieval."""

    def test_query_returns_results(self, tmp_path, sample_doc):
        """Test querying returns relevant results."""
        from konte.cli.app import app

        # Create, add, and build
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])
        runner.invoke(app, ["build", "test_project", "--skip-context", "--storage", str(tmp_path)])

        # Query
        result = runner.invoke(app, ["query", "test_project", "tariff code electronics", "--top-k", "3", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Query:" in result.stdout
        assert "Results:" in result.stdout
        assert "Top Score:" in result.stdout

    def test_query_semantic_mode(self, tmp_path, sample_doc):
        """Test querying with semantic mode."""
        from konte.cli.app import app

        # Create, add, and build
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])
        runner.invoke(app, ["build", "test_project", "--skip-context", "--storage", str(tmp_path)])

        # Query with semantic mode
        result = runner.invoke(app, ["query", "test_project", "integrated circuits", "--mode", "semantic", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Mode: semantic" in result.stdout

    def test_query_lexical_mode(self, tmp_path, sample_doc):
        """Test querying with lexical mode."""
        from konte.cli.app import app

        # Create, add, and build
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])
        runner.invoke(app, ["build", "test_project", "--skip-context", "--storage", str(tmp_path)])

        # Query with lexical mode
        result = runner.invoke(app, ["query", "test_project", "8542.31", "--mode", "lexical", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Mode: lexical" in result.stdout


@pytest.mark.integration
class TestCLIInfoAfterBuild:
    """Test CLI info command after building indexes."""

    def test_info_shows_indexes_built(self, tmp_path, sample_doc):
        """Test info shows indexes as built after build."""
        from konte.cli.app import app

        # Create, add, and build
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        runner.invoke(app, ["add", "test_project", str(sample_doc), "--storage", str(tmp_path)])
        runner.invoke(app, ["build", "test_project", "--skip-context", "--storage", str(tmp_path)])

        # Check info
        result = runner.invoke(app, ["info", "test_project", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "exists" in result.stdout  # FAISS Index and BM25 Index should show "exists"
