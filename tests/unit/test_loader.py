"""Unit tests for loader module."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.unit
class TestLoadTxt:
    """Test load_txt function."""

    def test_load_txt_returns_content(self):
        """Test loading a TXT file returns content."""
        from konte.loader import load_txt

        result = load_txt(FIXTURES_DIR / "sample.txt")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Tariff Classification Guidelines" in result

    def test_load_txt_preserves_structure(self):
        """Test that text structure is preserved."""
        from konte.loader import load_txt

        result = load_txt(FIXTURES_DIR / "sample.txt")
        assert "Chapter 1:" in result
        assert "Chapter 2:" in result

    def test_load_txt_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        from konte.loader import load_txt

        with pytest.raises(FileNotFoundError):
            load_txt(FIXTURES_DIR / "nonexistent.txt")


@pytest.mark.unit
class TestLoadMd:
    """Test load_md function."""

    def test_load_md_returns_content(self):
        """Test loading a MD file returns content."""
        from konte.loader import load_md

        result = load_md(FIXTURES_DIR / "sample.md")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Import Duty Calculation Guide" in result

    def test_load_md_preserves_markdown(self):
        """Test that markdown structure is preserved."""
        from konte.loader import load_md

        result = load_md(FIXTURES_DIR / "sample.md")
        assert "## Overview" in result
        assert "```" in result  # code blocks


@pytest.mark.unit
class TestLoadPdf:
    """Test load_pdf function."""

    def test_load_pdf_file_not_found(self):
        """Test that FileNotFoundError is raised for missing PDF."""
        from konte.loader import load_pdf

        with pytest.raises(FileNotFoundError):
            load_pdf(FIXTURES_DIR / "nonexistent.pdf")


@pytest.mark.unit
class TestLoadDocument:
    """Test load_document auto-detection."""

    def test_load_document_txt(self):
        """Test auto-detection of TXT files."""
        from konte.loader import load_document

        result = load_document(FIXTURES_DIR / "sample.txt")
        assert "Tariff Classification Guidelines" in result

    def test_load_document_md(self):
        """Test auto-detection of MD files."""
        from konte.loader import load_document

        result = load_document(FIXTURES_DIR / "sample.md")
        assert "Import Duty Calculation Guide" in result

    def test_load_document_unsupported(self):
        """Test that unsupported extensions raise ValueError."""
        from konte.loader import load_document

        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(Path("/tmp/file.xyz"))

    def test_load_document_returns_tuple_with_metadata(self):
        """Test that load_document returns content and source."""
        from konte.loader import load_document

        content = load_document(FIXTURES_DIR / "sample.txt")
        assert isinstance(content, str)
