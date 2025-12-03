"""Unit tests for chunker module."""

import pytest


@pytest.mark.unit
class TestTokenCounting:
    """Test token counting functionality."""

    def test_count_tokens_empty(self):
        """Test counting tokens in empty string."""
        from konte.chunker import count_tokens

        assert count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Test counting tokens in simple text."""
        from konte.chunker import count_tokens

        tokens = count_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_count_tokens_longer_text(self):
        """Test counting tokens in longer text."""
        from konte.chunker import count_tokens

        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = count_tokens(text)
        assert tokens > 50


@pytest.mark.unit
class TestSegmentSplitting:
    """Test segment splitting with overlap."""

    def test_segment_small_document(self):
        """Test that small documents remain as single segment."""
        from konte.chunker import segment_document

        text = "This is a small document."
        segments = segment_document(text, segment_size=8000, overlap=800)
        assert len(segments) == 1
        assert segments[0] == text

    def test_segment_large_document_creates_multiple(self):
        """Test that large documents are split into multiple segments."""
        from konte.chunker import segment_document

        text = "word " * 5000  # ~5000 tokens
        segments = segment_document(text, segment_size=1000, overlap=100)
        assert len(segments) > 1

    def test_segment_overlap_exists(self):
        """Test that segments have overlapping content."""
        from konte.chunker import segment_document

        text = "word " * 3000
        segments = segment_document(text, segment_size=1000, overlap=100)

        if len(segments) > 1:
            # Check that end of first segment appears in start of second
            first_end = segments[0][-200:]
            second_start = segments[1][:200]
            # Some overlap should exist
            assert len(set(first_end.split()) & set(second_start.split())) > 0

    def test_segment_no_word_breakage(self):
        """Test that segments don't break in middle of words."""
        from konte.chunker import segment_document

        text = "supercalifragilisticexpialidocious " * 500
        segments = segment_document(text, segment_size=100, overlap=10)

        for segment in segments:
            # Segment should not start or end with partial word
            assert not segment.startswith("ous ")
            assert not segment.endswith("supercalifrag")


@pytest.mark.unit
class TestChunkSplitting:
    """Test chunk splitting with overlap."""

    def test_chunk_small_segment(self):
        """Test that small segments remain as single chunk."""
        from konte.chunker import chunk_segment

        text = "This is a small segment."
        chunks = chunk_segment(text, chunk_size=800, overlap=80)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_creates_multiple(self):
        """Test that larger segments create multiple chunks."""
        from konte.chunker import chunk_segment

        text = "word " * 1000  # ~1000 tokens
        chunks = chunk_segment(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_chunk_overlap_exists(self):
        """Test that chunks have overlapping content."""
        from konte.chunker import chunk_segment

        text = "word " * 500
        chunks = chunk_segment(text, chunk_size=100, overlap=10)

        if len(chunks) > 1:
            first_end = chunks[0][-50:]
            second_start = chunks[1][:50]
            assert len(set(first_end.split()) & set(second_start.split())) > 0

    def test_chunk_no_word_breakage(self):
        """Test that chunks start and end at word boundaries."""
        from konte.chunker import chunk_segment

        text = "hello world this is a test document with many words " * 50
        chunks = chunk_segment(text, chunk_size=50, overlap=5)

        for chunk in chunks:
            stripped = chunk.strip()
            # Should not start with lowercase continuation of previous word
            # (allowing uppercase as sentence start is fine)
            if stripped:
                # Check that chunk doesn't obviously start mid-word
                assert not stripped[0].islower() or stripped.split()[0] in text.split()


@pytest.mark.unit
class TestCreateChunks:
    """Test full chunking pipeline."""

    def test_create_chunks_returns_chunk_objects(self):
        """Test that create_chunks returns Chunk objects."""
        from konte.chunker import create_chunks
        from konte.models import Chunk

        text = "This is test content. " * 50
        chunks = create_chunks(
            text,
            source="test.txt",
            segment_size=500,
            segment_overlap=50,
            chunk_size=100,
            chunk_overlap=10,
        )

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_create_chunks_has_correct_metadata(self):
        """Test that chunks have correct metadata."""
        from konte.chunker import create_chunks

        text = "Content here. " * 100
        chunks = create_chunks(
            text,
            source="document.pdf",
            segment_size=200,
            segment_overlap=20,
            chunk_size=50,
            chunk_overlap=5,
        )

        for chunk in chunks:
            assert chunk.source == "document.pdf"
            assert chunk.segment_idx >= 0
            assert chunk.chunk_idx >= 0
            assert chunk.chunk_id.startswith("document.pdf_s")

    def test_create_chunks_unique_ids(self):
        """Test that all chunk IDs are unique."""
        from konte.chunker import create_chunks

        text = "Some text content. " * 200
        chunks = create_chunks(
            text,
            source="test.txt",
            segment_size=300,
            segment_overlap=30,
            chunk_size=80,
            chunk_overlap=8,
        )

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
