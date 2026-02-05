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

    def test_segment_no_sentence_breakage(self):
        """Test that segments end at sentence boundaries, not mid-sentence."""
        from konte.chunker import segment_document

        sentences = [
            f"Sentence number {i} describes an important fact." for i in range(200)
        ]
        text = " ".join(sentences)
        segments = segment_document(text, segment_size=100, overlap=10)

        assert len(segments) > 1
        # All segments except the last should end with sentence-ending punctuation
        for segment in segments[:-1]:
            last_char = segment.rstrip()[-1]
            assert last_char in ".?!", (
                f"Segment should end at sentence boundary, but ends with: "
                f"...{segment[-40:]!r}"
            )


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

    def test_chunk_no_sentence_breakage(self):
        """Test that chunks end at sentence boundaries, not mid-sentence."""
        from konte.chunker import chunk_segment

        sentences = [
            f"This is chunk test sentence {i} with details." for i in range(100)
        ]
        text = " ".join(sentences)
        chunks = chunk_segment(text, chunk_size=50, overlap=5)

        assert len(chunks) > 1
        # All chunks except the last should end with sentence-ending punctuation
        for chunk in chunks[:-1]:
            last_char = chunk.rstrip()[-1]
            assert last_char in ".?!", (
                f"Chunk should end at sentence boundary, but ends with: "
                f"...{chunk[-40:]!r}"
            )


@pytest.mark.unit
class TestCreateChunks:
    """Test full chunking pipeline."""

    def test_create_chunks_returns_chunk_objects(self):
        """Test that create_chunks returns Chunk objects and segments map."""
        from konte.chunker import create_chunks
        from konte.models import Chunk

        text = "This is test content. " * 50
        chunks, segments_map = create_chunks(
            text,
            source="test.txt",
            segment_size=500,
            segment_overlap=50,
            chunk_size=100,
            chunk_overlap=10,
        )

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert isinstance(segments_map, dict)
        assert len(segments_map) > 0

    def test_create_chunks_has_correct_metadata(self):
        """Test that chunks have correct metadata."""
        from konte.chunker import create_chunks

        text = "Content here. " * 100
        chunks, segments_map = create_chunks(
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
            # Verify (source, segment_idx) maps to a segment
            key = (chunk.source, chunk.segment_idx)
            assert key in segments_map

    def test_create_chunks_unique_ids(self):
        """Test that all chunk IDs are unique."""
        from konte.chunker import create_chunks

        text = "Some text content. " * 200
        chunks, _ = create_chunks(
            text,
            source="test.txt",
            segment_size=300,
            segment_overlap=30,
            chunk_size=80,
            chunk_overlap=8,
        )

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.unit
class TestCreateChunksDataFlow:
    """Test data flow contract for create_chunks - segments_map correctness."""

    def test_segments_map_contains_actual_segments(self):
        """Verify segments_map contains segment text, not full document."""
        from konte.chunker import create_chunks, count_tokens

        # Create text large enough to produce MANY segments (10x segment size)
        text = "Word " * 10000  # ~10000 tokens
        chunks, segments_map = create_chunks(
            text,
            source="test.txt",
            segment_size=1000,
            segment_overlap=100,
            chunk_size=200,
            chunk_overlap=20,
        )

        doc_tokens = count_tokens(text)

        # Should have many segments (at least 5)
        assert len(segments_map) >= 5, f"Expected at least 5 segments, got {len(segments_map)}"

        # Each segment should be SIGNIFICANTLY smaller than full document
        # This is the critical assertion - catches bug where full doc stored as segment
        for seg_key, segment_text in segments_map.items():
            segment_tokens = count_tokens(segment_text)
            # Segment should be at most 1/4 of document size
            # With overlap, segments can be up to ~2x segment_size, so allow ~2000 tokens
            # But definitely NOT the full 10000 token document
            assert segment_tokens < doc_tokens / 4, (
                f"Segment {seg_key} has {segment_tokens} tokens, "
                f"should be less than 1/4 of document ({doc_tokens / 4} tokens). "
                f"Full document has {doc_tokens} tokens."
            )

    def test_chunk_segment_idx_maps_to_segments_map(self):
        """Verify every chunk.(source, segment_idx) has corresponding entry in segments_map."""
        from konte.chunker import create_chunks

        text = "Content " * 500
        chunks, segments_map = create_chunks(
            text,
            source="test.txt",
            segment_size=150,
            segment_overlap=15,
            chunk_size=40,
            chunk_overlap=4,
        )

        for chunk in chunks:
            key = (chunk.source, chunk.segment_idx)
            assert key in segments_map, (
                f"Chunk {chunk.chunk_id} has key={key} "
                f"not found in segments_map keys: {list(segments_map.keys())}"
            )

    def test_chunk_content_exists_in_its_segment(self):
        """Verify chunk content can be found within its declared segment."""
        from konte.chunker import create_chunks

        text = "Unique word alpha. " * 100 + "Unique word beta. " * 100 + "Unique word gamma. " * 100
        chunks, segments_map = create_chunks(
            text,
            source="test.txt",
            segment_size=100,
            segment_overlap=10,
            chunk_size=30,
            chunk_overlap=3,
        )

        for chunk in chunks:
            key = (chunk.source, chunk.segment_idx)
            segment_text = segments_map[key]
            # First 20 chars of chunk should appear in segment
            chunk_start = chunk.content[:20]
            assert chunk_start in segment_text, (
                f"Chunk content '{chunk_start}...' not found in segment {key}"
            )

    def test_segments_map_indices_are_contiguous(self):
        """Verify segments_map has contiguous indices starting from 0 for each source."""
        from konte.chunker import create_chunks

        text = "Test content. " * 300
        _, segments_map = create_chunks(
            text,
            source="test.txt",
            segment_size=100,
            segment_overlap=10,
            chunk_size=25,
            chunk_overlap=2,
        )

        # Extract indices for the source
        indices = sorted(idx for (source, idx) in segments_map.keys() if source == "test.txt")
        expected = list(range(len(indices)))
        assert indices == expected, (
            f"Segment indices not contiguous: got {indices}, expected {expected}"
        )


@pytest.mark.unit
class TestExtractMetadataFromSource:
    """Test metadata extraction from source filenames."""

    def test_single_word_company(self):
        """Test single word company name."""
        from konte.chunker import extract_metadata_from_source

        result = extract_metadata_from_source("ADOBE_2022_10K.md")
        assert result == {"company": "ADOBE", "year": "2022"}

    def test_multi_word_company(self):
        """Test multi-word company name with underscores."""
        from konte.chunker import extract_metadata_from_source

        result = extract_metadata_from_source("JOHNSON_JOHNSON_2022_10K.md")
        assert result == {"company": "JOHNSON_JOHNSON", "year": "2022"}

    def test_numeric_company(self):
        """Test company name starting with number."""
        from konte.chunker import extract_metadata_from_source

        result = extract_metadata_from_source("3M_2018_10K.md")
        assert result == {"company": "3M", "year": "2018"}

    def test_no_match(self):
        """Test filename that doesn't match pattern."""
        from konte.chunker import extract_metadata_from_source

        result = extract_metadata_from_source("readme.md")
        assert result == {}

    def test_lowercase_filename(self):
        """Test lowercase filename."""
        from konte.chunker import extract_metadata_from_source

        result = extract_metadata_from_source("apple_2023_annual.txt")
        assert result == {"company": "APPLE", "year": "2023"}
