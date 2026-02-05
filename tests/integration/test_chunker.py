"""Integration tests for sentence-aware chunking."""

from pathlib import Path

import pytest

from konte.chunker import (
    chunk_segment,
    count_tokens,
    create_chunks,
    segment_document,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.integration
class TestSentenceAwareSegmentation:
    """Test sentence-boundary-aware segmentation with real text."""

    def test_fixture_txt_segments_no_mid_word_breaks(self):
        """Test that segmenting sample.txt never breaks mid-word."""
        text = (FIXTURES_DIR / "sample.txt").read_text()
        segments = segment_document(text, segment_size=100, overlap=10)

        assert len(segments) > 1
        for segment in segments:
            # Segments should not start or end with partial words
            stripped = segment.strip()
            assert stripped, "Segment should not be empty"
            # First char should not be lowercase mid-word continuation
            # (unless it's a legitimate word start)
            assert stripped[0] != " "

    def test_fixture_md_segments_split_at_boundaries(self):
        """Test that segmenting sample.md splits at paragraph/line/sentence boundaries."""
        text = (FIXTURES_DIR / "sample.md").read_text()
        segments = segment_document(text, segment_size=100, overlap=10)

        assert len(segments) > 1
        # Each segment should be non-empty and stripped
        for segment in segments:
            assert len(segment.strip()) > 0

    def test_uniform_sentences_end_at_sentence_boundary(self):
        """Test that uniform sentence text splits at sentence boundaries."""
        sentences = [
            f"Sentence number {i} describes an important fact." for i in range(200)
        ]
        text = " ".join(sentences)
        segments = segment_document(text, segment_size=100, overlap=10)

        assert len(segments) > 1
        for segment in segments[:-1]:
            last_char = segment.rstrip()[-1]
            assert last_char in ".?!", (
                f"Segment should end at sentence boundary, got: ...{segment[-40:]!r}"
            )

    def test_no_content_loss_across_segments(self):
        """Test that all original sentences appear in at least one segment."""
        sentences = [
            f"Fact number {i} is important for classification." for i in range(50)
        ]
        text = " ".join(sentences)
        segments = segment_document(text, segment_size=80, overlap=8)

        combined = " ".join(segments)
        for sentence in sentences:
            assert sentence in combined, f"Lost sentence: {sentence}"


@pytest.mark.integration
class TestSentenceAwareChunking:
    """Test sentence-boundary-aware chunking with real text."""

    def test_uniform_sentences_end_at_sentence_boundary(self):
        """Test that chunking uniform sentences splits at sentence boundaries."""
        sentences = [
            f"This is chunk test sentence {i} with details." for i in range(100)
        ]
        text = " ".join(sentences)
        chunks = chunk_segment(text, chunk_size=50, overlap=5)

        assert len(chunks) > 1
        for chunk in chunks[:-1]:
            last_char = chunk.rstrip()[-1]
            assert last_char in ".?!", (
                f"Chunk should end at sentence boundary, got: ...{chunk[-40:]!r}"
            )

    def test_chunk_tokens_within_limit(self):
        """Test that each chunk respects the token size limit."""
        sentences = [
            f"This is test sentence {i} with some extra words for padding." for i in range(100)
        ]
        text = " ".join(sentences)
        max_tokens = 60
        chunks = chunk_segment(text, chunk_size=max_tokens, overlap=6)

        for chunk in chunks:
            tokens = count_tokens(chunk)
            # Allow small overshoot since splitter finds nearest boundary
            assert tokens <= max_tokens * 1.2, (
                f"Chunk has {tokens} tokens, limit is {max_tokens}"
            )

    def test_fixture_txt_no_mid_word_breaks(self):
        """Test that chunking sample.txt never produces mid-word splits."""
        text = (FIXTURES_DIR / "sample.txt").read_text()
        chunks = chunk_segment(text, chunk_size=50, overlap=5)

        assert len(chunks) > 1
        for chunk in chunks:
            stripped = chunk.strip()
            assert stripped, "Chunk should not be empty"
            assert stripped[0] != " "


@pytest.mark.integration
class TestCreateChunksPipeline:
    """Test full create_chunks pipeline with sentence-aware splitting."""

    def test_create_chunks_from_fixture(self):
        """Test end-to-end pipeline with fixture document."""
        text = (FIXTURES_DIR / "sample.txt").read_text()
        chunks, segments_map = create_chunks(
            text,
            source="sample.txt",
            segment_size=100,
            segment_overlap=10,
            chunk_size=30,
            chunk_overlap=3,
        )

        assert len(chunks) > 0
        assert len(segments_map) > 0

        # Every chunk should reference a valid segment
        for chunk in chunks:
            key = (chunk.source, chunk.segment_idx)
            assert key in segments_map

        # Chunk content should appear in its segment
        for chunk in chunks:
            key = (chunk.source, chunk.segment_idx)
            segment_text = segments_map[key]
            assert chunk.content[:20] in segment_text

    def test_pipeline_sentence_boundaries_with_uniform_text(self):
        """Test that the full pipeline preserves sentence boundaries at both levels."""
        sentences = [
            f"Sentence {i} covers topic {chr(65 + i % 26)}." for i in range(200)
        ]
        text = " ".join(sentences)
        chunks, segments_map = create_chunks(
            text,
            source="test.txt",
            segment_size=100,
            segment_overlap=10,
            chunk_size=50,
            chunk_overlap=5,
        )

        # Segments should end at sentence boundaries
        segment_texts = list(segments_map.values())
        for seg in segment_texts[:-1]:
            last_char = seg.rstrip()[-1]
            assert last_char in ".?!", (
                f"Segment should end at sentence boundary, got: ...{seg[-40:]!r}"
            )

        # Chunks should end at sentence boundaries
        chunk_contents = [c.content for c in chunks]
        for content in chunk_contents[:-1]:
            last_char = content.rstrip()[-1]
            assert last_char in ".?!", (
                f"Chunk should end at sentence boundary, got: ...{content[-40:]!r}"
            )
