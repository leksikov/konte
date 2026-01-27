"""Integration tests for context generation module (requires OPENAI_API_KEY)."""

import os
from pathlib import Path

import pytest

from konte.models import Chunk

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.mark.integration
class TestLoadPromptTemplate:
    """Test prompt template loading."""

    def test_load_default_prompt(self):
        """Test loading the default prompt template."""
        from konte.context import load_prompt_template

        template = load_prompt_template()
        assert "{segment}" in template
        assert "{chunk}" in template

    def test_load_custom_prompt(self, tmp_path):
        """Test loading a custom prompt template."""
        from konte.context import load_prompt_template

        custom_prompt = "Segment: {segment}\nChunk: {chunk}\nGenerate context."
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text(custom_prompt)

        template = load_prompt_template(prompt_file)
        assert template == custom_prompt


@pytest.mark.integration
class TestGenerateContext:
    """Test single context generation with real API calls."""

    @pytest.fixture
    def sample_segment(self):
        """Load sample segment from fixtures."""
        return (FIXTURES_DIR / "sample.txt").read_text()

    @pytest.fixture
    def sample_chunk(self, sample_segment):
        """Create a sample chunk from the segment."""
        content = sample_segment[:500]
        return Chunk(
            chunk_id="test_s0_c0",
            content=content,
            source="sample.txt",
            segment_idx=0,
            chunk_idx=0,
        )

    async def test_generate_context_returns_string(self, sample_segment, sample_chunk):
        """Test that generate_context returns a non-empty string."""
        from konte.context import generate_context

        context = await generate_context(
            segment=sample_segment,
            chunk=sample_chunk,
        )

        assert isinstance(context, str)
        assert len(context) > 0

    async def test_generate_context_is_relevant(self, sample_segment, sample_chunk):
        """Test that generated context is relevant to the content."""
        from konte.context import generate_context

        context = await generate_context(
            segment=sample_segment,
            chunk=sample_chunk,
        )

        # Context should mention something related to tariffs or classification
        # Support both English and Korean (Backend.AI may generate Korean)
        context_lower = context.lower()
        relevant_terms = [
            # English
            "tariff", "classification", "customs", "import", "duty", "chapter",
            # Korean
            "관세", "분류", "세관", "수입", "hs", "상품", "코드",
        ]
        has_relevant = any(term in context_lower for term in relevant_terms)
        assert has_relevant, f"Context not relevant: {context}"


@pytest.mark.integration
class TestGenerateContextsBatch:
    """Test batch context generation with real API calls."""

    @pytest.fixture
    def sample_segment(self):
        """Load sample segment from fixtures."""
        return (FIXTURES_DIR / "sample.txt").read_text()

    @pytest.fixture
    def sample_chunks(self, sample_segment):
        """Create multiple sample chunks."""
        chunks = []
        for i in range(3):
            start = i * 200
            end = start + 300
            content = sample_segment[start:end]
            chunk = Chunk(
                chunk_id=f"test_s0_c{i}",
                content=content,
                source="sample.txt",
                segment_idx=0,
                chunk_idx=i,
            )
            chunks.append(chunk)
        return chunks

    async def test_batch_returns_contextualized_chunks(
        self, sample_segment, sample_chunks
    ):
        """Test that batch generation returns ContextualizedChunk objects."""
        from konte.context import generate_contexts_batch
        from konte.models import ContextualizedChunk

        results = await generate_contexts_batch(
            segment=sample_segment,
            chunks=sample_chunks,
        )

        assert len(results) == len(sample_chunks)
        assert all(isinstance(r, ContextualizedChunk) for r in results)

    async def test_batch_skip_context_returns_empty(
        self, sample_segment, sample_chunks
    ):
        """Test that skip_context=True returns chunks with empty context."""
        from konte.context import generate_contexts_batch

        results = await generate_contexts_batch(
            segment=sample_segment,
            chunks=sample_chunks,
            skip_context=True,
        )

        assert len(results) == len(sample_chunks)
        for result in results:
            assert result.context == ""
            assert result.contextualized_content == result.chunk.content

    async def test_batch_preserves_chunk_order(self, sample_segment, sample_chunks):
        """Test that batch results maintain chunk order."""
        from konte.context import generate_contexts_batch

        results = await generate_contexts_batch(
            segment=sample_segment,
            chunks=sample_chunks,
        )

        for i, result in enumerate(results):
            assert result.chunk.chunk_id == sample_chunks[i].chunk_id
