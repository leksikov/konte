"""Unit tests for models module."""

import pytest
from pydantic import ValidationError


@pytest.mark.unit
class TestChunk:
    """Test Chunk model."""

    def test_chunk_creation(self):
        """Test creating a valid Chunk."""
        from konte.models import Chunk

        chunk = Chunk(
            chunk_id="doc1_s0_c0",
            content="This is chunk content.",
            source="document.pdf",
            segment_idx=0,
            chunk_idx=0,
        )
        assert chunk.chunk_id == "doc1_s0_c0"
        assert chunk.content == "This is chunk content."
        assert chunk.source == "document.pdf"
        assert chunk.segment_idx == 0
        assert chunk.chunk_idx == 0

    def test_chunk_requires_content(self):
        """Test that content is required."""
        from konte.models import Chunk

        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="id1",
                source="doc.txt",
                segment_idx=0,
                chunk_idx=0,
            )

    def test_chunk_metadata_optional(self):
        """Test that metadata is optional and defaults to empty dict."""
        from konte.models import Chunk

        chunk = Chunk(
            chunk_id="id1",
            content="content",
            source="doc.txt",
            segment_idx=0,
            chunk_idx=0,
        )
        assert chunk.metadata == {}


@pytest.mark.unit
class TestContextualizedChunk:
    """Test ContextualizedChunk model."""

    def test_contextualized_chunk_creation(self):
        """Test creating a valid ContextualizedChunk."""
        from konte.models import Chunk, ContextualizedChunk

        chunk = Chunk(
            chunk_id="id1",
            content="The revenue grew by 3%.",
            source="report.pdf",
            segment_idx=0,
            chunk_idx=0,
        )
        ctx_chunk = ContextualizedChunk(
            chunk=chunk,
            context="This is from ACME Corp Q2 2023 earnings report.",
        )
        assert ctx_chunk.chunk == chunk
        assert ctx_chunk.context == "This is from ACME Corp Q2 2023 earnings report."
        expected = "This is from ACME Corp Q2 2023 earnings report. The revenue grew by 3%."
        assert ctx_chunk.contextualized_content == expected

    def test_contextualized_content_computation(self):
        """Test that contextualized_content is computed correctly."""
        from konte.models import Chunk, ContextualizedChunk

        chunk = Chunk(
            chunk_id="id1",
            content="Original content here.",
            source="doc.txt",
            segment_idx=0,
            chunk_idx=0,
        )
        ctx_chunk = ContextualizedChunk(
            chunk=chunk,
            context="Context prefix.",
        )
        assert ctx_chunk.contextualized_content == "Context prefix. Original content here."

    def test_empty_context(self):
        """Test ContextualizedChunk with empty context."""
        from konte.models import Chunk, ContextualizedChunk

        chunk = Chunk(
            chunk_id="id1",
            content="Content only.",
            source="doc.txt",
            segment_idx=0,
            chunk_idx=0,
        )
        ctx_chunk = ContextualizedChunk(
            chunk=chunk,
            context="",
        )
        assert ctx_chunk.contextualized_content == "Content only."


@pytest.mark.unit
class TestRetrievalResult:
    """Test RetrievalResult model."""

    def test_retrieval_result_creation(self):
        """Test creating a valid RetrievalResult."""
        from konte.models import RetrievalResult

        result = RetrievalResult(
            content="Original chunk content.",
            context="Added context.",
            score=0.85,
            source="document.pdf",
            chunk_id="doc1_s0_c0",
        )
        assert result.content == "Original chunk content."
        assert result.context == "Added context."
        assert result.score == 0.85
        assert result.source == "document.pdf"
        assert result.chunk_id == "doc1_s0_c0"
        assert result.metadata == {}

    def test_score_bounds(self):
        """Test that score is between 0 and 1."""
        from konte.models import RetrievalResult

        with pytest.raises(ValidationError):
            RetrievalResult(
                content="content",
                context="context",
                score=1.5,
                source="doc.txt",
                chunk_id="id1",
            )

        with pytest.raises(ValidationError):
            RetrievalResult(
                content="content",
                context="context",
                score=-0.1,
                source="doc.txt",
                chunk_id="id1",
            )


@pytest.mark.unit
class TestRetrievalResponse:
    """Test RetrievalResponse model."""

    def test_retrieval_response_creation(self):
        """Test creating a valid RetrievalResponse."""
        from konte.models import RetrievalResult, RetrievalResponse

        results = [
            RetrievalResult(
                content="content1",
                context="ctx1",
                score=0.8,
                source="doc1.txt",
                chunk_id="id1",
            ),
            RetrievalResult(
                content="content2",
                context="ctx2",
                score=0.6,
                source="doc2.txt",
                chunk_id="id2",
            ),
        ]
        response = RetrievalResponse(
            results=results,
            query="test query",
            total_found=2,
            top_score=0.8,
            score_spread=0.2,
            has_high_confidence=True,
            suggested_action="deliver",
        )
        assert len(response.results) == 2
        assert response.query == "test query"
        assert response.total_found == 2
        assert response.top_score == 0.8
        assert response.score_spread == 0.2
        assert response.has_high_confidence is True
        assert response.suggested_action == "deliver"

    def test_suggested_action_values(self):
        """Test that suggested_action is one of valid values."""
        from konte.models import RetrievalResponse

        for action in ["deliver", "query_more", "refine_query"]:
            response = RetrievalResponse(
                results=[],
                query="q",
                total_found=0,
                top_score=0.0,
                score_spread=0.0,
                has_high_confidence=False,
                suggested_action=action,
            )
            assert response.suggested_action == action

    def test_invalid_suggested_action(self):
        """Test that invalid suggested_action raises error."""
        from konte.models import RetrievalResponse

        with pytest.raises(ValidationError):
            RetrievalResponse(
                results=[],
                query="q",
                total_found=0,
                top_score=0.0,
                score_spread=0.0,
                has_high_confidence=False,
                suggested_action="invalid_action",
            )


@pytest.mark.unit
class TestProjectConfig:
    """Test ProjectConfig model."""

    def test_project_config_defaults(self):
        """Test ProjectConfig with default values."""
        from konte.models import ProjectConfig
        from pathlib import Path

        config = ProjectConfig(
            name="test_project",
            storage_path=Path("/tmp/test"),
        )
        assert config.name == "test_project"
        assert config.storage_path == Path("/tmp/test")
        assert config.segment_size == 8000
        assert config.segment_overlap == 800
        assert config.chunk_size == 800
        assert config.chunk_overlap == 80
        assert config.context_min_tokens == 100
        assert config.context_max_tokens == 200
        assert config.embedding_model == "text-embedding-3-small"
        assert config.context_model is None  # Default: use Backend.AI or settings
        assert config.enable_faiss is True
        assert config.enable_bm25 is True
        assert config.fusion_weight_semantic == 0.5
        assert config.fusion_weight_lexical == 0.5

    def test_project_config_custom_values(self):
        """Test ProjectConfig with custom values."""
        from konte.models import ProjectConfig
        from pathlib import Path

        config = ProjectConfig(
            name="custom_project",
            storage_path=Path("/custom/path"),
            segment_size=4000,
            chunk_size=400,
            enable_bm25=False,
            fusion_weight_semantic=0.7,
            fusion_weight_lexical=0.3,
        )
        assert config.segment_size == 4000
        assert config.chunk_size == 400
        assert config.enable_bm25 is False
        assert config.fusion_weight_semantic == 0.7

    def test_project_config_serialization(self):
        """Test that ProjectConfig can be serialized to JSON."""
        from konte.models import ProjectConfig
        from pathlib import Path
        import json

        config = ProjectConfig(
            name="test",
            storage_path=Path("/tmp/test"),
        )
        json_str = config.model_dump_json()
        data = json.loads(json_str)
        assert data["name"] == "test"
        assert "storage_path" in data
