"""Unit tests for build checkpoint functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.unit
class TestCheckpointPath:
    """Test _checkpoint_path method."""

    def test_checkpoint_path_returns_correct_path(self, tmp_path):
        """Test that _checkpoint_path returns correct path."""
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)
        checkpoint_path = project._checkpoint_path()

        expected = tmp_path / "test_project" / "context_checkpoint.json"
        assert checkpoint_path == expected


@pytest.mark.unit
class TestLoadCheckpoint:
    """Test _load_checkpoint method."""

    def test_load_checkpoint_returns_none_if_not_exists(self, tmp_path):
        """Test that _load_checkpoint returns None if file doesn't exist."""
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)
        result = project._load_checkpoint()

        assert result is None

    def test_load_checkpoint_returns_checkpoint_if_exists(self, tmp_path):
        """Test that _load_checkpoint returns BuildCheckpoint if file exists."""
        from konte.project import Project
        import json

        project = Project.create(name="test_project", storage_path=tmp_path)

        # Create checkpoint file
        checkpoint_dir = tmp_path / "test_project"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "context_checkpoint.json"
        checkpoint_data = {
            "completed_segments": ["doc.pdf|0", "doc.pdf|1"],
            "contextualized_chunks": [
                {"chunk": {"chunk_id": "id1"}, "context": "ctx1"}
            ],
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data), encoding="utf-8")

        result = project._load_checkpoint()

        assert result is not None
        assert len(result.completed_segments) == 2
        assert result.completed_segments[0] == "doc.pdf|0"
        assert len(result.contextualized_chunks) == 1


@pytest.mark.unit
class TestSaveCheckpoint:
    """Test _save_checkpoint method."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        """Test that _save_checkpoint creates checkpoint file."""
        from konte.project import Project
        from konte.models import BuildCheckpoint

        project = Project.create(name="test_project", storage_path=tmp_path)
        checkpoint = BuildCheckpoint(
            completed_segments=["doc.pdf|0"],
            contextualized_chunks=[{"chunk": {"chunk_id": "id1"}, "context": "ctx"}],
        )

        project._save_checkpoint(checkpoint)

        checkpoint_path = tmp_path / "test_project" / "context_checkpoint.json"
        assert checkpoint_path.exists()

    def test_save_checkpoint_creates_directory(self, tmp_path):
        """Test that _save_checkpoint creates directory if needed."""
        from konte.project import Project
        from konte.models import BuildCheckpoint

        project = Project.create(name="new_project", storage_path=tmp_path)
        checkpoint = BuildCheckpoint()

        project._save_checkpoint(checkpoint)

        project_dir = tmp_path / "new_project"
        assert project_dir.exists()

    def test_save_checkpoint_content_is_valid_json(self, tmp_path):
        """Test that saved checkpoint is valid JSON."""
        from konte.project import Project
        from konte.models import BuildCheckpoint
        import json

        project = Project.create(name="test_project", storage_path=tmp_path)
        checkpoint = BuildCheckpoint(
            completed_segments=["doc.pdf|0", "doc.pdf|1"],
            contextualized_chunks=[
                {"chunk": {"chunk_id": "id1"}, "context": "ctx1"},
                {"chunk": {"chunk_id": "id2"}, "context": "ctx2"},
            ],
        )

        project._save_checkpoint(checkpoint)

        checkpoint_path = tmp_path / "test_project" / "context_checkpoint.json"
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert data["completed_segments"] == ["doc.pdf|0", "doc.pdf|1"]
        assert len(data["contextualized_chunks"]) == 2


@pytest.mark.unit
class TestClearCheckpoint:
    """Test _clear_checkpoint method."""

    def test_clear_checkpoint_removes_file(self, tmp_path):
        """Test that _clear_checkpoint removes checkpoint file."""
        from konte.project import Project
        import json

        project = Project.create(name="test_project", storage_path=tmp_path)

        # Create checkpoint file
        checkpoint_dir = tmp_path / "test_project"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "context_checkpoint.json"
        checkpoint_file.write_text(json.dumps({}), encoding="utf-8")
        assert checkpoint_file.exists()

        project._clear_checkpoint()

        assert not checkpoint_file.exists()

    def test_clear_checkpoint_no_error_if_not_exists(self, tmp_path):
        """Test that _clear_checkpoint doesn't error if file doesn't exist."""
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)

        # Should not raise
        project._clear_checkpoint()


@pytest.mark.unit
class TestBuildResume:
    """Test build() resume parameter behavior (mocked context generation)."""

    async def test_build_resume_false_ignores_checkpoint(self, tmp_path):
        """Test that resume=False ignores existing checkpoint."""
        from konte.project import Project
        from konte.models import Chunk
        import json

        project = Project.create(name="test_project", storage_path=tmp_path)

        # Add a chunk manually
        chunk = Chunk(
            chunk_id="id1",
            content="Test content",
            source="doc.pdf",
            segment_idx=0,
            chunk_idx=0,
        )
        project._chunks = [chunk]
        project._segments = {("doc.pdf", 0): "Full segment text"}

        # Create a checkpoint file
        checkpoint_dir = tmp_path / "test_project"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "context_checkpoint.json"
        checkpoint_data = {
            "completed_segments": ["doc.pdf|0"],
            "contextualized_chunks": [
                {
                    "chunk": chunk.model_dump(),
                    "context": "old context",
                }
            ],
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data), encoding="utf-8")

        # Mock context generation
        with patch("konte.project.generate_contexts_batch") as mock_gen:
            from konte.models import ContextualizedChunk

            mock_gen.return_value = [
                ContextualizedChunk(chunk=chunk, context="new context")
            ]

            await project.build(skip_context=True, enable_bm25=False, resume=False)

            # Should have called generate_contexts_batch (not skipped)
            assert mock_gen.called

    async def test_build_resume_true_uses_checkpoint(self, tmp_path):
        """Test that resume=True uses existing checkpoint."""
        from konte.project import Project
        from konte.models import Chunk
        import json

        project = Project.create(name="test_project", storage_path=tmp_path)

        # Add chunks manually
        chunk1 = Chunk(
            chunk_id="id1", content="Content 1", source="doc.pdf",
            segment_idx=0, chunk_idx=0,
        )
        chunk2 = Chunk(
            chunk_id="id2", content="Content 2", source="doc.pdf",
            segment_idx=1, chunk_idx=0,
        )
        project._chunks = [chunk1, chunk2]
        project._segments = {
            ("doc.pdf", 0): "Segment 0 text",
            ("doc.pdf", 1): "Segment 1 text",
        }

        # Create a checkpoint with segment 0 completed
        checkpoint_dir = tmp_path / "test_project"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "context_checkpoint.json"
        checkpoint_data = {
            "completed_segments": ["doc.pdf|0"],
            "contextualized_chunks": [
                {"chunk": chunk1.model_dump(), "context": "ctx1"},
            ],
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data), encoding="utf-8")

        # Mock context generation
        with patch("konte.project.generate_contexts_batch") as mock_gen:
            from konte.models import ContextualizedChunk

            # Only return for segment 1 (segment 0 is checkpointed)
            mock_gen.return_value = [
                ContextualizedChunk(chunk=chunk2, context="ctx2")
            ]

            await project.build(skip_context=True, enable_bm25=False, resume=True)

            # Should have called once (only for segment 1)
            assert mock_gen.call_count == 1
            # Should have 2 contextualized chunks total
            assert len(project._contextualized_chunks) == 2
