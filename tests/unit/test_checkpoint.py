"""Unit tests for build checkpoint functionality."""

import json
from itertools import pairwise
from unittest.mock import patch

import pytest

from konte.checkpoint import CHECKPOINT_FILENAME, LEGACY_CHECKPOINT_FILENAME, CheckpointLog


def _record(segment: str, chunk_id: str) -> dict:
    """Build the storage payload one finished segment would append."""
    return {
        "segments": [segment],
        "chunks": [{"chunk": {"chunk_id": chunk_id}, "context": f"ctx-{chunk_id}"}],
    }


@pytest.mark.unit
class TestCheckpointPath:
    """Test where the log lives."""

    def test_log_path_is_inside_project_dir(self, tmp_path):
        """Test that the log sits next to the project's other artifacts."""
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)

        assert project._checkpoint.path == tmp_path / "test_project" / CHECKPOINT_FILENAME


@pytest.mark.unit
class TestReadCheckpoint:
    """Test CheckpointLog.read."""

    def test_read_returns_none_if_not_exists(self, tmp_path):
        """Test that read returns None when nothing was ever written."""
        assert CheckpointLog(tmp_path).read() is None

    def test_read_returns_none_for_empty_log(self, tmp_path):
        """Test that a log with no records reads as no checkpoint."""
        (tmp_path / CHECKPOINT_FILENAME).write_text("", encoding="utf-8")

        assert CheckpointLog(tmp_path).read() is None

    def test_read_concatenates_records(self, tmp_path):
        """Test that every appended segment comes back in one checkpoint."""
        log = CheckpointLog(tmp_path)
        with log.appending() as appender:
            appender.append("doc.pdf|0", [{"chunk": {"chunk_id": "id1"}, "context": "ctx1"}])
            appender.append("doc.pdf|1", [{"chunk": {"chunk_id": "id2"}, "context": "ctx2"}])

        checkpoint = log.read()

        assert checkpoint is not None
        assert checkpoint.completed_segments == ["doc.pdf|0", "doc.pdf|1"]
        assert len(checkpoint.contextualized_chunks) == 2

    def test_read_skips_record_without_segments(self, tmp_path):
        """Test that a hand-edited record missing its key is skipped, not fatal."""
        path = tmp_path / CHECKPOINT_FILENAME
        path.write_text(
            json.dumps({"chunks": []}) + "\n" + json.dumps(_record("doc.pdf|1", "id2")) + "\n",
            encoding="utf-8",
        )

        checkpoint = CheckpointLog(tmp_path).read()

        assert checkpoint is not None
        assert checkpoint.completed_segments == ["doc.pdf|1"]


@pytest.mark.unit
class TestTornTailRecovery:
    """Test that a build killed mid-write leaves a usable checkpoint."""

    def test_partial_last_line_is_dropped(self, tmp_path):
        """Test that a record cut off mid-write does not break the resume."""
        path = tmp_path / CHECKPOINT_FILENAME
        intact = json.dumps(_record("doc.pdf|0", "id1")) + "\n"
        path.write_text(intact + '{"segments": ["doc.pdf|1"], "chu', encoding="utf-8")

        checkpoint = CheckpointLog(tmp_path).read()

        assert checkpoint is not None
        assert checkpoint.completed_segments == ["doc.pdf|0"]

    def test_partial_last_line_is_truncated_away(self, tmp_path):
        """Test that the torn tail is removed, so the next append starts clean."""
        path = tmp_path / CHECKPOINT_FILENAME
        intact = json.dumps(_record("doc.pdf|0", "id1")) + "\n"
        path.write_text(intact + '{"segments": ["doc.pdf|1"], "chu', encoding="utf-8")

        log = CheckpointLog(tmp_path)
        log.read()
        with log.appending() as appender:
            appender.append("doc.pdf|1", [{"chunk": {"chunk_id": "id2"}, "context": "ctx2"}])

        checkpoint = log.read()
        assert checkpoint is not None
        assert checkpoint.completed_segments == ["doc.pdf|0", "doc.pdf|1"]

    def test_terminated_but_invalid_line_is_dropped(self, tmp_path):
        """Test that a newline surviving a half-written record is not trusted."""
        path = tmp_path / CHECKPOINT_FILENAME
        intact = json.dumps(_record("doc.pdf|0", "id1")) + "\n"
        path.write_text(intact + '{"segments": ["doc\n', encoding="utf-8")

        checkpoint = CheckpointLog(tmp_path).read()

        assert checkpoint is not None
        assert checkpoint.completed_segments == ["doc.pdf|0"]
        assert path.stat().st_size == len(intact.encode("utf-8"))


@pytest.mark.unit
class TestAppendOnly:
    """Test that the log never rewrites what it already holds."""

    def test_appending_never_rewrites_earlier_bytes(self, tmp_path):
        """Test that each segment adds its own line and touches nothing before it."""
        log = CheckpointLog(tmp_path)
        prefixes = []

        with log.appending() as appender:
            for index in range(5):
                appender.append(f"doc.pdf|{index}", [{"chunk": {"chunk_id": f"id{index}"}}])
                prefixes.append(log.path.read_bytes())

        # Every snapshot is a prefix of the next: growth only, no rewriting.
        for earlier, later in pairwise(prefixes):
            assert later.startswith(earlier)

    def test_total_bytes_grow_linearly_with_segments(self, tmp_path):
        """Test that ten times the segments costs ten times the bytes, not a hundred."""
        chunks = [{"chunk": {"chunk_id": "x", "content": "y" * 500}, "context": "z" * 500}]

        sizes = {}
        for count in (10, 100):
            path = tmp_path / str(count)
            path.mkdir()
            log = CheckpointLog(path)
            with log.appending() as appender:
                for index in range(count):
                    appender.append(f"doc.pdf|{index}", chunks)
            sizes[count] = log.path.stat().st_size

        ratio = sizes[100] / sizes[10]
        assert 9 < ratio < 11  # quadratic rewriting would land near 100


@pytest.mark.unit
class TestLegacyCheckpoint:
    """Test that a checkpoint from the whole-file format still resumes."""

    def test_legacy_file_is_read(self, tmp_path):
        """Test that an interrupted build from an older version is picked up."""
        (tmp_path / LEGACY_CHECKPOINT_FILENAME).write_text(
            json.dumps(
                {
                    "completed_segments": ["doc.pdf|0", "doc.pdf|1"],
                    "contextualized_chunks": [{"chunk": {"chunk_id": "id1"}, "context": "ctx1"}],
                }
            ),
            encoding="utf-8",
        )

        checkpoint = CheckpointLog(tmp_path).read()

        assert checkpoint is not None
        assert checkpoint.completed_segments == ["doc.pdf|0", "doc.pdf|1"]
        assert len(checkpoint.contextualized_chunks) == 1

    def test_legacy_file_is_converted_and_removed(self, tmp_path):
        """Test that reading a legacy file migrates it, so a second resume works."""
        (tmp_path / LEGACY_CHECKPOINT_FILENAME).write_text(
            json.dumps(
                {
                    "completed_segments": ["doc.pdf|0"],
                    "contextualized_chunks": [{"chunk": {"chunk_id": "id1"}, "context": "ctx1"}],
                }
            ),
            encoding="utf-8",
        )

        log = CheckpointLog(tmp_path)
        log.read()

        assert not (tmp_path / LEGACY_CHECKPOINT_FILENAME).exists()
        assert log.path.exists()

        second = log.read()
        assert second is not None
        assert second.completed_segments == ["doc.pdf|0"]

    def test_truncated_legacy_file_is_discarded(self, tmp_path):
        """Test that the accident the checkpoint existed to prevent is survivable."""
        (tmp_path / LEGACY_CHECKPOINT_FILENAME).write_text(
            '{"completed_segments": ["doc.pdf|0"], "contextualized_ch',
            encoding="utf-8",
        )

        log = CheckpointLog(tmp_path)

        assert log.read() is None
        assert not (tmp_path / LEGACY_CHECKPOINT_FILENAME).exists()


@pytest.mark.unit
class TestClearCheckpoint:
    """Test CheckpointLog.clear."""

    def test_clear_removes_both_formats(self, tmp_path):
        """Test that clearing leaves no checkpoint of either format behind."""
        (tmp_path / CHECKPOINT_FILENAME).write_text("", encoding="utf-8")
        (tmp_path / LEGACY_CHECKPOINT_FILENAME).write_text("{}", encoding="utf-8")

        CheckpointLog(tmp_path).clear()

        assert not (tmp_path / CHECKPOINT_FILENAME).exists()
        assert not (tmp_path / LEGACY_CHECKPOINT_FILENAME).exists()

    def test_clear_no_error_if_not_exists(self, tmp_path):
        """Test that clearing an absent checkpoint does not error."""
        CheckpointLog(tmp_path).clear()


@pytest.mark.unit
class TestBuildResume:
    """Test build() resume parameter behavior (mocked context generation)."""

    async def test_build_resume_false_ignores_checkpoint(self, tmp_path):
        """Test that resume=False ignores existing checkpoint."""
        from konte.models import Chunk
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)

        chunk = Chunk(
            chunk_id="id1",
            content="Test content",
            source="doc.pdf",
            segment_idx=0,
            chunk_idx=0,
        )
        project._chunks = [chunk]
        project._segments = {("doc.pdf", 0): "Full segment text"}

        with project._checkpoint.appending() as appender:
            appender.append("doc.pdf|0", [{"chunk": chunk.model_dump(), "context": "old context"}])

        with patch("konte.project.generate_contexts_batch") as mock_gen:
            from konte.context import ContextBatch
            from konte.models import ContextualizedChunk

            mock_gen.return_value = ContextBatch(
                [ContextualizedChunk(chunk=chunk, context="new context")], 0, 0
            )

            # BM25-only: FAISS would construct a real OpenAIEmbeddings client
            await project.build(skip_context=True, enable_faiss=False, resume=False)

            assert mock_gen.called
            assert len(project._contextualized_chunks) == 1

    async def test_build_resume_true_uses_checkpoint(self, tmp_path):
        """Test that resume=True uses existing checkpoint."""
        from konte.models import Chunk
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)

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

        with project._checkpoint.appending() as appender:
            appender.append("doc.pdf|0", [{"chunk": chunk1.model_dump(), "context": "ctx1"}])

        with patch("konte.project.generate_contexts_batch") as mock_gen:
            from konte.context import ContextBatch
            from konte.models import ContextualizedChunk

            mock_gen.return_value = ContextBatch(
                [ContextualizedChunk(chunk=chunk2, context="ctx2")], 0, 0
            )

            # BM25-only: FAISS would construct a real OpenAIEmbeddings client
            await project.build(skip_context=True, enable_faiss=False, resume=True)

            assert mock_gen.call_count == 1
            assert len(project._contextualized_chunks) == 2

    async def test_build_clears_checkpoint_on_success(self, tmp_path):
        """Test that a finished build leaves no checkpoint to resume from."""
        from konte.models import Chunk, ContextualizedChunk
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)
        chunk = Chunk(
            chunk_id="id1", content="Content", source="doc.pdf", segment_idx=0, chunk_idx=0
        )
        project._chunks = [chunk]
        project._segments = {("doc.pdf", 0): "Segment text"}

        with patch("konte.project.generate_contexts_batch") as mock_gen:
            from konte.context import ContextBatch

            mock_gen.return_value = ContextBatch(
                [ContextualizedChunk(chunk=chunk, context="")], 0, 0
            )
            await project.build(skip_context=True, enable_faiss=False)

        assert not project._checkpoint.path.exists()


@pytest.mark.unit
class TestContextCoverage:
    """Test that a build which lost contexts says so instead of indexing anyway."""

    def _project(self, tmp_path, segments):
        """Build a project of `segments` one-chunk segments, ready to build."""
        from konte.models import Chunk
        from konte.project import Project

        project = Project.create(name="test_project", storage_path=tmp_path)
        project._chunks = [
            Chunk(
                chunk_id=f"id{i}",
                content=f"Content {i}",
                source="doc.pdf",
                segment_idx=i,
                chunk_idx=0,
            )
            for i in range(segments)
        ]
        project._segments = {("doc.pdf", i): f"Segment {i}" for i in range(segments)}
        return project

    def _batches(self, project, failed_segments):
        """Answer each segment's request, failing the ones named."""
        from konte.context import ContextBatch
        from konte.models import ContextualizedChunk

        def batch(segment, chunks, **kwargs):
            failed = chunks[0].segment_idx in failed_segments
            return ContextBatch(
                [ContextualizedChunk(chunk=c, context="" if failed else "ctx") for c in chunks],
                len(chunks) if failed else 0,
                0,
            )

        return patch("konte.project.generate_contexts_batch", side_effect=batch)

    async def test_losing_most_contexts_fails_the_build(self, tmp_path):
        """Test a corpus mostly without context is reported, not indexed."""
        project = self._project(tmp_path, segments=4)

        with (
            self._batches(project, failed_segments={0, 1, 2}),
            pytest.raises(RuntimeError, match="no generated context"),
        ):
            await project.build(enable_faiss=False)

        assert project._bm25 is None

    async def test_a_loss_under_the_threshold_still_builds(self, tmp_path):
        """Test one unlucky chunk does not throw away an otherwise good corpus."""
        from konte.config import settings

        project = self._project(tmp_path, segments=100)

        with patch.object(settings, "CONTEXT_FAILURE_THRESHOLD", 0.05):
            with self._batches(project, failed_segments={7}):
                await project.build(enable_faiss=False)

        assert project._bm25 is not None
        assert len(project._contextualized_chunks) == 100

    async def test_a_failed_segment_is_not_checkpointed(self, tmp_path):
        """Test a resumed build retries the segments that lost their contexts.

        Checkpointing them would retire those chunks context-free, and the
        resumed build would count no failures and report success.
        """
        project = self._project(tmp_path, segments=4)

        with self._batches(project, failed_segments={0, 1, 2}), pytest.raises(RuntimeError):
            await project.build(enable_faiss=False)

        checkpoint = project._checkpoint.read()
        assert checkpoint.completed_segments == ["doc.pdf|3"]

    async def test_resuming_retries_only_what_failed(self, tmp_path):
        """Test the retry costs one request per lost segment, not per segment."""
        project = self._project(tmp_path, segments=4)

        with self._batches(project, failed_segments={0, 1, 2}), pytest.raises(RuntimeError):
            await project.build(enable_faiss=False)

        with self._batches(project, failed_segments=set()) as retry:
            await project.build(enable_faiss=False, resume=True)

        assert retry.call_count == 3
        assert len(project._contextualized_chunks) == 4
        assert all(c.context == "ctx" for c in project._contextualized_chunks)

    async def test_skip_context_is_not_read_as_total_loss(self, tmp_path):
        """Test standard-RAG mode still builds, though every context is empty."""
        project = self._project(tmp_path, segments=4)

        await project.build(skip_context=True, enable_faiss=False)

        assert project._bm25 is not None
