"""Append-only record of the segments a build has already contextualized."""

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import structlog

from konte.models import BuildCheckpoint
from konte.storage import atomic_writer, read_jsonl

logger = structlog.get_logger()

CHECKPOINT_FILENAME = "context_checkpoint.jsonl"
LEGACY_CHECKPOINT_FILENAME = "context_checkpoint.json"


def _encode_record(segments: Sequence[str], chunks: Sequence[dict[str, Any]]) -> str:
    """Serialize one checkpoint record as a single newline-terminated line."""
    record = {"segments": list(segments), "chunks": list(chunks)}
    return json.dumps(record, ensure_ascii=False) + "\n"


class _SegmentAppender:
    """Writes one line per finished segment to an already-open checkpoint log."""

    __slots__ = ("_handle",)

    def __init__(self, handle: IO[str]) -> None:
        self._handle = handle

    def append(self, segment: str, chunks: Sequence[dict[str, Any]]) -> None:
        """Record one finished segment, flushed and synced before returning.

        Args:
            segment: Encoded segment key that just completed.
            chunks: Storage dicts for the chunks that segment produced.
        """
        self._handle.write(_encode_record([segment], chunks))
        self._handle.flush()
        os.fsync(self._handle.fileno())


class CheckpointLog:
    """The segments a build has finished, as one line per segment.

    A segment appends its own line and nothing already written is touched
    again: rewriting the accumulated state instead would store every chunk
    once per remaining segment, which is quadratic in the segment count.
    """

    __slots__ = ("_legacy_path", "_path")

    def __init__(self, directory: Path) -> None:
        """Bind the log to a project directory.

        Args:
            directory: Project directory holding the checkpoint files.
        """
        self._path = directory / CHECKPOINT_FILENAME
        self._legacy_path = directory / LEGACY_CHECKPOINT_FILENAME

    @property
    def path(self) -> Path:
        """Path: The log this build appends to."""
        return self._path

    def read(self) -> BuildCheckpoint | None:
        """Return what an earlier build recorded, or None when there is nothing.

        A torn trailing line is dropped and truncated away. A legacy whole-file
        checkpoint is converted to the current format first; one that was
        itself torn is discarded and the build starts over.

        Returns:
            The recorded state, or None when no usable checkpoint exists.
        """
        if self._path.exists():
            return self._read_log()
        if self._legacy_path.exists():
            return self._read_legacy()
        return None

    def clear(self) -> None:
        """Remove every checkpoint file, current and legacy."""
        self._path.unlink(missing_ok=True)
        self._legacy_path.unlink(missing_ok=True)

    @contextmanager
    def appending(self) -> Iterator[_SegmentAppender]:
        """Hold the log open for the duration of one build.

        Yields:
            The appender segments report themselves to.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            yield _SegmentAppender(handle)

    def _read_log(self) -> BuildCheckpoint | None:
        """Fold the log's records back into one checkpoint."""
        records, intact_bytes = read_jsonl(self._path)
        self._discard_torn_tail(intact_bytes)

        checkpoint = BuildCheckpoint()
        for record in records:
            if not isinstance(record, dict) or "segments" not in record:
                logger.warning("checkpoint_record_skipped", path=str(self._path))
                continue
            checkpoint.completed_segments.extend(record["segments"])
            checkpoint.contextualized_chunks.extend(record.get("chunks", []))

        return checkpoint if checkpoint.completed_segments else None

    def _discard_torn_tail(self, intact_bytes: int) -> None:
        """Cut a half-written trailing line, so the next append starts clean."""
        size = self._path.stat().st_size
        if size == intact_bytes:
            return

        logger.warning(
            "checkpoint_tail_discarded",
            path=str(self._path),
            dropped_bytes=size - intact_bytes,
        )
        os.truncate(self._path, intact_bytes)

    def _read_legacy(self) -> BuildCheckpoint | None:
        """Adopt a checkpoint left by the earlier whole-file format."""
        try:
            checkpoint = BuildCheckpoint(
                **json.loads(self._legacy_path.read_text(encoding="utf-8"))
            )
        except (TypeError, ValueError) as error:
            logger.warning(
                "checkpoint_legacy_unreadable",
                path=str(self._legacy_path),
                error=str(error),
            )
            self._legacy_path.unlink(missing_ok=True)
            return None

        self._convert_legacy(checkpoint)
        logger.info(
            "checkpoint_legacy_converted",
            path=str(self._path),
            completed_segments=len(checkpoint.completed_segments),
        )
        return checkpoint

    def _convert_legacy(self, checkpoint: BuildCheckpoint) -> None:
        """Rewrite a legacy checkpoint as the log's first record, then drop it.

        The legacy format mapped no chunk to its segment, so all of it becomes
        one record covering every segment it had completed.
        """
        with atomic_writer(self._path, encoding="utf-8") as handle:
            handle.write(
                _encode_record(
                    checkpoint.completed_segments,
                    checkpoint.contextualized_chunks,
                )
            )
        self._legacy_path.unlink(missing_ok=True)
