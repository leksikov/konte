"""The build pipeline: contextualize a project's chunks, then index them."""

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import structlog

from konte.checkpoint import CheckpointLog
from konte.config import settings
from konte.context import ContextBatch, generate_contexts_batch, load_prompt_template
from konte.models import (
    Chunk,
    ContextualizedChunk,
    ProjectConfig,
    SegmentKey,
    encode_segment_key,
)
from konte.stores import BM25Store, FAISSStore

logger = structlog.get_logger()


class _ContextTally(NamedTuple):
    """How many of a build's chunks lost or shortened their context."""

    failed: int
    truncated: int


class BuiltIndexes(NamedTuple):
    """The indexes one build produced; None where it was asked to skip one."""

    faiss: FAISSStore | None
    bm25: BM25Store | None


def _group_by_segment(chunks: list[Chunk]) -> dict[SegmentKey, list[Chunk]]:
    """Group chunks by the segment they were cut from, preserving order."""
    grouped: defaultdict[SegmentKey, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[(chunk.source, chunk.segment_idx)].append(chunk)
    return dict(grouped)


class ProjectBuilder:
    """One build: a context for every chunk, then the indexes over them.

    What it contextualized stays readable as `chunks` after the run raises, so
    a caller can keep the work a rejected build had already paid for.

    Args:
        config: The project's configuration.
        checkpoint: Log this build resumes from and appends to.
        enable_faiss: Build the semantic index. None follows the config.
        enable_bm25: Build the lexical index. None follows the config.

    Raises:
        ValueError: If both indexes are disabled.
    """

    __slots__ = ("_checkpoint", "_config", "_use_bm25", "_use_faiss", "chunks")

    def __init__(
        self,
        config: ProjectConfig,
        checkpoint: CheckpointLog,
        *,
        enable_faiss: bool | None = None,
        enable_bm25: bool | None = None,
    ) -> None:
        self._config = config
        self._checkpoint = checkpoint
        self._use_faiss = enable_faiss if enable_faiss is not None else config.enable_faiss
        self._use_bm25 = enable_bm25 if enable_bm25 is not None else config.enable_bm25

        if not self._use_faiss and not self._use_bm25:
            raise ValueError("At least one index (FAISS or BM25) must be enabled")

        self.chunks: list[ContextualizedChunk] = []

    async def run(
        self,
        chunks: list[Chunk],
        segments: dict[SegmentKey, str],
        *,
        skip_context: bool = False,
        resume: bool = True,
        prompt_path: Path | None = None,
    ) -> BuiltIndexes:
        """Contextualize the corpus and index it.

        Args:
            chunks: The raw corpus to build from.
            segments: Segment texts each chunk's context is written against.
            skip_context: If True, skip LLM context generation (standard RAG).
            resume: If True, resume from checkpoint if exists.
            prompt_path: Custom context prompt. Priority: this argument >
                config.context_prompt_path > settings.PROMPT_PATH.

        Raises:
            RuntimeError: If more than settings.CONTEXT_FAILURE_THRESHOLD of the
                corpus lost its context. Nothing is indexed; re-running retries
                only the segments that failed.
        """
        chunks_by_segment = _group_by_segment(chunks)
        completed = self._restore_checkpoint(resume, len(chunks_by_segment))
        prompt_template = (
            None
            if skip_context
            else load_prompt_template(prompt_path or self._config.context_prompt_path)
        )

        tally = await self._generate_contexts(
            chunks_by_segment, segments, completed, prompt_template, skip_context
        )
        # Before the indexes: a rejected corpus should cost no embedding calls.
        self._require_context_coverage(tally)
        return await self._build_indexes()

    def _restore_checkpoint(self, resume: bool, total_segments: int) -> set[str]:
        """Pick up an interrupted build, or start a fresh one.

        Args:
            resume: If False, any checkpoint on disk is ignored.
            total_segments: Segment count, reported in the resume log line.

        Returns:
            The segment keys already contextualized.
        """
        self.chunks = []

        checkpoint = self._checkpoint.read() if resume else None
        if checkpoint is None:
            # Appending to what an earlier build left would fold two runs into one.
            self._checkpoint.clear()
            return set()

        self.chunks = [
            ContextualizedChunk.from_storage_dict(item) for item in checkpoint.contextualized_chunks
        ]
        completed = set(checkpoint.completed_segments)
        logger.info(
            "checkpoint_resumed",
            completed_segments=len(completed),
            total_segments=total_segments,
        )
        return completed

    async def _generate_contexts(
        self,
        chunks_by_segment: dict[SegmentKey, list[Chunk]],
        segments: dict[SegmentKey, str],
        completed: set[str],
        prompt_template: str | None,
        skip_context: bool,
    ) -> _ContextTally:
        """Contextualize every not-yet-processed segment, concurrently.

        Each segment appends itself to the checkpoint log as it finishes, so an
        interrupted run resumes at a segment boundary. A segment that lost a
        context is left out of it: checkpointing one would retire those chunks
        context-free, and no later run would ask for them again.

        Returns:
            What this run's chunks lost, across every segment it processed.
        """
        total_segments = len(chunks_by_segment)
        logger.info(
            "context_generation_started",
            total_segments=total_segments,
            skip_context=skip_context,
        )

        pending: list[tuple[SegmentKey, str]] = []
        for seg_key in chunks_by_segment:
            seg_key_str = encode_segment_key(seg_key)
            if seg_key_str in completed:
                logger.info("segment_skipped", segment_key=seg_key_str, reason="checkpoint")
                continue
            pending.append((seg_key, seg_key_str))

        segment_gate = asyncio.Semaphore(settings.concurrency_limit)

        async def contextualize(
            seg_key: SegmentKey,
            seg_key_str: str,
        ) -> tuple[str, ContextBatch]:
            """Run one segment, tagged with the key it belongs to."""
            segment_chunks = chunks_by_segment[seg_key]
            async with segment_gate:
                logger.info(
                    "generating_context_for_segment",
                    segment_key=seg_key_str,
                    total_segments=total_segments,
                    num_chunks=len(segment_chunks),
                )
                return seg_key_str, await generate_contexts_batch(
                    segment=segments.get(seg_key, ""),
                    chunks=segment_chunks,
                    model=self._config.context_model,
                    prompt_template=prompt_template,
                    skip_context=skip_context,
                )

        produced: dict[str, list[ContextualizedChunk]] = {}
        failed = truncated = 0

        with self._checkpoint.appending() as log:
            tasks = [asyncio.ensure_future(contextualize(*item)) for item in pending]
            try:
                for finished in asyncio.as_completed(tasks):
                    seg_key_str, batch = await finished
                    produced[seg_key_str] = batch.chunks
                    failed += batch.failed
                    truncated += batch.truncated

                    if batch.failed:
                        logger.error(
                            "segment_contexts_incomplete",
                            segment_key=seg_key_str,
                            failed=batch.failed,
                            num_chunks=len(batch.chunks),
                        )
                        continue

                    log.append(seg_key_str, [c.to_storage_dict() for c in batch.chunks])
                    completed.add(seg_key_str)
                    logger.info(
                        "checkpoint_saved",
                        segment_key=seg_key_str,
                        completed=len(completed),
                        total=total_segments,
                    )
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        # Segment order, not the order the responses arrived.
        self.chunks.extend(chunk for _, seg_key_str in pending for chunk in produced[seg_key_str])

        logger.info(
            "context_generation_complete",
            num_chunks=len(self.chunks),
            skipped=skip_context,
            failed=failed,
            truncated=truncated,
        )
        return _ContextTally(failed, truncated)

    def _require_context_coverage(self, tally: _ContextTally) -> None:
        """Refuse to index a corpus that lost too much of its context.

        Nothing downstream can detect the loss: the index looks complete either
        way. Truncated contexts are reported but tolerated, being degraded
        rather than absent.

        Raises:
            RuntimeError: If the failed share exceeds
                settings.CONTEXT_FAILURE_THRESHOLD.
        """
        total = len(self.chunks)
        if tally.truncated:
            logger.warning(
                "context_truncated_total",
                truncated=tally.truncated,
                total_chunks=total,
            )
        if not tally.failed or not total:
            return

        share = tally.failed / total
        logger.error(
            "context_generation_incomplete",
            failed=tally.failed,
            total_chunks=total,
            share=round(share, 4),
            threshold=settings.CONTEXT_FAILURE_THRESHOLD,
        )
        if share <= settings.CONTEXT_FAILURE_THRESHOLD:
            return

        raise RuntimeError(
            f"{tally.failed} of {total} chunks ({share:.1%}) came back with no "
            f"generated context, over the {settings.CONTEXT_FAILURE_THRESHOLD:.1%} "
            "allowed by CONTEXT_FAILURE_THRESHOLD. Indexed, they would be "
            "searchable only by their own wording, so nothing was indexed. "
            "Check the context endpoint and run build() again to retry just "
            "the segments that failed, or raise CONTEXT_FAILURE_THRESHOLD to "
            "accept the loss."
        )

    async def _build_indexes(self) -> BuiltIndexes:
        """Build the enabled indexes over what was contextualized."""
        faiss: FAISSStore | None = None
        bm25: BM25Store | None = None

        if self._use_faiss:
            faiss = FAISSStore(embedding_model=self._config.embedding_model)
            await faiss.abuild_index(self.chunks)
            logger.info("faiss_index_built")

        if self._use_bm25:
            bm25 = BM25Store()
            bm25.build_index(self.chunks)
            logger.info("bm25_index_built")

        return BuiltIndexes(faiss, bm25)
