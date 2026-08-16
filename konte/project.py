"""Project class - main interface for contextual RAG."""

import asyncio
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Generic, NamedTuple, TypeVar

import structlog

from konte.checkpoint import CheckpointLog
from konte.chunker import create_chunks
from konte.config import settings
from konte.context import ContextBatch, generate_contexts_batch, load_prompt_template
from konte.generator import GeneratedAnswer, generate_answer
from konte.loader import load_document
from konte.models import (
    Chunk,
    ContextualizedChunk,
    MetadataFilter,
    ProjectConfig,
    RetrievalMode,
    RetrievalResponse,
    SegmentKey,
)
from konte.storage import read_json, write_json
from konte.stores import BM25Store, FAISSStore, Retriever
from konte.stores.bm25_store import INDEX_FILENAME as BM25_INDEX_FILENAME
from konte.stores.bm25_store import LEGACY_INDEX_FILENAME as LEGACY_BM25_INDEX_FILENAME
from konte.stores.faiss_store import INDEX_FILENAME as FAISS_INDEX_FILENAME

logger = structlog.get_logger()

_SEGMENT_KEY_SEPARATOR = "|"

_CONFIG_INDENT = 2  # config.json is the one artifact people open by hand

_T = TypeVar("_T")


class _ContextTally(NamedTuple):
    """How many of a build's chunks lost or shortened their context."""

    failed: int
    truncated: int


def _encode_segment_key(key: SegmentKey) -> str:
    """Flatten a (source, segment index) key into a JSON-safe string."""
    source, index = key
    return f"{source}{_SEGMENT_KEY_SEPARATOR}{index}"


def _decode_segment_key(raw: str) -> SegmentKey:
    """Parse a stored segment key.

    Keys with no separator were written by an older format that recorded only
    the segment index, so their source document is unrecoverable.
    """
    source, separator, index = raw.rpartition(_SEGMENT_KEY_SEPARATOR)
    if not separator:
        return ("unknown", int(raw))
    return (source, int(index))


def _group_by_segment(chunks: list[Chunk]) -> dict[SegmentKey, list[Chunk]]:
    """Group chunks by the segment they were cut from, preserving order."""
    grouped: defaultdict[SegmentKey, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[(chunk.source, chunk.segment_idx)].append(chunk)
    return dict(grouped)


def _pick_source(path: Path, taken: set[str]) -> str:
    """File a document under the shortest tail of its path no other one holds.

    Sharing a source name collides chunk ids and overwrites the segment map.
    The bare filename comes first: metadata filters are written against it.
    """
    resolved = path.resolve()
    parts = resolved.parts

    for depth in range(1, len(parts)):
        name = "/".join(parts[-depth:])
        if name not in taken:
            if depth > 1:
                logger.warning("document_source_disambiguated", path=str(resolved), source=name)
            return name

    name = str(resolved)
    suffix = 1
    while name in taken:
        suffix += 1
        name = f"{resolved}#{suffix}"
    return name


def _duplicate_source(
    stored: dict[SegmentKey, str],
    incoming: dict[SegmentKey, str],
    source: str,
) -> str | None:
    """The document already in the project holding exactly these segments.

    Segments cover the whole text, so matching all of another's is that
    document again — under a second name, just handed to it by _pick_source.
    """
    if not incoming:
        return None

    segment_counts: defaultdict[str, int] = defaultdict(int)
    for other, _ in stored:
        segment_counts[other] += 1

    size = len(incoming)
    for other, count in segment_counts.items():
        if count == size and all(
            stored[(other, index)] == incoming[(source, index)] for index in range(size)
        ):
            return other
    return None


def _parse_chunks(path: Path) -> list[Chunk]:
    """Rebuild the raw chunk list, empty when the artifact is absent."""
    data = read_json(path)
    return [] if data is None else [Chunk(**item) for item in data]


def _parse_segments(path: Path) -> dict[SegmentKey, str]:
    """Rebuild the segment texts, empty when the artifact is absent."""
    data = read_json(path)
    return {} if data is None else {_decode_segment_key(key): text for key, text in data.items()}


def _parse_contextualized(path: Path) -> list[ContextualizedChunk]:
    """Rebuild the contextualized chunks, empty when the artifact is absent."""
    data = read_json(path)
    return [] if data is None else [ContextualizedChunk.from_storage_dict(item) for item in data]


class _DeferredArtifact(Generic[_T]):
    """Instance attribute parsed from disk the first time it is read.

    Only building and saving read the corpus-sized artifacts, so a query-only
    caller never pays to rebuild them. A damaged artifact therefore surfaces
    at first use rather than at load().
    """

    __slots__ = ("_name",)

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: "Project | None", objtype: type | None = None) -> _T:
        if obj is None:
            return self  # type: ignore[return-value]
        try:
            return obj.__dict__[self._name]
        except KeyError:
            return obj._materialize(self._name)

    def __set__(self, obj: "Project", value: _T) -> None:
        obj.__dict__[self._name] = value
        obj._deferred.pop(self._name, None)


class Project:
    """Main interface for contextual RAG operations."""

    _chunks = _DeferredArtifact[list[Chunk]]()
    _segments = _DeferredArtifact[dict[SegmentKey, str]]()
    _contextualized_chunks = _DeferredArtifact[list[ContextualizedChunk]]()

    def __init__(self, config: ProjectConfig):
        """Initialize project with configuration.

        Args:
            config: Project configuration.
        """
        self._config = config
        self._deferred: dict[str, Callable[[], Any]] = {}
        self._chunks = []
        self._contextualized_chunks = []
        self._segments = {}
        self._checkpoint = CheckpointLog(self.project_dir)
        self._faiss: FAISSStore | None = None
        self._bm25: BM25Store | None = None
        self._retriever: Retriever | None = None

    @property
    def config(self) -> ProjectConfig:
        """Get project configuration."""
        return self._config

    @property
    def project_dir(self) -> Path:
        """Get project directory path."""
        return self._config.storage_path / self._config.name

    def add_documents(self, file_paths: Sequence[str | Path]) -> int:
        """Add documents to the project.

        Loads, segments, and chunks documents.

        A document is filed under its filename, or under as much of its path as
        it takes to be unique — two `report.md` would otherwise collide.

        Args:
            file_paths: Document paths to add.

        Returns:
            Number of chunks created.

        Raises:
            ValueError: If a document's text is already in the project; the
                ones ahead of it in `file_paths` are already added.
        """
        added = 0
        taken = {chunk.source for chunk in self._chunks}

        for raw_path in file_paths:
            file_path = Path(raw_path)
            logger.info("loading_document", path=str(file_path))

            source = _pick_source(file_path, taken)
            chunks, segments_map = create_chunks(
                text=load_document(file_path),
                source=source,
                segment_size=self._config.segment_size,
                segment_overlap=self._config.segment_overlap,
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )

            duplicate = _duplicate_source(self._segments, segments_map, source)
            if duplicate is not None:
                raise ValueError(
                    f"{file_path} holds the same text as '{duplicate}', already in "
                    f"project '{self._config.name}'. Indexing it twice would spend "
                    "two of every response's results on one passage."
                )

            taken.add(source)
            self._segments.update(segments_map)
            self._chunks.extend(chunks)
            added += len(chunks)

            logger.info(
                "document_chunked",
                path=str(file_path),
                source=source,
                num_chunks=len(chunks),
            )

        logger.info("documents_added", total_chunks=len(self._chunks))
        return added

    def set_metadata(
        self,
        metadata: dict[str, Any],
        source: str | None = None,
    ) -> int:
        """Set custom metadata on chunks (post-hoc, before build).

        Merges metadata into existing chunk metadata. Call after add_documents()
        and before build().

        Args:
            metadata: Key-value pairs to merge into chunk metadata.
            source: If provided, only apply to chunks from this source file.
                Substring match on chunk.source (e.g. "report" matches "annual_report.pdf").

        Returns:
            Number of chunks updated.
        """
        updated = 0
        for chunk in self._chunks:
            if source is not None and source not in chunk.source:
                continue
            chunk.metadata.update(metadata)
            updated += 1

        logger.info("metadata_set", updated=updated, keys=list(metadata.keys()), source=source)
        return updated

    async def build(
        self,
        skip_context: bool = False,
        enable_faiss: bool | None = None,
        enable_bm25: bool | None = None,
        resume: bool = True,
        prompt_path: Path | None = None,
    ) -> None:
        """Build indexes from added documents.

        Args:
            skip_context: If True, skip LLM context generation (standard RAG).
            enable_faiss: Enable FAISS index. Defaults to config setting.
            enable_bm25: Enable BM25 index. Defaults to config setting.
            resume: If True, resume from checkpoint if exists.
            prompt_path: Path to custom context prompt. Priority:
                prompt_path arg > config.context_prompt_path > settings.PROMPT_PATH.

        Raises:
            ValueError: If both indexes are disabled.
            RuntimeError: If more than settings.CONTEXT_FAILURE_THRESHOLD of the
                corpus lost its context. Nothing is indexed; re-running retries
                only the segments that failed.
        """
        if not self._chunks:
            logger.warning("build_no_chunks")
            return

        use_faiss = enable_faiss if enable_faiss is not None else self._config.enable_faiss
        use_bm25 = enable_bm25 if enable_bm25 is not None else self._config.enable_bm25
        if not use_faiss and not use_bm25:
            raise ValueError("At least one index (FAISS or BM25) must be enabled")

        chunks_by_segment = _group_by_segment(self._chunks)
        completed = self._restore_checkpoint(resume, len(chunks_by_segment))
        prompt_template = (
            None
            if skip_context
            else load_prompt_template(prompt_path or self._config.context_prompt_path)
        )

        tally = await self._generate_contexts(
            chunks_by_segment, completed, prompt_template, skip_context
        )
        # Before the indexes: a rejected corpus should cost no embedding calls.
        self._require_context_coverage(tally)
        await self._build_indexes(use_faiss=use_faiss, use_bm25=use_bm25)

        self._checkpoint.clear()
        logger.info("checkpoint_cleared")
        logger.info("project_build_complete")

    def query(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        use_keyword_extraction: bool | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
        inject_evidence: str | None = None,
        inject_position: int | None = None,
    ) -> RetrievalResponse:
        """Query the project (sync, no reranking).

        With keyword extraction on, lexical and hybrid modes spend one blocking
        LLM call per distinct query; pass False for a purely in-memory lookup.

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            use_keyword_extraction: Extract keywords before BM25 search, at the
                cost of one LLM call. None follows settings.BM25_KEYWORD_EXTRACTION.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "company": "ACME", "year": 2024}
            source_filter: Substring match on chunk source field.
                Example: "JOHNSON" matches "JOHNSON_JOHNSON_2022_10K.md"
            inject_evidence: For ablation study - inject this text.
            inject_position: Position to inject (0=top, None=random).

        Returns:
            RetrievalResponse with results, empty if the project was never built.
        """
        if self._retriever is None:
            logger.warning("query_no_retriever")
            return RetrievalResponse.empty(query)

        return self._retriever.retrieve(
            query,
            mode=mode,
            top_k=top_k or settings.DEFAULT_TOP_K,
            use_keyword_extraction=use_keyword_extraction,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
            inject_evidence=inject_evidence,
            inject_position=inject_position,
        )

    async def query_async(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        rerank: bool = False,
        rerank_initial_k: int = 50,
        use_keyword_extraction: bool | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResponse:
        """Query the project (async, with optional reranking).

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            rerank: If True, rerank via the configured RERANKER_BASE_URL endpoint.
            rerank_initial_k: Number of candidates to retrieve before reranking.
            use_keyword_extraction: Extract keywords before BM25 search, at the
                cost of one LLM call. None follows
                settings.BM25_KEYWORD_EXTRACTION, except under rerank=True with
                mode="hybrid", where BM25 has always seen the raw query.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "company": "ACME", "year": 2024}
            source_filter: Substring match on chunk source field.

        Returns:
            RetrievalResponse with results, empty if the project was never built.
        """
        if self._retriever is None:
            logger.warning("query_no_retriever")
            return RetrievalResponse.empty(query)

        k = top_k or settings.DEFAULT_TOP_K

        if rerank:
            return await self._retriever.retrieve_with_rerank(
                query,
                mode=mode,
                top_k=k,
                initial_k=rerank_initial_k,
                use_keyword_extraction=use_keyword_extraction,
                metadata_filter=metadata_filter,
                source_filter=source_filter,
            )

        return await self._retriever.retrieve_async(
            query,
            mode=mode,
            top_k=k,
            use_keyword_extraction=use_keyword_extraction,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )

    def as_retriever(self) -> Callable[[str], RetrievalResponse]:
        """Return a callable retriever for Agno integration.

        Returns:
            Callable that takes a query and returns RetrievalResponse.
        """
        return self.query

    async def query_with_answer(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        max_chunks: int = 10,
        prompt_template: str | None = None,
        timeout: float = 60.0,
        rerank: bool = False,
        rerank_initial_k: int = 50,
        use_keyword_extraction: bool | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> tuple[RetrievalResponse, GeneratedAnswer]:
        """Query the project and generate an LLM answer from retrieved chunks.

        This is the full RAG pipeline: retrieval + answer generation.

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results to retrieve. Defaults to settings.DEFAULT_TOP_K.
            max_chunks: Maximum chunks to use for answer generation.
            prompt_template: Custom prompt template with {context} and {question} placeholders.
            timeout: LLM request timeout in seconds.
            rerank: If True, rerank via the configured RERANKER_BASE_URL endpoint.
            rerank_initial_k: Number of candidates to retrieve before reranking.
            use_keyword_extraction: Extract keywords before BM25 search, at the
                cost of one LLM call on top of answer generation. None follows
                settings.BM25_KEYWORD_EXTRACTION.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "company": "ACME", "year": 2024}
            source_filter: Substring match on chunk source field.

        Returns:
            Tuple of (RetrievalResponse, GeneratedAnswer).
        """
        retrieval_response = await self.query_async(
            query,
            mode=mode,
            top_k=top_k,
            rerank=rerank,
            rerank_initial_k=rerank_initial_k,
            use_keyword_extraction=use_keyword_extraction,
            metadata_filter=metadata_filter,
            source_filter=source_filter,
        )

        answer = await generate_answer(
            question=query,
            retrieval_response=retrieval_response,
            prompt_template=prompt_template,
            max_chunks=max_chunks,
            timeout=timeout,
        )

        logger.info(
            "query_with_answer_complete",
            query=query[:50],
            chunks_retrieved=len(retrieval_response.results),
            answer_length=len(answer.answer),
            rerank=rerank,
        )

        return retrieval_response, answer

    def save(self) -> None:
        """Save project state to disk.

        Stores storage_path and context_prompt_path as relative paths in config.json
        so the project is portable across machines.

        Each file is replaced in one step, but the set of them is not
        transactional: a crash partway through leaves newer artifacts beside
        older ones, which a rebuild resolves.
        """
        project_dir = self.project_dir
        project_dir.mkdir(parents=True, exist_ok=True)

        write_json(project_dir / "config.json", self._portable_config(), indent=_CONFIG_INDENT)
        write_json(project_dir / "raw_chunks.json", [c.model_dump() for c in self._chunks])
        write_json(
            project_dir / "segments.json",
            {_encode_segment_key(key): text for key, text in self._segments.items()},
        )
        write_json(
            project_dir / "chunks.json",
            [c.to_storage_dict() for c in self._contextualized_chunks],
        )

        if self._faiss is not None:
            self._faiss.save(project_dir)

        if self._bm25 is not None:
            self._bm25.save(project_dir)

        logger.info("project_saved", path=str(project_dir))

    def load(self) -> None:
        """Load project state from disk.

        Only the indexes are read here; the stored corpus is parsed on first
        use. See _DeferredArtifact.

        Raises:
            FileNotFoundError: If the project directory does not exist.
        """
        project_dir = self.project_dir

        if not project_dir.exists():
            raise FileNotFoundError(f"Project not found: {project_dir}")

        self._defer("_chunks", lambda: _parse_chunks(project_dir / "raw_chunks.json"))
        self._defer("_segments", lambda: _parse_segments(project_dir / "segments.json"))
        self._defer(
            "_contextualized_chunks",
            lambda: _parse_contextualized(project_dir / "chunks.json"),
        )

        self._load_indexes(project_dir)
        self._attach_retriever()

        logger.info(
            "project_loaded",
            path=str(project_dir),
            faiss=self._faiss is not None,
            bm25=self._bm25 is not None,
        )

    @classmethod
    def create(
        cls,
        name: str,
        storage_path: Path | None = None,
        **kwargs: Any,
    ) -> "Project":
        """Create a new project.

        Args:
            name: Project name.
            storage_path: Base storage path. Defaults to settings.STORAGE_PATH.
            **kwargs: Additional ProjectConfig parameters.

        Returns:
            New Project instance.
        """
        defaults: dict[str, Any] = {
            "storage_path": storage_path or settings.STORAGE_PATH,
            "segment_size": settings.SEGMENT_SIZE,
            "segment_overlap": settings.SEGMENT_OVERLAP,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_model": settings.EMBEDDING_MODEL,
            "context_model": settings.CONTEXT_MODEL,
        }
        defaults.update(kwargs)
        return cls(ProjectConfig(name=name, **defaults))

    @classmethod
    def open(cls, name: str, storage_path: Path | None = None) -> "Project":
        """Open an existing project.

        Resolves relative paths in config.json back to absolute using the
        actual storage location, so projects are portable across machines.

        Args:
            name: Project name.
            storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

        Returns:
            Loaded Project instance.

        Raises:
            FileNotFoundError: If the project's config.json does not exist.
        """
        path = storage_path or settings.STORAGE_PATH
        project_dir = path / name
        config_path = project_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")

        config_data = json.loads(config_path.read_text(encoding="utf-8"))

        if not Path(config_data.get("storage_path", "")).is_absolute():
            config_data["storage_path"] = str(path.resolve())

        prompt_path = config_data.get("context_prompt_path")
        if prompt_path and not Path(prompt_path).is_absolute():
            config_data["context_prompt_path"] = str((project_dir / prompt_path).resolve())

        project = cls(ProjectConfig(**config_data))
        project.load()
        return project

    def _defer(self, name: str, loader: Callable[[], Any]) -> None:
        """Arrange for an artifact to be parsed the first time it is read."""
        self.__dict__.pop(name, None)
        self._deferred[name] = loader

    def _materialize(self, name: str) -> Any:
        """Run one artifact's deferred parse and keep the result."""
        value = self._deferred.pop(name)()
        self.__dict__[name] = value
        return value

    def _restore_checkpoint(self, resume: bool, total_segments: int) -> set[str]:
        """Pick up an interrupted build, or start a fresh one.

        Args:
            resume: If False, any checkpoint on disk is ignored.
            total_segments: Segment count, reported in the resume log line.

        Returns:
            The segment keys already contextualized.
        """
        self._contextualized_chunks = []

        checkpoint = self._checkpoint.read() if resume else None
        if checkpoint is None:
            # Appending to what an earlier build left would fold two runs into one.
            self._checkpoint.clear()
            return set()

        self._contextualized_chunks = [
            ContextualizedChunk.from_storage_dict(item)
            for item in checkpoint.contextualized_chunks
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
            seg_key_str = _encode_segment_key(seg_key)
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
                    segment=self._segments.get(seg_key, ""),
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
        self._contextualized_chunks.extend(
            chunk for _, seg_key_str in pending for chunk in produced[seg_key_str]
        )

        logger.info(
            "context_generation_complete",
            num_chunks=len(self._contextualized_chunks),
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
        total = len(self._contextualized_chunks)
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

    async def _build_indexes(self, *, use_faiss: bool, use_bm25: bool) -> None:
        """Build the enabled indexes and wire a retriever over them."""
        if use_faiss:
            self._faiss = FAISSStore(embedding_model=self._config.embedding_model)
            await self._faiss.abuild_index(self._contextualized_chunks)
            logger.info("faiss_index_built")

        if use_bm25:
            self._bm25 = BM25Store()
            self._bm25.build_index(self._contextualized_chunks)
            logger.info("bm25_index_built")

        self._attach_retriever()

    def _attach_retriever(self) -> None:
        """Wire a retriever over the current indexes."""
        self._retriever = Retriever(
            faiss_store=self._faiss,
            bm25_store=self._bm25,
            semantic_weight=self._config.fusion_weight_semantic,
            lexical_weight=self._config.fusion_weight_lexical,
        )

    def _load_indexes(self, project_dir: Path) -> None:
        """Attach whichever indexes exist on disk and are enabled in config.

        A pickled index counts as present so load() can refuse it by name,
        rather than the project opening quietly without one.
        """
        if (project_dir / FAISS_INDEX_FILENAME).exists() and self._config.enable_faiss:
            self._faiss = FAISSStore(embedding_model=self._config.embedding_model)
            self._faiss.load(project_dir)

        if self._config.enable_bm25 and (
            (project_dir / BM25_INDEX_FILENAME).exists()
            or (project_dir / LEGACY_BM25_INDEX_FILENAME).exists()
        ):
            self._bm25 = BM25Store()
            # Passed as a callable: reading the chunks here would defeat the
            # deferred parse.
            self._bm25.load(project_dir, lambda: self._contextualized_chunks)

    def _portable_config(self) -> dict[str, Any]:
        """Serialize the config with absolute paths made relative where possible.

        An absolute storage_path would pin the project to the machine that wrote
        it, so it is stored relative to the project's parent directory and
        rebased by open().
        """
        data = self._config.model_dump(mode="json")

        storage_path = Path(data["storage_path"])
        if storage_path.is_absolute():
            try:
                data["storage_path"] = str(storage_path.relative_to(self.project_dir.parent))
            except ValueError:
                data["storage_path"] = "."

        prompt_path = data.get("context_prompt_path")
        if prompt_path and Path(prompt_path).is_absolute():
            try:
                data["context_prompt_path"] = str(
                    Path(prompt_path).relative_to(self.project_dir)
                )
            except ValueError:
                pass  # Prompt lives outside the project; keep it absolute.

        return data
