"""Project class - main interface for contextual RAG."""

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import structlog

from konte.chunker import create_chunks
from konte.config import settings
from konte.context import generate_contexts_batch, load_prompt_template
from konte.generator import GeneratedAnswer, generate_answer
from konte.loader import load_document
from konte.models import (
    BuildCheckpoint,
    Chunk,
    ContextualizedChunk,
    MetadataFilter,
    ProjectConfig,
    RetrievalMode,
    RetrievalResponse,
    SegmentKey,
)
from konte.stores import BM25Store, FAISSStore, Retriever

logger = structlog.get_logger()

_SEGMENT_KEY_SEPARATOR = "|"


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


def _write_json(path: Path, data: Any) -> None:
    """Write data to path as indented UTF-8 JSON."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    """Read UTF-8 JSON from path, or return None when the file is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class Project:
    """Main interface for contextual RAG operations."""

    def __init__(self, config: ProjectConfig):
        """Initialize project with configuration.

        Args:
            config: Project configuration.
        """
        self._config = config
        self._chunks: list[Chunk] = []
        self._contextualized_chunks: list[ContextualizedChunk] = []
        self._segments: dict[SegmentKey, str] = {}
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

        Args:
            file_paths: Document paths to add.

        Returns:
            Number of chunks created.
        """
        added = 0

        for raw_path in file_paths:
            file_path = Path(raw_path)
            logger.info("loading_document", path=str(file_path))

            chunks, segments_map = create_chunks(
                text=load_document(file_path),
                source=file_path.name,
                segment_size=self._config.segment_size,
                segment_overlap=self._config.segment_overlap,
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )

            self._segments.update(segments_map)
            self._chunks.extend(chunks)
            added += len(chunks)

            logger.info("document_chunked", path=str(file_path), num_chunks=len(chunks))

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
        """
        if not self._chunks:
            logger.warning("build_no_chunks")
            return

        use_faiss = enable_faiss if enable_faiss is not None else self._config.enable_faiss
        use_bm25 = enable_bm25 if enable_bm25 is not None else self._config.enable_bm25
        if not use_faiss and not use_bm25:
            raise ValueError("At least one index (FAISS or BM25) must be enabled")

        chunks_by_segment = _group_by_segment(self._chunks)
        checkpoint, completed = self._restore_checkpoint(resume, len(chunks_by_segment))
        prompt_template = (
            None
            if skip_context
            else load_prompt_template(prompt_path or self._config.context_prompt_path)
        )

        await self._generate_contexts(
            chunks_by_segment, checkpoint, completed, prompt_template, skip_context
        )
        self._build_indexes(use_faiss=use_faiss, use_bm25=use_bm25)

        self._clear_checkpoint()
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
        """
        project_dir = self.project_dir
        project_dir.mkdir(parents=True, exist_ok=True)

        _write_json(project_dir / "config.json", self._portable_config())
        _write_json(project_dir / "raw_chunks.json", [c.model_dump() for c in self._chunks])
        _write_json(
            project_dir / "segments.json",
            {_encode_segment_key(key): text for key, text in self._segments.items()},
        )
        _write_json(
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

        Raises:
            FileNotFoundError: If the project directory does not exist.
        """
        project_dir = self.project_dir

        if not project_dir.exists():
            raise FileNotFoundError(f"Project not found: {project_dir}")

        raw_chunks = _read_json(project_dir / "raw_chunks.json")
        if raw_chunks is not None:
            self._chunks = [Chunk(**item) for item in raw_chunks]

        segments = _read_json(project_dir / "segments.json")
        if segments is not None:
            self._segments = {_decode_segment_key(key): text for key, text in segments.items()}

        contextualized = _read_json(project_dir / "chunks.json")
        if contextualized:
            self._contextualized_chunks = [
                ContextualizedChunk.from_storage_dict(item) for item in contextualized
            ]

        self._load_indexes(project_dir)
        self._retriever = Retriever(faiss_store=self._faiss, bm25_store=self._bm25)

        logger.info(
            "project_loaded",
            path=str(project_dir),
            num_chunks=len(self._chunks),
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
            "context_min_tokens": settings.CONTEXT_MIN_TOKENS,
            "context_max_tokens": settings.CONTEXT_MAX_TOKENS,
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

    def _checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        return self.project_dir / "context_checkpoint.json"

    def _load_checkpoint(self) -> BuildCheckpoint | None:
        """Load checkpoint if exists."""
        data = _read_json(self._checkpoint_path())
        return None if data is None else BuildCheckpoint(**data)

    def _save_checkpoint(self, checkpoint: BuildCheckpoint) -> None:
        """Save checkpoint to disk."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path().write_text(
            checkpoint.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _clear_checkpoint(self) -> None:
        """Remove checkpoint file after successful build."""
        self._checkpoint_path().unlink(missing_ok=True)

    def _restore_checkpoint(
        self,
        resume: bool,
        total_segments: int,
    ) -> tuple[BuildCheckpoint, set[str]]:
        """Pick up an interrupted build, or start a fresh one.

        Args:
            resume: If False, any checkpoint on disk is ignored.
            total_segments: Segment count, reported in the resume log line.

        Returns:
            Tuple of (checkpoint to append to, segment keys already contextualized).
        """
        self._contextualized_chunks = []

        checkpoint = self._load_checkpoint() if resume else None
        if checkpoint is None:
            return BuildCheckpoint(), set()

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
        return checkpoint, completed

    async def _generate_contexts(
        self,
        chunks_by_segment: dict[SegmentKey, list[Chunk]],
        checkpoint: BuildCheckpoint,
        completed: set[str],
        prompt_template: str | None,
        skip_context: bool,
    ) -> None:
        """Contextualize every not-yet-processed segment.

        The checkpoint is rewritten after each segment because context generation
        is the expensive half of a build; an interrupted run then resumes at a
        segment boundary instead of from the beginning.
        """
        total_segments = len(chunks_by_segment)
        logger.info(
            "context_generation_started",
            total_segments=total_segments,
            skip_context=skip_context,
        )

        for seg_key, segment_chunks in chunks_by_segment.items():
            seg_key_str = _encode_segment_key(seg_key)

            if seg_key_str in completed:
                logger.info("segment_skipped", segment_key=seg_key_str, reason="checkpoint")
                continue

            logger.info(
                "generating_context_for_segment",
                segment_key=seg_key_str,
                total_segments=total_segments,
                num_chunks=len(segment_chunks),
            )
            ctx_chunks = await generate_contexts_batch(
                segment=self._segments.get(seg_key, ""),
                chunks=segment_chunks,
                model=self._config.context_model,
                prompt_template=prompt_template,
                skip_context=skip_context,
            )
            self._contextualized_chunks.extend(ctx_chunks)

            checkpoint.completed_segments.append(seg_key_str)
            checkpoint.contextualized_chunks.extend(c.to_storage_dict() for c in ctx_chunks)
            self._save_checkpoint(checkpoint)
            logger.info(
                "checkpoint_saved",
                segment_key=seg_key_str,
                completed=len(checkpoint.completed_segments),
                total=total_segments,
            )

        logger.info(
            "context_generation_complete",
            num_chunks=len(self._contextualized_chunks),
            skipped=skip_context,
        )

    def _build_indexes(self, *, use_faiss: bool, use_bm25: bool) -> None:
        """Build the enabled indexes and wire a retriever over them."""
        if use_faiss:
            self._faiss = FAISSStore(embedding_model=self._config.embedding_model)
            self._faiss.build_index(self._contextualized_chunks)
            logger.info("faiss_index_built")

        if use_bm25:
            self._bm25 = BM25Store()
            self._bm25.build_index(self._contextualized_chunks)
            logger.info("bm25_index_built")

        self._retriever = Retriever(faiss_store=self._faiss, bm25_store=self._bm25)

    def _load_indexes(self, project_dir: Path) -> None:
        """Attach whichever indexes exist on disk and are enabled in config."""
        if (project_dir / "faiss.faiss").exists() and self._config.enable_faiss:
            self._faiss = FAISSStore(embedding_model=self._config.embedding_model)
            self._faiss.load(project_dir)

        if (project_dir / "bm25.pkl").exists() and self._config.enable_bm25:
            self._bm25 = BM25Store()
            self._bm25.load(project_dir)

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
