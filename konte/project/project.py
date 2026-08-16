"""Project class - main interface for contextual RAG."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

import structlog

from konte.checkpoint import CheckpointLog
from konte.chunker import create_chunks
from konte.config import settings
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
    validate_project_name,
)
from konte.project.builder import ProjectBuilder
from konte.project.store import ProjectStore
from konte.stores import BM25Store, FAISSStore, Retriever

logger = structlog.get_logger()

_T = TypeVar("_T")


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
    """Main interface for contextual RAG operations.

    Owns the corpus and the indexes; ProjectBuilder fills them and ProjectStore
    keeps them on disk.
    """

    _chunks = _DeferredArtifact[list[Chunk]]()
    _segments = _DeferredArtifact[dict[SegmentKey, str]]()
    _contextualized_chunks = _DeferredArtifact[list[ContextualizedChunk]]()

    def __init__(self, config: ProjectConfig):
        """Initialize project with configuration.

        Args:
            config: Project configuration.
        """
        self._config = config
        self._store = ProjectStore(config.storage_path / config.name)
        self._deferred: dict[str, Callable[[], Any]] = {}
        self._chunks = []
        self._contextualized_chunks = []
        self._segments = {}
        self._checkpoint = CheckpointLog(self._store.directory)
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
        return self._store.directory

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

        builder = ProjectBuilder(
            self._config,
            self._checkpoint,
            enable_faiss=enable_faiss,
            enable_bm25=enable_bm25,
        )
        try:
            built = await builder.run(
                self._chunks,
                self._segments,
                skip_context=skip_context,
                resume=resume,
                prompt_path=prompt_path,
            )
        finally:
            # A rejected build's contexts are paid for; keep them.
            self._contextualized_chunks = builder.chunks

        # A skipped index keeps the one the project already has.
        if built.faiss is not None:
            self._faiss = built.faiss
        if built.bm25 is not None:
            self._bm25 = built.bm25
        self._attach_retriever()

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
        """
        self._store.write(
            self._config,
            chunks=self._chunks,
            segments=self._segments,
            contextualized_chunks=self._contextualized_chunks,
            faiss=self._faiss,
            bm25=self._bm25,
        )

    def load(self) -> None:
        """Load project state from disk.

        Only the indexes are read here; the stored corpus is parsed on first
        use. See _DeferredArtifact.

        Raises:
            FileNotFoundError: If the project directory does not exist.
        """
        store = self._store

        if not store.exists():
            raise FileNotFoundError(f"Project not found: {store.directory}")

        self._defer("_chunks", store.read_chunks)
        self._defer("_segments", store.read_segments)
        self._defer("_contextualized_chunks", store.read_contextualized_chunks)

        # Passed as a callable: reading the chunks here would defeat the
        # deferred parse.
        self._faiss, self._bm25 = store.read_indexes(
            self._config, lambda: self._contextualized_chunks
        )
        self._attach_retriever()

        logger.info(
            "project_loaded",
            path=str(store.directory),
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

        Raises:
            ValueError: If the name is not a single path component.
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
            ValueError: If the name is not a single path component.
            FileNotFoundError: If the project's config.json does not exist.
        """
        path = storage_path or settings.STORAGE_PATH
        store = ProjectStore(path / validate_project_name(name))

        project = cls(store.read_config(path))
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

    def _attach_retriever(self) -> None:
        """Wire a retriever over the current indexes."""
        self._retriever = Retriever(
            faiss_store=self._faiss,
            bm25_store=self._bm25,
            semantic_weight=self._config.fusion_weight_semantic,
            lexical_weight=self._config.fusion_weight_lexical,
        )
