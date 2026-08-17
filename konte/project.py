"""Project - the object every entry point drives contextual RAG through."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import structlog

from konte.answer import GeneratedAnswer, generate_answer
from konte.contextualize.pipeline import BuildPipeline
from konte.domain.config import ProjectConfig, validate_project_name
from konte.domain.corpus import Corpus
from konte.domain.models import (
    MetadataFilter,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
)
from konte.ingest.documents import add_documents
from konte.persistence.repository import ProjectRepository
from konte.retrieval.bundle import IndexBundle, IndexSelection
from konte.runtime.settings import settings

logger = structlog.get_logger()


class Project:
    """Main interface for contextual RAG operations.

    A facade over the layers that do the work: ingest fills the corpus, the
    build pipeline contextualizes it, the index bundle ranks against it, and
    the repository is the only part that knows any of it is on disk.
    """

    __slots__ = ("_config", "_corpus", "_indexes", "_repository")

    def __init__(self, config: ProjectConfig):
        """Initialize project with configuration.

        Args:
            config: Project configuration.
        """
        self._config = config
        self._repository = ProjectRepository(config.storage_path / config.name)
        self._corpus = Corpus()
        self._indexes = IndexBundle(config)

    @property
    def config(self) -> ProjectConfig:
        """Get project configuration."""
        return self._config

    @property
    def corpus(self) -> Corpus:
        """Corpus: The chunks this project holds, and the segments behind them."""
        return self._corpus

    @property
    def project_dir(self) -> Path:
        """Get project directory path."""
        return self._repository.directory

    def add_documents(self, file_paths: Sequence[str | Path]) -> int:
        """Add documents to the project.

        Loads, segments, and chunks documents.

        Args:
            file_paths: Document paths to add.

        Returns:
            Number of chunks created.

        Raises:
            ValueError: If a document's text is already in the project; the
                ones ahead of it in `file_paths` are already added.
        """
        return add_documents(self._corpus, self._config, file_paths)

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
        for chunk in self._corpus.chunks:
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
        if not self._corpus.chunks:
            logger.warning("build_no_chunks")
            return

        # Before the pipeline: an unbuildable selection should cost no LLM calls.
        selection = IndexSelection.resolve(self._config, enable_faiss, enable_bm25)

        pipeline = BuildPipeline(self._config, self._repository.checkpoint)
        try:
            contextualized = await pipeline.run(
                self._corpus,
                skip_context=skip_context,
                resume=resume,
                prompt_path=prompt_path,
            )
        finally:
            # A rejected build's contexts are paid for; keep them.
            self._corpus.contextualized_chunks = pipeline.chunks

        # A skipped index keeps the one the project already has.
        built = await IndexBundle.build(self._config, contextualized, selection)
        self._indexes = self._indexes.replacing(built)

        self._repository.checkpoint.clear()
        logger.info("checkpoint_cleared")
        logger.info("project_build_complete")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Answer one retrieval request, blocking on whatever it needs.

        Args:
            request: What to retrieve; see RetrievalRequest.

        Returns:
            RetrievalResponse with results, empty if the project was never built.
        """
        if self._indexes.is_empty:
            logger.warning("query_no_retriever")
            return RetrievalResponse.empty(request.query)
        return self._indexes.retrieve(request)

    async def aretrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Answer one retrieval request without blocking the loop.

        Args:
            request: What to retrieve; see RetrievalRequest.

        Returns:
            RetrievalResponse with results, empty if the project was never built.
        """
        if self._indexes.is_empty:
            logger.warning("query_no_retriever")
            return RetrievalResponse.empty(request.query)
        return await self._indexes.aretrieve(request)

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
        Every argument is a RetrievalRequest field, documented there.

        Returns:
            RetrievalResponse with results, empty if the project was never built.
        """
        return self.retrieve(
            RetrievalRequest(
                query=query,
                mode=mode,
                top_k=top_k,
                use_keyword_extraction=use_keyword_extraction,
                metadata_filter=metadata_filter,
                source_filter=source_filter,
                inject_evidence=inject_evidence,
                inject_position=inject_position,
            )
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

        Every argument is a RetrievalRequest field, documented there.

        Returns:
            RetrievalResponse with results, empty if the project was never built.
        """
        return await self.aretrieve(
            RetrievalRequest(
                query=query,
                mode=mode,
                top_k=top_k,
                rerank=rerank,
                rerank_initial_k=rerank_initial_k,
                use_keyword_extraction=use_keyword_extraction,
                metadata_filter=metadata_filter,
                source_filter=source_filter,
            )
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

        This is the full RAG pipeline: retrieval + answer generation. The
        retrieval arguments are RetrievalRequest fields, documented there.

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
                cost of one LLM call on top of answer generation.
            metadata_filter: Filter results by metadata (equality match, AND logic).
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
        self._repository.write(self._config, self._corpus, self._indexes)

    def load(self) -> None:
        """Load project state from disk.

        Only the indexes are read here; the stored corpus is parsed on first
        use. See Corpus.

        Raises:
            FileNotFoundError: If the project directory does not exist.
        """
        repository = self._repository

        if not repository.exists():
            raise FileNotFoundError(f"Project not found: {repository.directory}")

        self._corpus = repository.read_corpus()
        self._indexes = repository.read_indexes(self._config, self._corpus)

        logger.info(
            "project_loaded",
            path=str(repository.directory),
            faiss=self._indexes.faiss is not None,
            bm25=self._indexes.bm25 is not None,
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
        repository = ProjectRepository(path / validate_project_name(name))

        project = cls(repository.read_config(path))
        project.load()
        return project
