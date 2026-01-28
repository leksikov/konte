"""Project class - main interface for contextual RAG."""

import json
from pathlib import Path
from typing import Any, Callable

import structlog

from konte.chunker import create_chunks
from konte.config import settings
from konte.context import generate_contexts_batch
from konte.generator import GeneratedAnswer, generate_answer
from konte.loader import load_document
from konte.models import Chunk, ContextualizedChunk, ProjectConfig, RetrievalResponse
from konte.stores import BM25Store, FAISSStore, Retriever, RetrievalMode

logger = structlog.get_logger()


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
        self._segments: dict[tuple[str, int], str] = {}  # (source, segment_idx) -> segment text
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

    def add_documents(self, file_paths: list[Path]) -> int:
        """Add documents to the project.

        Loads, segments, and chunks documents.

        Args:
            file_paths: List of document paths to add.

        Returns:
            Number of chunks created.
        """
        all_chunks = []

        for file_path in file_paths:
            file_path = Path(file_path)
            logger.info("loading_document", path=str(file_path))

            content = load_document(file_path)

            chunks, segments_map = create_chunks(
                text=content,
                source=file_path.name,
                segment_size=self._config.segment_size,
                segment_overlap=self._config.segment_overlap,
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )

            # Store segment text for context generation
            self._segments.update(segments_map)

            all_chunks.extend(chunks)
            logger.info(
                "document_chunked",
                path=str(file_path),
                num_chunks=len(chunks),
            )

        self._chunks.extend(all_chunks)
        logger.info("documents_added", total_chunks=len(self._chunks))
        return len(all_chunks)

    async def build(
        self,
        skip_context: bool = False,
        enable_faiss: bool | None = None,
        enable_bm25: bool | None = None,
    ) -> None:
        """Build indexes from added documents.

        Args:
            skip_context: If True, skip LLM context generation (standard RAG).
            enable_faiss: Enable FAISS index. Defaults to config setting.
            enable_bm25: Enable BM25 index. Defaults to config setting.
        """
        if not self._chunks:
            logger.warning("build_no_chunks")
            return

        use_faiss = enable_faiss if enable_faiss is not None else self._config.enable_faiss
        use_bm25 = enable_bm25 if enable_bm25 is not None else self._config.enable_bm25

        if not use_faiss and not use_bm25:
            raise ValueError("At least one index (FAISS or BM25) must be enabled")

        # Group chunks by segment for context generation
        chunks_by_segment: dict[tuple[str, int], list[Chunk]] = {}
        for chunk in self._chunks:
            key = (chunk.source, chunk.segment_idx)
            if key not in chunks_by_segment:
                chunks_by_segment[key] = []
            chunks_by_segment[key].append(chunk)

        # Generate context for each segment's chunks
        self._contextualized_chunks = []
        total_segments = len(chunks_by_segment)
        logger.info(
            "context_generation_started",
            total_segments=total_segments,
            skip_context=skip_context,
        )

        for seg_key, segment_chunks in chunks_by_segment.items():
            segment_text = self._segments.get(seg_key, "")
            logger.info(
                "generating_context_for_segment",
                segment_index=seg_key,
                total_segments=total_segments,
                num_chunks=len(segment_chunks),
            )

            ctx_chunks = await generate_contexts_batch(
                segment=segment_text,
                chunks=segment_chunks,
                model=self._config.context_model,
                skip_context=skip_context,
            )
            self._contextualized_chunks.extend(ctx_chunks)

        logger.info(
            "context_generation_complete",
            num_chunks=len(self._contextualized_chunks),
            skipped=skip_context,
        )

        # Build indexes
        if use_faiss:
            self._faiss = FAISSStore(embedding_model=self._config.embedding_model)
            self._faiss.build_index(self._contextualized_chunks)
            logger.info("faiss_index_built")

        if use_bm25:
            self._bm25 = BM25Store()
            self._bm25.build_index(self._contextualized_chunks)
            logger.info("bm25_index_built")

        # Create retriever
        self._retriever = Retriever(
            faiss_store=self._faiss,
            bm25_store=self._bm25,
        )

        logger.info("project_build_complete")

    def query(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Query the project (sync, no reranking).

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "company": "ACME", "year": 2024}

        Returns:
            RetrievalResponse with results.
        """
        if self._retriever is None:
            logger.warning("query_no_retriever")
            return RetrievalResponse(
                results=[],
                query=query,
                total_found=0,
                top_score=0.0,
                score_spread=0.0,
                has_high_confidence=False,
                suggested_action="refine_query",
            )

        k = top_k or settings.DEFAULT_TOP_K
        return self._retriever.retrieve(query, mode=mode, top_k=k, metadata_filter=metadata_filter)

    async def query_async(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        rerank: bool = False,
        rerank_initial_k: int = 50,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Query the project (async, with optional reranking).

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.
            rerank: If True, apply Qwen3-Reranker-8B reranking.
            rerank_initial_k: Number of candidates to retrieve before reranking.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "company": "ACME", "year": 2024}

        Returns:
            RetrievalResponse with results.
        """
        if self._retriever is None:
            logger.warning("query_no_retriever")
            return RetrievalResponse(
                results=[],
                query=query,
                total_found=0,
                top_score=0.0,
                score_spread=0.0,
                has_high_confidence=False,
                suggested_action="refine_query",
            )

        k = top_k or settings.DEFAULT_TOP_K

        if rerank:
            return await self._retriever.retrieve_with_rerank(
                query, mode=mode, top_k=k, initial_k=rerank_initial_k,
                metadata_filter=metadata_filter
            )
        else:
            return self._retriever.retrieve(query, mode=mode, top_k=k, metadata_filter=metadata_filter)

    def as_retriever(self) -> Callable[[str], RetrievalResponse]:
        """Return a callable retriever for Agno integration.

        Returns:
            Callable that takes a query and returns RetrievalResponse.
        """
        return lambda q: self.query(q)

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
        metadata_filter: dict[str, Any] | None = None,
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
            rerank: If True, apply Qwen3-Reranker-8B reranking.
            rerank_initial_k: Number of candidates to retrieve before reranking.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "company": "ACME", "year": 2024}

        Returns:
            Tuple of (RetrievalResponse, GeneratedAnswer).
        """
        # First, retrieve chunks (with optional reranking)
        retrieval_response = await self.query_async(
            query, mode=mode, top_k=top_k, rerank=rerank, rerank_initial_k=rerank_initial_k,
            metadata_filter=metadata_filter
        )

        # Then, generate answer using LLM
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
        """Save project state to disk."""
        project_dir = self.project_dir
        project_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = project_dir / "config.json"
        config_path.write_text(
            self._config.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Save raw chunks (before build)
        raw_chunks_path = project_dir / "raw_chunks.json"
        raw_chunks_data = [c.model_dump() for c in self._chunks]
        raw_chunks_path.write_text(json.dumps(raw_chunks_data, indent=2), encoding="utf-8")

        # Save segments (before build) - convert tuple keys to string for JSON
        segments_path = project_dir / "segments.json"
        segments_data = {f"{source}|{idx}": text for (source, idx), text in self._segments.items()}
        segments_path.write_text(json.dumps(segments_data, indent=2), encoding="utf-8")

        # Save contextualized chunks (after build)
        chunks_path = project_dir / "chunks.json"
        chunks_data = [
            {
                "chunk": c.chunk.model_dump(),
                "context": c.context,
            }
            for c in self._contextualized_chunks
        ]
        chunks_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

        # Save indexes
        if self._faiss is not None:
            self._faiss.save(project_dir)

        if self._bm25 is not None:
            self._bm25.save(project_dir)

        logger.info("project_saved", path=str(project_dir))

    def load(self) -> None:
        """Load project state from disk."""
        project_dir = self.project_dir

        if not project_dir.exists():
            raise FileNotFoundError(f"Project not found: {project_dir}")

        # Load raw chunks (before build)
        raw_chunks_path = project_dir / "raw_chunks.json"
        if raw_chunks_path.exists():
            raw_chunks_data = json.loads(raw_chunks_path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**item) for item in raw_chunks_data]

        # Load segments (before build) - parse string keys back to tuples
        segments_path = project_dir / "segments.json"
        if segments_path.exists():
            segments_data = json.loads(segments_path.read_text(encoding="utf-8"))
            self._segments = {}
            for k, v in segments_data.items():
                if "|" in k:  # New format: "source|idx"
                    source, idx = k.rsplit("|", 1)
                    self._segments[(source, int(idx))] = v
                else:  # Backward compat: old int-only format
                    self._segments[("unknown", int(k))] = v

        # Load contextualized chunks (after build)
        chunks_path = project_dir / "chunks.json"
        if chunks_path.exists():
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            if chunks_data:  # Only load if not empty
                self._contextualized_chunks = [
                    ContextualizedChunk(
                        chunk=Chunk(**item["chunk"]),
                        context=item["context"],
                    )
                    for item in chunks_data
                ]

        # Load indexes
        faiss_path = project_dir / "faiss.faiss"
        if faiss_path.exists() and self._config.enable_faiss:
            self._faiss = FAISSStore(embedding_model=self._config.embedding_model)
            self._faiss.load(project_dir)

        bm25_path = project_dir / "bm25.pkl"
        if bm25_path.exists() and self._config.enable_bm25:
            self._bm25 = BM25Store()
            self._bm25.load(project_dir)

        # Create retriever
        self._retriever = Retriever(
            faiss_store=self._faiss,
            bm25_store=self._bm25,
        )

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
        **kwargs,
    ) -> "Project":
        """Create a new project.

        Args:
            name: Project name.
            storage_path: Base storage path. Defaults to settings.STORAGE_PATH.
            **kwargs: Additional ProjectConfig parameters.

        Returns:
            New Project instance.
        """
        # Build defaults from settings (SSoT)
        defaults = {
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
        # Allow kwargs to override defaults
        defaults.update(kwargs)
        config = ProjectConfig(name=name, **defaults)
        return cls(config)

    @classmethod
    def open(cls, name: str, storage_path: Path | None = None) -> "Project":
        """Open an existing project.

        Args:
            name: Project name.
            storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

        Returns:
            Loaded Project instance.
        """
        path = storage_path or settings.STORAGE_PATH
        config_path = path / name / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config = ProjectConfig(**config_data)

        project = cls(config)
        project.load()
        return project
