"""Project class - main interface for contextual RAG."""

import json
from pathlib import Path
from typing import Callable

import structlog

from konte.bm25_store import BM25Store
from konte.chunker import create_chunks
from konte.context import generate_contexts_batch
from konte.faiss_store import FAISSStore
from konte.loader import load_document
from konte.models import Chunk, ContextualizedChunk, ProjectConfig, RetrievalResponse
from konte.retriever import Retriever, RetrievalMode
from konte.settings import settings

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
        self._segments: dict[int, str] = {}  # segment_idx -> segment text
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

            chunks = create_chunks(
                text=content,
                source=file_path.name,
                segment_size=self._config.segment_size,
                segment_overlap=self._config.segment_overlap,
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )

            # Store segment text for context generation
            for chunk in chunks:
                if chunk.segment_idx not in self._segments:
                    # Extract segment from content (approximate)
                    # In practice, we should store segments during chunking
                    self._segments[chunk.segment_idx] = content

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
        chunks_by_segment: dict[int, list[Chunk]] = {}
        for chunk in self._chunks:
            seg_idx = chunk.segment_idx
            if seg_idx not in chunks_by_segment:
                chunks_by_segment[seg_idx] = []
            chunks_by_segment[seg_idx].append(chunk)

        # Generate context for each segment's chunks
        self._contextualized_chunks = []
        for seg_idx, segment_chunks in chunks_by_segment.items():
            segment_text = self._segments.get(seg_idx, "")

            ctx_chunks = await generate_contexts_batch(
                segment=segment_text,
                chunks=segment_chunks,
                model=self._config.context_model,
                max_concurrent=settings.MAX_CONCURRENT_CALLS,
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
    ) -> RetrievalResponse:
        """Query the project.

        Args:
            query: Query string.
            mode: Retrieval mode - "hybrid", "semantic", or "lexical".
            top_k: Number of results. Defaults to settings.DEFAULT_TOP_K.

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
        return self._retriever.retrieve(query, mode=mode, top_k=k)

    def as_retriever(self) -> Callable[[str], RetrievalResponse]:
        """Return a callable retriever for Agno integration.

        Returns:
            Callable that takes a query and returns RetrievalResponse.
        """
        return lambda q: self.query(q)

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

        # Save chunks
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

        # Load chunks
        chunks_path = project_dir / "chunks.json"
        if chunks_path.exists():
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            self._contextualized_chunks = [
                ContextualizedChunk(
                    chunk=Chunk(**item["chunk"]),
                    context=item["context"],
                )
                for item in chunks_data
            ]
            self._chunks = [c.chunk for c in self._contextualized_chunks]

        # Load indexes
        faiss_path = project_dir / "faiss.index"
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
            num_chunks=len(self._contextualized_chunks),
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
        path = storage_path or settings.STORAGE_PATH
        config = ProjectConfig(name=name, storage_path=path, **kwargs)
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
