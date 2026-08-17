"""A project's indexes and the retriever that reads them, as one object."""

from pathlib import Path
from typing import NamedTuple

import structlog

from konte.domain.config import ProjectConfig
from konte.domain.corpus import Corpus
from konte.domain.models import ContextualizedChunk, RetrievalRequest, RetrievalResponse
from konte.index.bm25_store import INDEX_FILENAME as BM25_INDEX_FILENAME
from konte.index.bm25_store import LEGACY_INDEX_FILENAME as LEGACY_BM25_INDEX_FILENAME
from konte.index.bm25_store import BM25Store
from konte.index.chunks import ChunkSource
from konte.index.faiss_store import INDEX_FILENAME as FAISS_INDEX_FILENAME
from konte.index.faiss_store import FAISSStore
from konte.retrieval.retriever import Retriever

logger = structlog.get_logger()


class IndexSelection(NamedTuple):
    """Which indexes a build was asked to produce."""

    faiss: bool
    bm25: bool

    @classmethod
    def resolve(
        cls,
        config: ProjectConfig,
        enable_faiss: bool | None,
        enable_bm25: bool | None,
    ) -> "IndexSelection":
        """Settle the enable flags against the project's configuration.

        Args:
            config: The project's configuration.
            enable_faiss: Build the semantic index. None follows the config.
            enable_bm25: Build the lexical index. None follows the config.

        Returns:
            The resolved pair.

        Raises:
            ValueError: If both indexes are disabled, which would leave nothing
                to retrieve against. Raised before a build spends anything.
        """
        selection = cls(
            config.enable_faiss if enable_faiss is None else enable_faiss,
            config.enable_bm25 if enable_bm25 is None else enable_bm25,
        )
        if not selection.faiss and not selection.bm25:
            raise ValueError("At least one index (FAISS or BM25) must be enabled")
        return selection


class IndexBundle:
    """The indexes a project ranks against, wired to one retriever.

    Building, loading and querying all go through here, so the stores and the
    retriever over them cannot drift apart.
    """

    __slots__ = ("_bm25", "_config", "_faiss", "_retriever")

    def __init__(
        self,
        config: ProjectConfig,
        faiss: FAISSStore | None = None,
        bm25: BM25Store | None = None,
    ) -> None:
        """Wire a retriever over whichever indexes were passed.

        Args:
            config: The project's configuration, read for the fusion weights.
            faiss: The semantic index, or None where there is none.
            bm25: The lexical index, or None where there is none.
        """
        self._config = config
        self._faiss = faiss
        self._bm25 = bm25
        self._retriever = Retriever(
            faiss_store=faiss,
            bm25_store=bm25,
            semantic_weight=config.fusion_weight_semantic,
            lexical_weight=config.fusion_weight_lexical,
        )

    @classmethod
    async def build(
        cls,
        config: ProjectConfig,
        chunks: list[ContextualizedChunk],
        selection: IndexSelection,
    ) -> "IndexBundle":
        """Build the selected indexes over already-contextualized chunks.

        Args:
            config: The project's configuration.
            chunks: What to index.
            selection: Which indexes to build.

        Returns:
            A bundle holding what was built, and None for what was skipped.
        """
        faiss: FAISSStore | None = None
        bm25: BM25Store | None = None

        if selection.faiss:
            faiss = FAISSStore(embedding_model=config.embedding_model)
            await faiss.abuild_index(chunks)
            logger.info("faiss_index_built")

        if selection.bm25:
            bm25 = BM25Store()
            bm25.build_index(chunks)
            logger.info("bm25_index_built")

        return cls(config, faiss, bm25)

    @classmethod
    def load(cls, directory: Path, config: ProjectConfig, corpus: Corpus) -> "IndexBundle":
        """Attach whichever indexes exist on disk and are enabled in config.

        A pickled index counts as present so the store can refuse it by name,
        rather than the project opening quietly without one.

        Args:
            directory: Directory the project's artifacts live in.
            config: The project's configuration.
            corpus: Read on the first query that needs a chunk rather than a
                rank. Both indexes bind to one reading of it.

        Returns:
            A bundle holding what was on disk.
        """
        faiss: FAISSStore | None = None
        bm25: BM25Store | None = None
        chunks = ChunkSource(lambda: corpus.contextualized_chunks)

        if config.enable_faiss and (directory / FAISS_INDEX_FILENAME).exists():
            faiss = FAISSStore(embedding_model=config.embedding_model)
            faiss.load(directory, chunks)

        if config.enable_bm25 and (
            (directory / BM25_INDEX_FILENAME).exists()
            or (directory / LEGACY_BM25_INDEX_FILENAME).exists()
        ):
            bm25 = BM25Store()
            bm25.load(directory, chunks)

        return cls(config, faiss, bm25)

    def save(self, directory: Path) -> None:
        """Write whichever indexes this bundle holds; an absent one is left alone."""
        if self._faiss is not None:
            self._faiss.save(directory)
        if self._bm25 is not None:
            self._bm25.save(directory)

    def replacing(self, built: "IndexBundle") -> "IndexBundle":
        """Take every index `built` produced, keeping this one's for the rest.

        A build that skipped an index leaves the project the one it already had,
        rather than dropping it.

        Args:
            built: What the build produced.

        Returns:
            The merged bundle.
        """
        return IndexBundle(
            built._config,
            built._faiss if built._faiss is not None else self._faiss,
            built._bm25 if built._bm25 is not None else self._bm25,
        )

    @property
    def faiss(self) -> FAISSStore | None:
        """FAISSStore | None: The semantic index, where there is one."""
        return self._faiss

    @property
    def bm25(self) -> BM25Store | None:
        """BM25Store | None: The lexical index, where there is one."""
        return self._bm25

    @property
    def is_empty(self) -> bool:
        """bool: True when there is no index to rank against."""
        return self._faiss is None and self._bm25 is None

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Answer one request, blocking on whatever it needs."""
        return self._retriever.retrieve(request)

    async def aretrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Answer one request without blocking the loop."""
        return await self._retriever.aretrieve(request)
