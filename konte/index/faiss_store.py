"""FAISS vector store for semantic search."""

import asyncio
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import structlog
from langchain_openai import OpenAIEmbeddings

from konte.domain.models import ContextualizedChunk, MetadataFilter
from konte.index.chunks import ChunkSource
from konte.index.embeddings import query_vector
from konte.persistence.integrity import SIGNATURE_SUFFIX, sign, verify
from konte.persistence.storage import read_json, write_json
from konte.runtime.settings import settings

logger = structlog.get_logger()

INDEX_FILENAME = "faiss.faiss"
DOCSTORE_FILENAME = "faiss_docstore.json"

# Where LangChain's save_local pickled the docstore.
LEGACY_DOCSTORE_FILENAME = "faiss.pkl"

SIGNED_FILENAMES = (INDEX_FILENAME, DOCSTORE_FILENAME)

# Bump on any change to the payload save() writes.
_DOCSTORE_VERSION = 2

# Older payloads carry a document per vector; only their count is read.
_LEGACY_DOCSTORE_VERSION = 1

# Documents per embedding request, kept well under the OpenAI embedding API's
# 300K-token request limit.
_EMBED_BATCH_SIZE = 100

# One embedding request that has been sent but not yet folded into the index.
_PendingBatches = deque[tuple[int, "asyncio.Future[list[list[float]]]"]]


def _to_similarity(distance: float) -> float:
    """Convert an L2 distance into a 0-1 score, where 1.0 is an exact match."""
    return float(1.0 / (1.0 + distance))


def _batch_spans(total: int, batch_size: int) -> list[tuple[int, int]]:
    """Split a corpus into the half-open position ranges one request covers."""
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def _texts(chunks: Sequence[ContextualizedChunk], span: tuple[int, int]) -> list[str]:
    """Return the text one embedding request carries.

    Joined per request: joining the whole corpus at once doubles its memory.
    """
    start, stop = span
    return [chunks[position].contextualized_content for position in range(start, stop)]


def _read_vector_count(path: Path) -> int | None:
    """Read how many vectors the stored payload accounts for.

    Returns:
        The recorded count, or None where the payload does not carry one.

    Raises:
        ValueError: If the payload is not a version this reads.
    """
    payload = read_json(path)

    if isinstance(payload, dict):
        version = payload.get("version")
        if version == _DOCSTORE_VERSION:
            count = payload.get("count")
            return count if isinstance(count, int) else None
        if version == _LEGACY_DOCSTORE_VERSION:
            # Saving the project rewrites this as a payload of just the count.
            logger.warning("faiss_docstore_legacy_payload", path=str(path))
            return len(payload.get("index_to_id", {}))

    raise ValueError(
        f"FAISS docstore at {path} is not version {_DOCSTORE_VERSION}. "
        "Rebuild the project to rewrite it."
    )


class FAISSStore:
    """FAISS vector store for semantic search on contextualized chunks.

    Vectors only: FAISS numbers them in insertion order, so a hit is the chunk
    at that position in the corpus the index was built over.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
    ):
        """Initialize FAISS store.

        Args:
            embedding_model: OpenAI embedding model name.
                Defaults to settings.EMBEDDING_MODEL.
        """
        self._embedding_model = embedding_model or settings.EMBEDDING_MODEL
        kwargs: dict[str, Any] = {"model": self._embedding_model}
        if settings.OPENAI_API_KEY:
            kwargs["api_key"] = settings.OPENAI_API_KEY
        self._embeddings = OpenAIEmbeddings(**kwargs)
        self._index: faiss.Index | None = None
        self._chunks = ChunkSource.holding()

    def build_index(
        self,
        chunks: list[ContextualizedChunk],
        batch_size: int = _EMBED_BATCH_SIZE,
    ) -> None:
        """Build FAISS index from contextualized chunks.

        One request at a time; abuild_index() is the concurrent form.

        Args:
            chunks: List of contextualized chunks to index.
            batch_size: Number of documents to embed per request.
        """
        spans = _batch_spans(len(chunks), batch_size)
        if not spans:
            logger.warning("faiss_build_empty_chunks")
            return

        self._reset()
        for number, span in enumerate(spans, start=1):
            vectors = self._embeddings.embed_documents(_texts(chunks, span))
            self._absorb(number, len(spans), vectors)
        self._chunks = ChunkSource.holding(chunks)

        logger.info("faiss_index_built", num_chunks=len(chunks))

    async def abuild_index(
        self,
        chunks: list[ContextualizedChunk],
        batch_size: int = _EMBED_BATCH_SIZE,
        max_concurrency: int | None = None,
    ) -> None:
        """Build FAISS index from contextualized chunks, embedding concurrently.

        A sliding window keeps `max_concurrency` requests on the wire. Batches
        are absorbed in the order they were submitted, so the index does not
        depend on which response arrived first.

        Args:
            chunks: List of contextualized chunks to index.
            batch_size: Number of documents to embed per request.
            max_concurrency: Requests in flight. Defaults to
                settings.MAX_CONCURRENT_CALLS.
        """
        spans = _batch_spans(len(chunks), batch_size)
        if not spans:
            logger.warning("faiss_build_empty_chunks")
            return

        if max_concurrency is None:
            max_concurrency = settings.concurrency_limit
        window = max(1, max_concurrency)

        self._reset()
        pending: _PendingBatches = deque()

        try:
            for number, span in enumerate(spans, start=1):
                if len(pending) == window:
                    await self._absorb_oldest(pending, len(spans))
                request = self._embeddings.aembed_documents(_texts(chunks, span))
                pending.append((number, asyncio.ensure_future(request)))
            while pending:
                await self._absorb_oldest(pending, len(spans))
        except BaseException:
            for _, task in pending:
                task.cancel()
            await asyncio.gather(*(task for _, task in pending), return_exceptions=True)
            raise

        self._chunks = ChunkSource.holding(chunks)

        logger.info("faiss_index_built", num_chunks=len(chunks))

    async def _absorb_oldest(self, pending: "_PendingBatches", total_batches: int) -> None:
        """Wait for the longest-outstanding request and fold it into the index."""
        number, task = pending.popleft()
        self._absorb(number, total_batches, await task)

    def _absorb(
        self,
        number: int,
        total_batches: int,
        vectors: list[list[float]],
    ) -> None:
        """Add one embedded batch, creating the index on the first."""
        if total_batches > 1:
            logger.info("faiss_building_batch", batch=number, total_batches=total_batches)

        rows = np.asarray(vectors, dtype=np.float32)
        if self._index is None:
            self._index = faiss.IndexFlatL2(rows.shape[1])
        self._index.add(rows)

    def _reset(self) -> None:
        """Drop whatever this store held, so a rebuild starts from nothing."""
        self._index = None
        self._chunks = ChunkSource.holding()

    def save(self, directory: Path) -> None:
        """Save FAISS index to disk.

        The payload beside the index is JSON rather than LangChain's pickled
        docstore: reading a pickle runs whatever it holds.

        Args:
            directory: Directory to save index files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index = self._index
        if index is None:
            logger.warning("faiss_save_no_index")
            return

        faiss.write_index(index, str(directory / INDEX_FILENAME))
        write_json(
            directory / DOCSTORE_FILENAME,
            {"version": _DOCSTORE_VERSION, "count": int(index.ntotal)},
        )

        sign(directory, SIGNED_FILENAMES)

        for stale in (LEGACY_DOCSTORE_FILENAME, f"{LEGACY_DOCSTORE_FILENAME}{SIGNATURE_SUFFIX}"):
            (directory / stale).unlink(missing_ok=True)

        logger.info("faiss_index_saved", directory=str(directory))

    def load(self, directory: Path, chunks: ChunkSource) -> None:
        """Load FAISS index from disk and bind it to the corpus it was built over.

        Args:
            directory: Directory containing index files.
            chunks: The chunks the index was built over, in index order. Read
                on the first query, not here.

        Raises:
            FileNotFoundError: If index files don't exist.
            IntegrityError: If the index does not match what was recorded for it.
            ValueError: If the docstore is not a version this reads, or accounts
                for a different number of vectors than the index holds.
        """
        directory = Path(directory)

        index_path = directory / INDEX_FILENAME
        docstore_path = directory / DOCSTORE_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not docstore_path.exists():
            if (directory / LEGACY_DOCSTORE_FILENAME).exists():
                raise ValueError(
                    f"{directory / LEGACY_DOCSTORE_FILENAME} holds a pickled docstore, "
                    f"and reading a pickle runs whatever it holds, so it is never "
                    f"loaded. Rebuild the project to write {DOCSTORE_FILENAME} instead."
                )
            raise FileNotFoundError(f"FAISS docstore not found: {docstore_path}")

        verify(directory, SIGNED_FILENAMES)

        count = _read_vector_count(docstore_path)
        index = faiss.read_index(str(index_path))
        if count is not None and index.ntotal != count:
            raise ValueError(
                f"{index_path} holds {index.ntotal} vectors, but {docstore_path} "
                f"accounts for {count}. Rebuild the project to rewrite them."
            )

        self._index = index
        self._chunks = chunks

        logger.info(
            "faiss_index_loaded",
            directory=str(directory),
            num_chunks=index.ntotal,
        )

    def query(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the FAISS index.

        The query is embedded once per model and text for the process; see
        konte.index.embeddings.

        Args:
            query: Query string.
            top_k: Number of results to return. Defaults to settings.DEFAULT_TOP_K.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "page_no": 5}
            source_filter: Substring match on chunk source field.
                Example: "JOHNSON" matches "JOHNSON_JOHNSON_2022_10K.md"

        Returns:
            List of (chunk, score) tuples, sorted by score descending.
        """
        index = self._index
        if index is None:
            logger.warning("faiss_query_empty_index")
            return []

        k = top_k or settings.DEFAULT_TOP_K

        # Filtered before the search rather than after it, so a restrictive
        # filter still yields k results instead of whatever survives a global top-k.
        selected = self._chunks.select(metadata_filter, source_filter)
        if selected is None:
            return self._search(index, query, k)
        if not selected.size:
            return []

        # Both bound to names: the selector reads the buffer during the search.
        positions = selected.astype(np.int64)
        selector = faiss.IDSelectorArray(positions)
        return self._search(
            index,
            query,
            min(k, positions.size),
            faiss.SearchParametersIVF(sel=selector),
        )

    def _search(
        self,
        index: faiss.Index,
        query: str,
        k: int,
        params: faiss.SearchParameters | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Rank the index against one query, through a selector where given.

        Searched directly rather than through LangChain's
        similarity_search_with_score, which embeds the string itself and so
        reaches the endpoint around the cache.
        """
        chunks = self._ranked_over(index)
        if not chunks:
            logger.warning("faiss_query_no_corpus")
            return []

        distances, positions = index.search(
            query_vector(self._embeddings, self._embedding_model, query),
            k,
            params=params,
        )

        return [
            (chunks[position], _to_similarity(distance))
            for distance, position in zip(distances[0], positions[0], strict=True)
            if position != -1  # FAISS pads unfilled result slots with -1
        ]

    def _ranked_over(self, index: faiss.Index) -> Sequence[ContextualizedChunk]:
        """The corpus this index numbers, refused where it does not line up.

        An absent corpus reads as empty, like every other artifact a project
        directory may be missing.

        Raises:
            ValueError: If the corpus is not the length the index was built over.
        """
        chunks = self._chunks.chunks
        if chunks and len(chunks) != index.ntotal:
            raise ValueError(
                f"The FAISS index holds {index.ntotal} vectors, but the corpus it "
                f"ranks over has {len(chunks)} chunks, so a vector no longer names "
                "the chunk it was built from. Rebuild the project."
            )
        return chunks

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._index is None or self._index.ntotal == 0
