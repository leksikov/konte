"""FAISS vector store for semantic search using LangChain."""

import asyncio
from collections import defaultdict, deque
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from konte.config import settings
from konte.integrity import sign, verify
from konte.models import Chunk, ContextualizedChunk, MetadataFilter
from konte.stores.base import matches_filter_value

logger = structlog.get_logger()

# What LangChain's save_local writes under index_name="faiss", and what this
# store's signatures cover. The docstore half is a pickle.
SIGNED_FILENAMES = ("faiss.faiss", "faiss.pkl")

_RESERVED_METADATA_FIELDS = frozenset(
    {
        "chunk_id",
        "source",
        "segment_idx",
        "chunk_idx",
        "context_length",
        "context",
        "original_content",
    }
)

# Chunk payload, not filter keys: posting them would hash every chunk's full text.
_UNPOSTED_METADATA_FIELDS = frozenset({"context_length", "context", "original_content"})

# Documents per embedding request, kept well under the OpenAI embedding API's
# 300K-token request limit.
_EMBED_BATCH_SIZE = 100


def _matches_faiss_filter(metadata: dict[str, Any], metadata_filter: MetadataFilter) -> bool:
    """Check if document metadata matches the filter (AND logic).

    Values can be a single value (equality) or a list (match any).
    """
    return all(
        matches_filter_value(metadata.get(key), value) for key, value in metadata_filter.items()
    )


def _to_document(chunk: ContextualizedChunk) -> Document:
    """Flatten a chunk into the LangChain Document the index embeds and stores.

    The context's length rides along so the chunk can be rebuilt from the index
    alone, without a side-car file and without a second copy of its text.
    """
    return Document(
        page_content=chunk.contextualized_content,
        metadata={
            "chunk_id": chunk.chunk.chunk_id,
            "source": chunk.chunk.source,
            "segment_idx": chunk.chunk.segment_idx,
            "chunk_idx": chunk.chunk.chunk_idx,
            "context_length": len(chunk.context),
            **chunk.chunk.metadata,
        },
    )


def _split_content(doc: Document) -> tuple[str, str]:
    """Recover a stored document's (context, original content).

    ContextualizedChunk joins the two with a single space, so the context's
    length locates the cut. An older index carries both halves outright.
    """
    metadata = doc.metadata

    context = metadata.get("context")
    if context is not None:
        return context, metadata.get("original_content", "")

    length = metadata.get("context_length")
    if not isinstance(length, int) or length <= 0:
        return "", doc.page_content
    return doc.page_content[:length], doc.page_content[length + 1 :]


def _from_document(doc: Document) -> ContextualizedChunk:
    """Rebuild a chunk from the flattened metadata written by _to_document()."""
    metadata = doc.metadata
    context, content = _split_content(doc)
    return ContextualizedChunk(
        chunk=Chunk(
            chunk_id=metadata.get("chunk_id", ""),
            content=content,
            source=metadata.get("source", ""),
            segment_idx=metadata.get("segment_idx", 0),
            chunk_idx=metadata.get("chunk_idx", 0),
            metadata={
                key: value
                for key, value in metadata.items()
                if key not in _RESERVED_METADATA_FIELDS
            },
        ),
        context=context,
    )


def _document_batches(
    chunks: Sequence[ContextualizedChunk],
    batch_size: int,
) -> list[list[Document]]:
    """Split the corpus into the units that go out as one embedding request."""
    documents = [_to_document(chunk) for chunk in chunks]
    return [
        documents[start : start + batch_size] for start in range(0, len(documents), batch_size)
    ]


def _page_contents(documents: Sequence[Document]) -> list[str]:
    """Return the text an embedding request carries for one batch."""
    return [document.page_content for document in documents]


# One embedding request that has been sent but not yet folded into the index.
_PendingBatches = deque[tuple[int, list[Document], "asyncio.Future[list[list[float]]]"]]


def _to_similarity(distance: float) -> float:
    """Convert an L2 distance into a 0-1 score, where 1.0 is an exact match."""
    return float(1.0 / (1.0 + distance))


def _lookup_document(vectorstore: FAISS, docstore_id: str) -> Document | None:
    """Fetch a stored Document by docstore id.

    LangChain's docstore returns an error *string* instead of raising when an id
    is missing, so the result has to be type-checked before it is used.
    """
    doc = vectorstore.docstore.search(docstore_id)
    return doc if isinstance(doc, Document) else None


def _document_passes(
    doc: Document,
    metadata_filter: MetadataFilter | None,
    source_filter: str | None,
) -> bool:
    """Check a stored Document against both filter forms."""
    if metadata_filter and not _matches_faiss_filter(doc.metadata, metadata_filter):
        return False
    return not source_filter or source_filter in doc.metadata.get("source", "")


def _select_matching_ids(
    vectorstore: FAISS,
    metadata_filter: MetadataFilter | None,
    source_filter: str | None,
) -> list[int]:
    """Return the FAISS internal ids whose documents satisfy the filters."""
    return [
        faiss_id
        for faiss_id, docstore_id in vectorstore.index_to_docstore_id.items()
        if (doc := _lookup_document(vectorstore, docstore_id)) is not None
        and _document_passes(doc, metadata_filter, source_filter)
    ]


class _FilterIndex:
    """Inverted index over a docstore: field value -> the FAISS ids carrying it."""

    __slots__ = ("_absent", "_all_ids", "_postings", "_unposted")

    def __init__(self, vectorstore: FAISS) -> None:
        postings: dict[str, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
        unposted = set(_UNPOSTED_METADATA_FIELDS)
        all_ids: list[int] = []

        for faiss_id, docstore_id in vectorstore.index_to_docstore_id.items():
            doc = _lookup_document(vectorstore, docstore_id)
            if doc is None:
                continue
            all_ids.append(faiss_id)
            for key, value in doc.metadata.items():
                if key in unposted:
                    continue
                try:
                    postings[key][value].append(faiss_id)
                except TypeError:
                    unposted.add(key)

        self._all_ids = all_ids
        self._postings = {key: dict(values) for key, values in postings.items()}
        self._unposted = unposted
        self._absent: dict[str, frozenset[int]] = {}

    def select(
        self,
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> list[int] | None:
        """Return the ids satisfying both filters, or None if the index cannot answer.

        None (a filtered field has no posting list) means the caller must fall
        back to scanning; an empty list means the filters matched nothing.
        """
        fields = set(metadata_filter or ())
        if source_filter:
            fields.add("source")
        if fields & self._unposted:
            return None

        matches = [self._ids_for(key, value) for key, value in (metadata_filter or {}).items()]
        if source_filter:
            matches.append(self._ids_matching_source(source_filter))
        if not matches:
            return list(self._all_ids)

        # set.intersection walks its receiver, so lead with the smallest.
        matches.sort(key=len)
        return sorted(set.intersection(*matches))

    def _ids_for(self, field: str, expected: Any) -> set[int]:
        """Ids whose field satisfies one filter value; a list means match-any."""
        postings = self._postings.get(field, {})
        wanted = expected if isinstance(expected, list) else (expected,)

        matched: set[int] = set()
        for value in wanted:
            try:
                posted = postings.get(value)
            except TypeError:  # an unhashable filter value equals no posted one
                continue
            if posted:
                matched.update(posted)
            if value is None:
                matched.update(self._absent_from(field))
        return matched

    def _ids_matching_source(self, needle: str) -> set[int]:
        """Ids whose source contains the needle."""
        return set(
            chain.from_iterable(
                ids
                for source, ids in self._postings.get("source", {}).items()
                if isinstance(source, str) and needle in source
            )
        )

    def _absent_from(self, field: str) -> frozenset[int]:
        """Ids of documents carrying no such field — metadata.get() reads them as None."""
        absent = self._absent.get(field)
        if absent is None:
            posted = chain.from_iterable(self._postings.get(field, {}).values())
            absent = frozenset(self._all_ids).difference(posted)
            self._absent[field] = absent
        return absent


class FAISSStore:
    """FAISS vector store for semantic search on contextualized chunks."""

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
        self._vectorstore: FAISS | None = None
        self._filter_index: _FilterIndex | None = None

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
        batches = _document_batches(chunks, batch_size)
        if not batches:
            logger.warning("faiss_build_empty_chunks")
            return

        self._reset()
        for number, batch in enumerate(batches, start=1):
            vectors = self._embeddings.embed_documents(_page_contents(batch))
            self._absorb(number, len(batches), batch, vectors)

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
        batches = _document_batches(chunks, batch_size)
        if not batches:
            logger.warning("faiss_build_empty_chunks")
            return

        if max_concurrency is None:
            max_concurrency = settings.concurrency_limit
        window = max(1, max_concurrency)

        self._reset()
        pending: _PendingBatches = deque()

        try:
            for number, batch in enumerate(batches, start=1):
                if len(pending) == window:
                    await self._absorb_oldest(pending, len(batches))
                request = self._embeddings.aembed_documents(_page_contents(batch))
                pending.append((number, batch, asyncio.ensure_future(request)))
            while pending:
                await self._absorb_oldest(pending, len(batches))
        except BaseException:
            for _, _, task in pending:
                task.cancel()
            await asyncio.gather(*(task for _, _, task in pending), return_exceptions=True)
            raise

        logger.info("faiss_index_built", num_chunks=len(chunks))

    async def _absorb_oldest(self, pending: "_PendingBatches", total_batches: int) -> None:
        """Wait for the longest-outstanding request and fold it into the index."""
        number, batch, task = pending.popleft()
        self._absorb(number, total_batches, batch, await task)

    def _absorb(
        self,
        number: int,
        total_batches: int,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        """Add one embedded batch, creating the vectorstore on the first."""
        if total_batches > 1:
            logger.info("faiss_building_batch", batch=number, total_batches=total_batches)

        pairs = zip((document.page_content for document in documents), vectors, strict=True)
        metadatas = [document.metadata for document in documents]

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_embeddings(
                text_embeddings=pairs,
                embedding=self._embeddings,
                metadatas=metadatas,
            )
        else:
            self._vectorstore.add_embeddings(pairs, metadatas=metadatas)

    def _reset(self) -> None:
        """Drop whatever this store held, so a rebuild starts from nothing."""
        self._vectorstore = None
        self._filter_index = None

    def save(self, directory: Path) -> None:
        """Save FAISS index to disk.

        Args:
            directory: Directory to save index files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._vectorstore is None:
            logger.warning("faiss_save_no_index")
            return

        # LangChain FAISS save_local saves both index and docstore
        self._vectorstore.save_local(str(directory), index_name="faiss")
        sign(directory, SIGNED_FILENAMES)

        logger.info("faiss_index_saved", directory=str(directory))

    def load(self, directory: Path) -> None:
        """Load FAISS index from disk.

        Only the index and its docstore are read; queries rebuild the chunks
        they return from the documents they matched.

        Args:
            directory: Directory containing index files.

        Raises:
            FileNotFoundError: If index files don't exist.
            IntegrityError: If the index is not the one this installation signed.
        """
        directory = Path(directory)

        index_path = directory / "faiss.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # The docstore is a pickle, and reading one runs what it holds, so both
        # files are authenticated before LangChain is pointed at them.
        verify(directory, SIGNED_FILENAMES)

        self._vectorstore = FAISS.load_local(
            str(directory),
            embeddings=self._embeddings,
            index_name="faiss",
            allow_dangerous_deserialization=True,
        )
        self._filter_index = None

        logger.info(
            "faiss_index_loaded",
            directory=str(directory),
            num_chunks=self._vectorstore.index.ntotal,
        )

    def query(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the FAISS index.

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
        if self._vectorstore is None:
            logger.warning("faiss_query_empty_index")
            return []

        k = top_k or settings.DEFAULT_TOP_K

        if metadata_filter or source_filter:
            return self._query_filtered(
                self._vectorstore, query, k, metadata_filter, source_filter
            )

        return [
            (_from_document(doc), _to_similarity(distance))
            for doc, distance in self._vectorstore.similarity_search_with_score(query, k=k)
        ]

    def _query_filtered(
        self,
        vectorstore: FAISS,
        query: str,
        k: int,
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Search only the documents that satisfy the filters.

        Filtering runs before the vector search via a FAISS id selector rather
        than after it, so a restrictive filter still yields k results instead of
        however many happen to survive from a global top-k.
        """
        valid_ids = self._matching_ids(vectorstore, metadata_filter, source_filter)
        if not valid_ids:
            return []

        selector = faiss.IDSelectorArray(np.array(valid_ids, dtype=np.int64))
        query_vector = np.array([self._embeddings.embed_query(query)], dtype=np.float32)
        distances, indices = vectorstore.index.search(
            query_vector,
            min(k, len(valid_ids)),
            params=faiss.SearchParametersIVF(sel=selector),
        )

        results = []
        for distance, faiss_idx in zip(distances[0], indices[0], strict=True):
            if faiss_idx == -1:  # FAISS pads unfilled result slots with -1
                continue
            docstore_id = vectorstore.index_to_docstore_id.get(faiss_idx)
            doc = _lookup_document(vectorstore, docstore_id) if docstore_id else None
            if doc is not None:
                results.append((_from_document(doc), _to_similarity(distance)))
        return results

    def _matching_ids(
        self,
        vectorstore: FAISS,
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> list[int]:
        """Resolve the filters to FAISS ids, indexing the docstore on first use."""
        if self._filter_index is None:
            self._filter_index = _FilterIndex(vectorstore)
        selected = self._filter_index.select(metadata_filter, source_filter)
        if selected is None:
            return _select_matching_ids(vectorstore, metadata_filter, source_filter)
        return selected

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        if self._vectorstore is None:
            return True
        return self._vectorstore.index.ntotal == 0
