"""FAISS vector store for semantic search using LangChain."""

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from konte.config import settings
from konte.models import Chunk, ContextualizedChunk, MetadataFilter
from konte.stores.base import matches_filter_value

logger = structlog.get_logger()

_RESERVED_METADATA_FIELDS = frozenset(
    {"chunk_id", "source", "segment_idx", "chunk_idx", "context", "original_content"}
)


def _matches_faiss_filter(metadata: dict[str, Any], metadata_filter: MetadataFilter) -> bool:
    """Check if document metadata matches the filter (AND logic).

    Values can be a single value (equality) or a list (match any).
    """
    return all(
        matches_filter_value(metadata.get(key), value) for key, value in metadata_filter.items()
    )


def _to_document(chunk: ContextualizedChunk) -> Document:
    """Flatten a chunk into the LangChain Document the index embeds and stores.

    The original content and the generated context both ride along in metadata
    so the chunk can be rebuilt from the index alone, without a side-car file.
    """
    return Document(
        page_content=chunk.contextualized_content,
        metadata={
            "chunk_id": chunk.chunk.chunk_id,
            "source": chunk.chunk.source,
            "segment_idx": chunk.chunk.segment_idx,
            "chunk_idx": chunk.chunk.chunk_idx,
            "context": chunk.context,
            "original_content": chunk.chunk.content,
            **chunk.chunk.metadata,
        },
    )


def _from_document(doc: Document) -> ContextualizedChunk:
    """Rebuild a chunk from the flattened metadata written by _to_document()."""
    metadata = doc.metadata
    return ContextualizedChunk(
        chunk=Chunk(
            chunk_id=metadata.get("chunk_id", ""),
            content=metadata.get("original_content", ""),
            source=metadata.get("source", ""),
            segment_idx=metadata.get("segment_idx", 0),
            chunk_idx=metadata.get("chunk_idx", 0),
            metadata={
                key: value
                for key, value in metadata.items()
                if key not in _RESERVED_METADATA_FIELDS
            },
        ),
        context=metadata.get("context", ""),
    )


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

    def build_index(
        self,
        chunks: list[ContextualizedChunk],
        batch_size: int = 100,
    ) -> None:
        """Build FAISS index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
            batch_size: Number of documents to embed per request. Kept well
                under the OpenAI embedding API's 300K-token request limit.
        """
        if not chunks:
            logger.warning("faiss_build_empty_chunks")
            return

        documents = [_to_document(chunk) for chunk in chunks]
        batches = [
            documents[start : start + batch_size]
            for start in range(0, len(documents), batch_size)
        ]

        if len(batches) > 1:
            logger.info("faiss_building_batch", batch=1, total_batches=len(batches))
        self._vectorstore = FAISS.from_documents(
            documents=batches[0],
            embedding=self._embeddings,
        )
        for batch_num, batch in enumerate(batches[1:], start=2):
            logger.info("faiss_building_batch", batch=batch_num, total_batches=len(batches))
            self._vectorstore.add_documents(batch)

        logger.info(
            "faiss_index_built",
            num_chunks=len(chunks),
        )

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

        logger.info("faiss_index_saved", directory=str(directory))

    def load(self, directory: Path) -> None:
        """Load FAISS index from disk.

        Only the index and its docstore are read; queries rebuild the chunks
        they return from the documents they matched.

        Args:
            directory: Directory containing index files.

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        directory = Path(directory)

        index_path = directory / "faiss.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self._vectorstore = FAISS.load_local(
            str(directory),
            embeddings=self._embeddings,
            index_name="faiss",
            # The docstore is a pickle: only load index dirs this library wrote.
            allow_dangerous_deserialization=True,
        )

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
        valid_ids = _select_matching_ids(vectorstore, metadata_filter, source_filter)
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

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        if self._vectorstore is None:
            return True
        return self._vectorstore.index.ntotal == 0
