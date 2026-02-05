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
from konte.models import Chunk, ContextualizedChunk


def _matches_faiss_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
    """Check if document metadata matches the filter (AND logic)."""
    for key, value in metadata_filter.items():
        if metadata.get(key) != value:
            return False
    return True

logger = structlog.get_logger()


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
        self._embeddings = OpenAIEmbeddings(model=self._embedding_model)
        self._vectorstore: FAISS | None = None
        self._chunks: list[ContextualizedChunk] = []

    def build_index(
        self,
        chunks: list[ContextualizedChunk],
        batch_size: int = 100,
    ) -> None:
        """Build FAISS index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
            batch_size: Number of documents to embed per batch (default: 500).
        """
        if not chunks:
            logger.warning("faiss_build_empty_chunks")
            return

        self._chunks = chunks

        # Create LangChain Documents with metadata
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.contextualized_content,
                metadata={
                    "chunk_id": chunk.chunk.chunk_id,
                    "source": chunk.chunk.source,
                    "segment_idx": chunk.chunk.segment_idx,
                    "chunk_idx": chunk.chunk.chunk_idx,
                    "context": chunk.context,
                    "original_content": chunk.chunk.content,
                    **chunk.chunk.metadata,  # Include custom metadata
                },
            )
            documents.append(doc)

        # Build FAISS index in batches to avoid token limit errors
        # OpenAI embedding API has 300K token limit per request
        if len(documents) <= batch_size:
            self._vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self._embeddings,
            )
        else:
            # Build first batch
            logger.info(
                "faiss_building_batch",
                batch=1,
                total_batches=(len(documents) + batch_size - 1) // batch_size,
            )
            self._vectorstore = FAISS.from_documents(
                documents=documents[:batch_size],
                embedding=self._embeddings,
            )

            # Add remaining batches
            for i in range(batch_size, len(documents), batch_size):
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                logger.info(
                    "faiss_building_batch",
                    batch=batch_num,
                    total_batches=total_batches,
                )
                batch_docs = documents[i : i + batch_size]
                self._vectorstore.add_documents(batch_docs)

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

        Args:
            directory: Directory containing index files.

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        directory = Path(directory)

        # Check if index file exists
        index_path = directory / "faiss.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # LangChain FAISS load_local
        self._vectorstore = FAISS.load_local(
            str(directory),
            embeddings=self._embeddings,
            index_name="faiss",
            allow_dangerous_deserialization=True,  # Required for loading pickle
        )

        # Rebuild chunks from vectorstore docstore
        self._chunks = []
        standard_fields = {"chunk_id", "source", "segment_idx", "chunk_idx", "context", "original_content"}
        if self._vectorstore.docstore:
            for doc_id in self._vectorstore.index_to_docstore_id.values():
                doc = self._vectorstore.docstore.search(doc_id)
                if doc and hasattr(doc, "metadata"):
                    custom_metadata = {k: v for k, v in doc.metadata.items() if k not in standard_fields}
                    chunk = Chunk(
                        chunk_id=doc.metadata.get("chunk_id", ""),
                        content=doc.metadata.get("original_content", ""),
                        source=doc.metadata.get("source", ""),
                        segment_idx=doc.metadata.get("segment_idx", 0),
                        chunk_idx=doc.metadata.get("chunk_idx", 0),
                        metadata=custom_metadata,
                    )
                    ctx_chunk = ContextualizedChunk(
                        chunk=chunk,
                        context=doc.metadata.get("context", ""),
                    )
                    self._chunks.append(ctx_chunk)

        logger.info(
            "faiss_index_loaded",
            directory=str(directory),
            num_chunks=len(self._chunks),
        )

    def query(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
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

        # Pre-retrieval filtering: use FAISS IDSelector to search only matching documents
        if metadata_filter or source_filter:
            # Find FAISS internal IDs that match the filter
            valid_faiss_ids = []
            for faiss_id, docstore_id in self._vectorstore.index_to_docstore_id.items():
                doc = self._vectorstore.docstore.search(docstore_id)
                if doc and hasattr(doc, "metadata"):
                    if metadata_filter and not _matches_faiss_filter(doc.metadata, metadata_filter):
                        continue
                    if source_filter and source_filter not in doc.metadata.get("source", ""):
                        continue
                    valid_faiss_ids.append(faiss_id)

            if not valid_faiss_ids:
                return []

            # Use FAISS IDSelectorArray for pre-filtered search
            id_selector = faiss.IDSelectorArray(np.array(valid_faiss_ids, dtype=np.int64))
            search_params = faiss.SearchParametersIVF(sel=id_selector)

            # Embed the query
            query_embedding = self._embeddings.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)

            # Search only among filtered IDs
            search_k = min(k, len(valid_faiss_ids))
            distances, indices = self._vectorstore.index.search(
                query_vector, search_k, params=search_params
            )

            # Build results from search output
            results = []
            for i, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
                if faiss_idx == -1:  # No result
                    continue

                docstore_id = self._vectorstore.index_to_docstore_id.get(faiss_idx)
                if not docstore_id:
                    continue

                doc = self._vectorstore.docstore.search(docstore_id)
                if not doc:
                    continue

                similarity = 1.0 / (1.0 + distance)
                standard_fields = {"chunk_id", "source", "segment_idx", "chunk_idx", "context", "original_content"}
                custom_metadata = {k: v for k, v in doc.metadata.items() if k not in standard_fields}

                chunk = Chunk(
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    content=doc.metadata.get("original_content", ""),
                    source=doc.metadata.get("source", ""),
                    segment_idx=doc.metadata.get("segment_idx", 0),
                    chunk_idx=doc.metadata.get("chunk_idx", 0),
                    metadata=custom_metadata,
                )
                ctx_chunk = ContextualizedChunk(
                    chunk=chunk,
                    context=doc.metadata.get("context", ""),
                )
                results.append((ctx_chunk, float(similarity)))

            return results

        # No filter: use standard LangChain search
        results_with_scores = self._vectorstore.similarity_search_with_score(query, k=k)

        results = []
        for doc, distance in results_with_scores:
            similarity = 1.0 / (1.0 + distance)
            standard_fields = {"chunk_id", "source", "segment_idx", "chunk_idx", "context", "original_content"}
            custom_metadata = {k: v for k, v in doc.metadata.items() if k not in standard_fields}

            chunk = Chunk(
                chunk_id=doc.metadata.get("chunk_id", ""),
                content=doc.metadata.get("original_content", ""),
                source=doc.metadata.get("source", ""),
                segment_idx=doc.metadata.get("segment_idx", 0),
                chunk_idx=doc.metadata.get("chunk_idx", 0),
                metadata=custom_metadata,
            )
            ctx_chunk = ContextualizedChunk(
                chunk=chunk,
                context=doc.metadata.get("context", ""),
            )
            results.append((ctx_chunk, float(similarity)))

        return results

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        if self._vectorstore is None:
            return True
        return self._vectorstore.index.ntotal == 0
