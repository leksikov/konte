"""FAISS vector store for semantic search using LangChain."""

from pathlib import Path

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from konte.config import settings
from konte.models import Chunk, ContextualizedChunk

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

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build FAISS index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
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
                },
            )
            documents.append(doc)

        # Build FAISS index using LangChain
        self._vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self._embeddings,
        )

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
        if self._vectorstore.docstore:
            for doc_id in self._vectorstore.index_to_docstore_id.values():
                doc = self._vectorstore.docstore.search(doc_id)
                if doc and hasattr(doc, "metadata"):
                    chunk = Chunk(
                        chunk_id=doc.metadata.get("chunk_id", ""),
                        content=doc.metadata.get("original_content", ""),
                        source=doc.metadata.get("source", ""),
                        segment_idx=doc.metadata.get("segment_idx", 0),
                        chunk_idx=doc.metadata.get("chunk_idx", 0),
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
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the FAISS index.

        Args:
            query: Query string.
            top_k: Number of results to return. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            List of (chunk, score) tuples, sorted by score descending.
        """
        if self._vectorstore is None:
            logger.warning("faiss_query_empty_index")
            return []

        k = top_k or settings.DEFAULT_TOP_K

        # LangChain similarity_search_with_score returns (doc, score) tuples
        # Score is L2 distance (lower = more similar)
        results_with_scores = self._vectorstore.similarity_search_with_score(query, k=k)

        results = []
        for doc, distance in results_with_scores:
            # Convert L2 distance to similarity score (0-1 range)
            # L2 distance: 0 = identical, larger = more different
            # Use exponential decay: score = exp(-distance/2)
            similarity = 1.0 / (1.0 + distance)

            chunk = Chunk(
                chunk_id=doc.metadata.get("chunk_id", ""),
                content=doc.metadata.get("original_content", ""),
                source=doc.metadata.get("source", ""),
                segment_idx=doc.metadata.get("segment_idx", 0),
                chunk_idx=doc.metadata.get("chunk_idx", 0),
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
