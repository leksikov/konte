"""FAISS vector store for semantic search."""

import json
from pathlib import Path

import faiss
import numpy as np
import structlog
from langchain_openai import OpenAIEmbeddings

from konte.models import Chunk, ContextualizedChunk
from konte.settings import settings

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
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[ContextualizedChunk] = []
        self._chunk_ids: list[str] = []

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build FAISS index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
        """
        if not chunks:
            logger.warning("faiss_build_empty_chunks")
            return

        self._chunks = chunks
        self._chunk_ids = [c.chunk.chunk_id for c in chunks]

        # Get embeddings for contextualized content
        texts = [c.contextualized_content for c in chunks]
        embeddings = self._embeddings.embed_documents(texts)

        # Convert to numpy array and normalize for cosine similarity
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embedding_matrix)

        # Create index using inner product (cosine similarity after normalization)
        dimension = embedding_matrix.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embedding_matrix)

        logger.info(
            "faiss_index_built",
            num_chunks=len(chunks),
            dimension=dimension,
        )

    def save(self, directory: Path) -> None:
        """Save FAISS index and metadata to disk.

        Args:
            directory: Directory to save index files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            logger.warning("faiss_save_no_index")
            return

        # Save FAISS index
        index_path = directory / "faiss.index"
        faiss.write_index(self._index, str(index_path))

        # Save chunk IDs mapping
        ids_path = directory / "faiss_ids.json"
        ids_path.write_text(json.dumps(self._chunk_ids), encoding="utf-8")

        # Save chunks data
        chunks_path = directory / "faiss_chunks.json"
        chunks_data = [
            {
                "chunk": c.chunk.model_dump(),
                "context": c.context,
            }
            for c in self._chunks
        ]
        chunks_path.write_text(json.dumps(chunks_data), encoding="utf-8")

        logger.info("faiss_index_saved", directory=str(directory))

    def load(self, directory: Path) -> None:
        """Load FAISS index and metadata from disk.

        Args:
            directory: Directory containing index files.
        """
        directory = Path(directory)

        index_path = directory / "faiss.index"
        ids_path = directory / "faiss_ids.json"
        chunks_path = directory / "faiss_chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # Load FAISS index
        self._index = faiss.read_index(str(index_path))

        # Load chunk IDs
        self._chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))

        # Load chunks data
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._chunks = [
            ContextualizedChunk(
                chunk=Chunk(**item["chunk"]),
                context=item["context"],
            )
            for item in chunks_data
        ]

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
        if self._index is None or self._index.ntotal == 0:
            logger.warning("faiss_query_empty_index")
            return []

        k = top_k or settings.DEFAULT_TOP_K
        k = min(k, len(self._chunks))

        # Get query embedding
        query_embedding = self._embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._chunks):
                # Convert inner product score to 0-1 range
                # Inner product of normalized vectors gives cosine similarity (-1 to 1)
                # Map to 0-1 range
                normalized_score = (score + 1) / 2
                results.append((self._chunks[idx], float(normalized_score)))

        return results

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._index is None or self._index.ntotal == 0
