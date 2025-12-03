"""BM25 lexical search store."""

import json
import pickle
from pathlib import Path

import structlog
from rank_bm25 import BM25Okapi

from konte.models import Chunk, ContextualizedChunk
from konte.settings import settings

logger = structlog.get_logger()


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


class BM25Store:
    """BM25 store for lexical search on contextualized chunks."""

    def __init__(self):
        """Initialize BM25 store."""
        self._index: BM25Okapi | None = None
        self._chunks: list[ContextualizedChunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build BM25 index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
        """
        if not chunks:
            logger.warning("bm25_build_empty_chunks")
            return

        self._chunks = chunks

        # Tokenize contextualized content
        self._tokenized_corpus = [
            _tokenize(c.contextualized_content)
            for c in chunks
        ]

        self._index = BM25Okapi(self._tokenized_corpus)

        logger.info("bm25_index_built", num_chunks=len(chunks))

    def save(self, directory: Path) -> None:
        """Save BM25 index and metadata to disk.

        Args:
            directory: Directory to save index files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            logger.warning("bm25_save_no_index")
            return

        # Save BM25 index using pickle
        index_path = directory / "bm25.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(
                {
                    "index": self._index,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )

        # Save chunks data as JSON
        chunks_path = directory / "bm25_chunks.json"
        chunks_data = [
            {
                "chunk": c.chunk.model_dump(),
                "context": c.context,
            }
            for c in self._chunks
        ]
        chunks_path.write_text(json.dumps(chunks_data), encoding="utf-8")

        logger.info("bm25_index_saved", directory=str(directory))

    def load(self, directory: Path) -> None:
        """Load BM25 index and metadata from disk.

        Args:
            directory: Directory containing index files.
        """
        directory = Path(directory)

        index_path = directory / "bm25.pkl"
        chunks_path = directory / "bm25_chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        # Load BM25 index
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            self._index = data["index"]
            self._tokenized_corpus = data["tokenized_corpus"]

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
            "bm25_index_loaded",
            directory=str(directory),
            num_chunks=len(self._chunks),
        )

    def query(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the BM25 index.

        Args:
            query: Query string.
            top_k: Number of results to return. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            List of (chunk, score) tuples, sorted by score descending.
        """
        if self._index is None or not self._chunks:
            logger.warning("bm25_query_empty_index")
            return []

        k = top_k or settings.DEFAULT_TOP_K
        k = min(k, len(self._chunks))

        # Tokenize query
        tokenized_query = _tokenize(query)

        # Get BM25 scores
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Normalize scores to 0-1 range
        # BM25 scores can be negative for non-matching terms, clamp to 0
        min_score = min(scores)
        max_score = max(scores)

        # Shift to positive range if needed, then normalize
        score_range = max_score - min_score if max_score != min_score else 1.0

        results = []
        for idx in top_indices:
            # Shift and normalize to 0-1
            if score_range > 0:
                normalized_score = (scores[idx] - min_score) / score_range
            else:
                normalized_score = 0.0
            # Ensure score is in valid range
            normalized_score = max(0.0, min(1.0, normalized_score))
            results.append((self._chunks[idx], float(normalized_score)))

        return results

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._index is None or not self._chunks
