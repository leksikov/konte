"""BM25 lexical search store."""

import json
import pickle
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from konte.config import settings
from konte.models import Chunk, ContextualizedChunk

logger = structlog.get_logger()


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


def _matches_filter(
    chunk: ContextualizedChunk, metadata_filter: dict[str, Any]
) -> bool:
    """Check if a chunk matches the metadata filter (AND logic).

    Args:
        chunk: The chunk to check.
        metadata_filter: Filter with key-value pairs (equality match).

    Returns:
        True if all filter conditions match.
    """
    chunk_metadata = chunk.chunk.metadata
    # Also check standard fields: source, segment_idx, chunk_idx
    for key, value in metadata_filter.items():
        if key == "source":
            if chunk.chunk.source != value:
                return False
        elif key == "segment_idx":
            if chunk.chunk.segment_idx != value:
                return False
        elif key == "chunk_idx":
            if chunk.chunk.chunk_idx != value:
                return False
        else:
            # Check custom metadata
            if chunk_metadata.get(key) != value:
                return False
    return True


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
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the BM25 index.

        Args:
            query: Query string.
            top_k: Number of results to return. Defaults to settings.DEFAULT_TOP_K.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "year": 2024}

        Returns:
            List of (chunk, score) tuples, sorted by score descending.
        """
        if self._index is None or not self._chunks:
            logger.warning("bm25_query_empty_index")
            return []

        k = top_k or settings.DEFAULT_TOP_K

        # If filtering, get more results initially to ensure enough after filtering
        initial_k = k * 3 if metadata_filter else k
        initial_k = min(initial_k, len(self._chunks))

        # Tokenize query
        tokenized_query = _tokenize(query)

        # Get BM25 scores
        scores = self._index.get_scores(tokenized_query)

        # Get top indices (more if filtering)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:initial_k]

        # Normalize scores to 0-1 range
        # BM25 scores can be negative for non-matching terms, clamp to 0
        min_score = min(scores)
        max_score = max(scores)

        # Shift to positive range if needed, then normalize
        score_range = max_score - min_score if max_score != min_score else 1.0

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]

            # Apply metadata filter if provided
            if metadata_filter and not _matches_filter(chunk, metadata_filter):
                continue

            # Shift and normalize to 0-1
            if score_range > 0:
                normalized_score = (scores[idx] - min_score) / score_range
            else:
                normalized_score = 0.0
            # Ensure score is in valid range
            normalized_score = max(0.0, min(1.0, normalized_score))
            results.append((chunk, float(normalized_score)))

            # Stop if we have enough results
            if len(results) >= k:
                break

        return results

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._index is None or not self._chunks
