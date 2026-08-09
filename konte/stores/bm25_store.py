"""BM25 lexical search store."""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from konte.config import settings
from konte.models import ContextualizedChunk, MetadataFilter
from konte.storage import atomic_writer, write_json
from konte.stores.base import matches_filter_value

logger = structlog.get_logger()


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


def _chunk_field(chunk: ContextualizedChunk, key: str) -> Any:
    """Read a filterable field, preferring the chunk's own columns over metadata."""
    if key == "source":
        return chunk.chunk.source
    if key == "segment_idx":
        return chunk.chunk.segment_idx
    if key == "chunk_idx":
        return chunk.chunk.chunk_idx
    return chunk.chunk.metadata.get(key)


def _matches_filter(chunk: ContextualizedChunk, metadata_filter: MetadataFilter) -> bool:
    """Check if a chunk matches the metadata filter (AND logic).

    Values can be a single value (equality) or a list (match any).

    Args:
        chunk: The chunk to check.
        metadata_filter: Filter with key-value pairs.

    Returns:
        True if all filter conditions match.
    """
    return all(
        matches_filter_value(_chunk_field(chunk, key), value)
        for key, value in metadata_filter.items()
    )


def _normalize(scores: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Rescale raw BM25 scores into 0-1 against the filtered candidate set."""
    value_range = maximum - minimum
    if value_range <= 0:
        return np.zeros_like(scores)
    return np.clip((scores - minimum) / value_range, 0.0, 1.0)


def _rank_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the positions of the k highest scores, best first.

    Splitting on the k-th score rather than argpartition's k-th position is
    what makes ties reproducible: a score tied across the cut resolves to the
    lower position.
    """
    threshold = np.partition(scores, -k)[-k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    top = np.concatenate((above, tied[: k - above.size]))
    return top[np.argsort(-scores[top], kind="stable")]


class BM25Store:
    """BM25 store for lexical search on contextualized chunks."""

    def __init__(self):
        """Initialize BM25 store."""
        self._index: BM25Okapi | None = None
        self._chunks: list[ContextualizedChunk] = []

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build BM25 index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
        """
        if not chunks:
            logger.warning("bm25_build_empty_chunks")
            return

        self._chunks = chunks
        self._index = BM25Okapi([_tokenize(c.contextualized_content) for c in chunks])

        logger.info("bm25_index_built", num_chunks=len(chunks))

    def save(self, directory: Path) -> None:
        """Save BM25 index and metadata to disk.

        The ranking model is pickled because rank_bm25 exposes no serialization
        format of its own; the chunk payload is written as JSON alongside it.
        The tokenized corpus is not stored — BM25Okapi scores from the
        per-document term frequencies it already holds.

        Args:
            directory: Directory to save index files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            logger.warning("bm25_save_no_index")
            return

        with atomic_writer(directory / "bm25.pkl") as handle:
            pickle.dump({"index": self._index}, handle)

        write_json(directory / "bm25_chunks.json", [c.to_storage_dict() for c in self._chunks])

        logger.info("bm25_index_saved", directory=str(directory))

    def load(self, directory: Path) -> None:
        """Load BM25 index and metadata from disk.

        Args:
            directory: Directory containing index files.

        Raises:
            FileNotFoundError: If either index file is missing.
        """
        directory = Path(directory)
        index_path = directory / "bm25.pkl"
        chunks_path = directory / "bm25_chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"BM25 chunk data not found: {chunks_path}")

        # Unpickling executes arbitrary code: only load index dirs this library wrote.
        with index_path.open("rb") as f:
            data = pickle.load(f)
        self._index = data["index"]

        self._chunks = [
            ContextualizedChunk.from_storage_dict(item)
            for item in json.loads(chunks_path.read_text(encoding="utf-8"))
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
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the BM25 index.

        Args:
            query: Query string.
            top_k: Number of results to return. Defaults to settings.DEFAULT_TOP_K.
            metadata_filter: Filter results by metadata (equality match, AND logic).
                Example: {"source": "doc.pdf", "year": 2024}
            source_filter: Substring match on chunk source field.
                Example: "JOHNSON" matches "JOHNSON_JOHNSON_2022_10K.md"

        Returns:
            List of (chunk, score) tuples, sorted by score descending. Scores are
            normalized to 0-1 across the surviving candidates, so they are only
            comparable within one response.
        """
        if self._index is None or not self._chunks:
            logger.warning("bm25_query_empty_index")
            return []

        candidates = self._filter_indices(metadata_filter, source_filter)
        if candidates is not None and not candidates.size:
            return []

        # IDF is computed over the whole corpus, so scoring cannot be restricted
        # to the candidates; only the ranking that follows is.
        scores = self._index.get_scores(_tokenize(query))
        if candidates is not None:
            scores = scores[candidates]

        k = min(top_k or settings.DEFAULT_TOP_K, scores.size)
        if k <= 0:
            return []

        ranked = _rank_top_k(scores, k)
        normalized = _normalize(scores[ranked], scores.min(), scores.max())
        positions = ranked if candidates is None else candidates[ranked]

        return [
            (self._chunks[position], score)
            for position, score in zip(positions.tolist(), normalized.tolist(), strict=True)
        ]

    def _filter_indices(
        self,
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> np.ndarray | None:
        """Return corpus positions satisfying both filters, or None for no filter.

        Filtering runs before ranking, so a restrictive filter still yields
        top_k results instead of however many survive a global top-k.
        """
        if not metadata_filter and not source_filter:
            return None

        matched = enumerate(self._chunks)
        if metadata_filter:
            matched = ((i, c) for i, c in matched if _matches_filter(c, metadata_filter))
        if source_filter:
            matched = ((i, c) for i, c in matched if source_filter in c.chunk.source)
        return np.fromiter((i for i, _ in matched), dtype=np.intp)

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._index is None or not self._chunks
