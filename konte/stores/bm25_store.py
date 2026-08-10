"""BM25 lexical search store."""

import pickle
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from konte.config import settings
from konte.models import ContextualizedChunk, MetadataFilter
from konte.storage import atomic_writer
from konte.stores.base import matches_filter_value

logger = structlog.get_logger()

# Returns the chunks an index was built over, in index order.
Corpus = Callable[[], Sequence[ContextualizedChunk]]

LEGACY_CHUNKS_FILENAME = "bm25_chunks.json"


class LexicalResults(NamedTuple):
    """Ranked chunks, and how much of the query each one carries.

    `results` scores are normalized against the candidates they were drawn
    from, so the top chunk always reads 1.0 whatever it matched. `coverage`
    is absolute: the share of the query's terms, weighted by how rare each is.
    """

    results: list[tuple[ContextualizedChunk, float]]
    coverage: dict[str, float]


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


def _filter_indices(
    chunks: Sequence[ContextualizedChunk],
    metadata_filter: MetadataFilter | None,
    source_filter: str | None,
) -> np.ndarray | None:
    """Return corpus positions satisfying both filters, or None for no filter.

    Filtering runs before ranking, so a restrictive filter still yields top_k
    results instead of however many survive a global top-k.
    """
    if not metadata_filter and not source_filter:
        return None

    matched = enumerate(chunks)
    if metadata_filter:
        matched = ((i, c) for i, c in matched if _matches_filter(c, metadata_filter))
    if source_filter:
        matched = ((i, c) for i, c in matched if source_filter in c.chunk.source)
    return np.fromiter((i for i, _ in matched), dtype=np.intp)


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


def _term_weights(index: BM25Okapi, tokens: Sequence[str], unseen: float) -> dict[str, float]:
    """Weight each distinct query term by how much it narrows the corpus.

    An unindexed term weighs as much as the rarest indexed one, so naming
    something absent cannot read as a full match. Negative IDF — a term in
    most documents, or any term of a corpus too small to separate — clamps away.
    """
    idf = index.idf
    return {token: max(idf.get(token, unseen), 0.0) for token in dict.fromkeys(tokens)}


def _coverage(index: BM25Okapi, weights: dict[str, float], positions: Sequence[int]) -> list[float]:
    """Score how much of the weighted query each of the given documents carries."""
    if not weights:
        return [0.0] * len(positions)

    total = sum(weights.values())
    if total <= 0:  # no weight survived the clamp; fall back to counting terms
        weights = dict.fromkeys(weights, 1.0)
        total = float(len(weights))

    frequencies = index.doc_freqs
    return [
        sum(weight for term, weight in weights.items() if term in frequencies[position]) / total
        for position in positions
    ]


class BM25Store:
    """BM25 store for lexical search on contextualized chunks."""

    def __init__(self):
        """Initialize BM25 store."""
        self._index: BM25Okapi | None = None
        self._corpus: Corpus = list  # nothing indexed yet
        self._unseen_idf = 0.0

    def _attach(self, index: BM25Okapi, corpus: Corpus) -> None:
        """Bind a ranking model to the chunks it was built over.

        The rarest term's IDF is read here rather than per query: scanning the
        vocabulary is not free on a corpus worth ranking.
        """
        self._index = index
        self._corpus = corpus
        self._unseen_idf = max(index.idf.values(), default=0.0)

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build BM25 index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
        """
        if not chunks:
            logger.warning("bm25_build_empty_chunks")
            return

        index = BM25Okapi([_tokenize(c.contextualized_content) for c in chunks])
        self._attach(index, lambda: chunks)

        logger.info("bm25_index_built", num_chunks=len(chunks))

    def save(self, directory: Path) -> None:
        """Save the BM25 ranking model to disk.

        The model is pickled because rank_bm25 exposes no serialization format
        of its own. The tokenized corpus is not stored — BM25Okapi scores from
        the per-document term frequencies it already holds.

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

        # An earlier version kept a second copy of the corpus here.
        (directory / LEGACY_CHUNKS_FILENAME).unlink(missing_ok=True)

        logger.info("bm25_index_saved", directory=str(directory))

    def load(self, directory: Path, corpus: Corpus) -> None:
        """Load the BM25 ranking model from disk and bind it to a corpus.

        Args:
            directory: Directory containing index files.
            corpus: Returns the chunks the index was built over, in index
                order. Called on the first query, not here.

        Raises:
            FileNotFoundError: If the index file is missing.
        """
        directory = Path(directory)
        index_path = directory / "bm25.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        # Unpickling executes arbitrary code: only load index dirs this library wrote.
        with index_path.open("rb") as f:
            data = pickle.load(f)
        self._attach(data["index"], cache(corpus))

        logger.info("bm25_index_loaded", directory=str(directory))

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
            comparable within one response. query_with_coverage() is the variant
            that also reports how much of the query each chunk matched.
        """
        return self.query_with_coverage(query, top_k, metadata_filter, source_filter).results

    def query_with_coverage(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> LexicalResults:
        """Query the BM25 index, reporting matched query coverage alongside the ranking.

        Coverage reads the term frequencies the ranking model already holds, so
        it costs a few dictionary lookups per returned chunk.

        Args:
            query: Query string.
            top_k: Number of results to return. Defaults to settings.DEFAULT_TOP_K.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            LexicalResults pairing the ranked chunks with their coverage.
        """
        chunks = self._corpus()
        if self._index is None or not chunks:
            logger.warning("bm25_query_empty_index")
            return LexicalResults([], {})

        candidates = _filter_indices(chunks, metadata_filter, source_filter)
        if candidates is not None and not candidates.size:
            return LexicalResults([], {})

        # IDF is computed over the whole corpus, so scoring cannot be restricted
        # to the candidates; only the ranking that follows is.
        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)
        if candidates is not None:
            scores = scores[candidates]

        k = min(top_k or settings.DEFAULT_TOP_K, scores.size)
        if k <= 0:
            return LexicalResults([], {})

        ranked = _rank_top_k(scores, k)
        normalized = _normalize(scores[ranked], scores.min(), scores.max())
        positions = (ranked if candidates is None else candidates[ranked]).tolist()

        results = [
            (chunks[position], score)
            for position, score in zip(positions, normalized.tolist(), strict=True)
        ]
        weights = _term_weights(self._index, tokens, self._unseen_idf)
        return LexicalResults(
            results,
            {
                chunk.chunk.chunk_id: matched
                for (chunk, _), matched in zip(
                    results, _coverage(self._index, weights, positions), strict=True
                )
            },
        )

    @property
    def is_empty(self) -> bool:
        """bool: True when there is no index to rank against.

        Answered from the ranking model alone: counting the corpus here would
        defeat deferring it.
        """
        return self._index is None
