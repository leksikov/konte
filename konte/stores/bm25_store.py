"""BM25 lexical search store."""

import pickle
import re
from collections import Counter, defaultdict
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

# A stale index stops matching silently, not loudly. Bump on any _tokenize change.
_TOKENIZER_VERSION = 2

# Scripts with no word spacing: "탈수기는" is a noun plus its particle, so these
# index as character bigrams the way Elasticsearch's CJK analyzer does.
_CJK = (
    "ᄀ-ᇿ"  # Hangul Jamo
    "぀-ヿ"  # Hiragana, Katakana
    "㄰-㆏"  # Hangul Compatibility Jamo
    "ㇰ-ㇿ"  # Katakana Phonetic Extensions
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs
    "ꥠ-꥿"  # Hangul Jamo Extended-A
    "가-힣"  # Hangul Syllables
    "ힰ-퟿"  # Hangul Jamo Extended-B
    "豈-﫿"  # CJK Compatibility Ideographs
    "ｦ-ﾝ"  # Halfwidth Katakana
    "\U00020000-\U0002fa1f"  # Extensions B-F, Compatibility Supplement
)

# Interior punctuation only: "8542.31" survives, "FY2022," sheds its comma.
_WORD_RE = re.compile(f"[^\\W_{_CJK}]+(?:[.,'’][^\\W_{_CJK}]+)*")
_CJK_RE = re.compile(f"[{_CJK}]+")


class LexicalResults(NamedTuple):
    """Ranked chunks, and how much of the query each one carries.

    `results` scores are normalized against the candidates they were drawn
    from, so the top chunk always reads 1.0 whatever it matched. `coverage`
    is absolute: the share of the query's terms, weighted by how rare each is.
    """

    results: list[tuple[ContextualizedChunk, float]]
    coverage: dict[str, float]


def _tokenize(text: str) -> list[str]:
    """Split text into the terms BM25 indexes and matches.

    Args:
        text: Text to tokenize.

    Returns:
        Lowercase tokens. BM25 reads them as a bag, so the CJK bigrams trail
        the words rather than sitting where they were drawn from.
    """
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    if text.isascii():  # no CJK possible
        return tokens

    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens += [run[i : i + 2] for i in range(len(run) - 1)]
    return tokens


# Read off the chunk, never from metadata of the same name.
_CHUNK_COLUMNS = frozenset({"source", "segment_idx", "chunk_idx"})


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


class _FilterIndex:
    """Inverted index over the corpus: field value -> the positions carrying it.

    Filtering otherwise walks every chunk in Python on every query; the walk
    happens once here instead, on the first filtered query.
    """

    __slots__ = ("_absent", "_postings", "_size", "_unposted")

    def __init__(self, chunks: Sequence[ContextualizedChunk]) -> None:
        postings: dict[str, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
        unposted: set[str] = set()

        for position, contextualized in enumerate(chunks):
            chunk = contextualized.chunk
            postings["source"][chunk.source].append(position)
            postings["segment_idx"][chunk.segment_idx].append(position)
            postings["chunk_idx"][chunk.chunk_idx].append(position)
            for key, value in chunk.metadata.items():
                if key in _CHUNK_COLUMNS or key in unposted:
                    continue
                try:
                    postings[key][value].append(position)
                except TypeError:
                    unposted.add(key)

        self._size = len(chunks)
        # Appended in corpus order, so posting lists ascend like the scan's.
        self._postings = {
            key: {value: np.array(found, dtype=np.intp) for value, found in values.items()}
            for key, values in postings.items()
        }
        self._unposted = unposted
        self._absent: dict[str, np.ndarray] = {}

    def select(
        self,
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> np.ndarray | None:
        """Return the positions satisfying both filters, or None to fall back.

        None means a filtered field is unposted and the caller must scan; an
        empty array means the filters matched nothing.
        """
        if set(metadata_filter or ()) & self._unposted:
            return None

        matched = [
            self._positions_for(key, value) for key, value in (metadata_filter or {}).items()
        ]
        if source_filter:
            matched.append(self._positions_matching_source(source_filter))
        if len(matched) == 1:
            return matched[0]

        # A mask per field beats intersecting position lists, which re-sorts.
        mask = np.ones(self._size, dtype=bool)
        for positions in matched:
            keep = np.zeros(self._size, dtype=bool)
            keep[positions] = True
            mask &= keep
        return np.flatnonzero(mask)

    def _positions_for(self, field: str, expected: Any) -> np.ndarray:
        """Positions whose field satisfies one filter value; a list means match-any."""
        postings = self._postings.get(field, {})
        wanted = expected if isinstance(expected, list) else (expected,)

        found: list[np.ndarray] = []
        for value in wanted:
            try:
                posted = postings.get(value)
            except TypeError:  # an unhashable filter value equals no posted one
                continue
            if posted is not None:
                found.append(posted)
            if value is None:
                found.append(self._absent_from(field))
        return self._union(found)

    def _positions_matching_source(self, needle: str) -> np.ndarray:
        """Positions whose source contains the needle."""
        return self._union(
            [
                positions
                for source, positions in self._postings.get("source", {}).items()
                if needle in source
            ]
        )

    def _union(self, found: list[np.ndarray]) -> np.ndarray:
        """Merge posting lists of one field, which cannot overlap, back into order."""
        if not found:
            return np.empty(0, dtype=np.intp)
        if len(found) == 1:
            return found[0]

        mask = np.zeros(self._size, dtype=bool)
        for positions in found:
            mask[positions] = True
        return np.flatnonzero(mask)

    def _absent_from(self, field: str) -> np.ndarray:
        """Positions of chunks carrying no such field — _chunk_field reads them as None."""
        absent = self._absent.get(field)
        if absent is None:
            mask = np.ones(self._size, dtype=bool)
            for positions in self._postings.get(field, {}).values():
                mask[positions] = False
            self._absent[field] = absent = np.flatnonzero(mask)
        return absent


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


class _Postings(NamedTuple):
    """Which documents hold each term, as spans of two corpus-wide arrays."""

    documents: np.ndarray  # document positions, grouped by term
    frequencies: np.ndarray  # term counts, aligned with documents
    spans: dict[str, tuple[int, int]]  # term -> half-open span into both


def _invert(index: BM25Okapi) -> _Postings:
    """Group the per-document term frequencies by term.

    BM25Okapi.get_scores walks the whole corpus once per query token, and
    bigrams make a CJK query several times longer.
    """
    grouped: dict[str, tuple[list[int], list[int]]] = {}
    for position, frequencies in enumerate(index.doc_freqs):
        for term, count in frequencies.items():
            entry = grouped.get(term)
            if entry is None:
                grouped[term] = entry = ([], [])
            entry[0].append(position)
            entry[1].append(count)

    total = sum(len(documents) for documents, _ in grouped.values())
    documents = np.empty(total, dtype=np.int32)
    counts = np.empty(total, dtype=np.float32)
    spans = {}

    start = 0
    for term, (term_documents, term_counts) in grouped.items():
        stop = start + len(term_documents)
        documents[start:stop] = term_documents
        counts[start:stop] = term_counts
        spans[term] = (start, stop)
        start = stop

    return _Postings(documents, counts, spans)


def _score(
    index: BM25Okapi,
    postings: _Postings,
    tokens: Sequence[str],
    length_norm: np.ndarray,
) -> np.ndarray:
    """Score every document against the query terms.

    Scores identical to BM25Okapi.get_scores; keep them that way.
    """
    scores = np.zeros(len(index.doc_freqs))
    saturation = index.k1 + 1

    for token, count in Counter(tokens).items():
        idf = index.idf.get(token)
        span = postings.spans.get(token)
        if not idf or span is None:  # unindexed, or floored to zero weight
            continue
        documents = postings.documents[span[0] : span[1]]
        matched = postings.frequencies[span[0] : span[1]].astype(np.float64)
        scores[documents] += (count * idf) * (
            matched * saturation / (matched + length_norm[documents])
        )

    return scores


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
        self._length_norm = np.zeros(0)
        self._postings: _Postings | None = None
        self._filters: _FilterIndex | None = None

    def _attach(self, index: BM25Okapi, corpus: Corpus) -> None:
        """Bind a ranking model to the chunks it was built over.

        The rarest term's IDF and the length normalization are derived here
        rather than per query: both scan a structure the size of the corpus.
        Postings wait for the first query, where the corpus is read too.
        """
        self._index = index
        self._corpus = corpus
        self._unseen_idf = max(index.idf.values(), default=0.0)
        self._postings = None
        self._filters = None
        doc_len = np.array(index.doc_len, dtype=np.float64)
        self._length_norm = index.k1 * (1 - index.b + index.b * doc_len / index.avgdl)

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
            pickle.dump({"index": self._index, "tokenizer": _TOKENIZER_VERSION}, handle)

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
            ValueError: If the index was built by a different tokenizer.
        """
        directory = Path(directory)
        index_path = directory / "bm25.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        # Unpickling executes arbitrary code: only load index dirs this library wrote.
        with index_path.open("rb") as f:
            data = pickle.load(f)

        written_by = data.get("tokenizer", 1)
        if written_by != _TOKENIZER_VERSION:
            raise ValueError(
                f"BM25 index at {index_path} was built by tokenizer v{written_by}, "
                f"but this version indexes v{_TOKENIZER_VERSION} terms. "
                "Rebuild the project to reindex it."
            )

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

        candidates = self._candidates(chunks, metadata_filter, source_filter)
        if candidates is not None and not candidates.size:
            return LexicalResults([], {})

        if self._postings is None:
            self._postings = _invert(self._index)

        # IDF is computed over the whole corpus, so scoring cannot be restricted
        # to the candidates; only the ranking that follows is.
        tokens = _tokenize(query)
        scores = _score(self._index, self._postings, tokens, self._length_norm)
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

    def _candidates(
        self,
        chunks: Sequence[ContextualizedChunk],
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> np.ndarray | None:
        """Resolve the filters to corpus positions, indexing the corpus on first use."""
        if not metadata_filter and not source_filter:
            return None

        if self._filters is None:
            self._filters = _FilterIndex(chunks)
        selected = self._filters.select(metadata_filter, source_filter)
        if selected is None:
            return _filter_indices(chunks, metadata_filter, source_filter)
        return selected

    @property
    def is_empty(self) -> bool:
        """bool: True when there is no index to rank against.

        Answered from the ranking model alone: counting the corpus here would
        defeat deferring it.
        """
        return self._index is None
