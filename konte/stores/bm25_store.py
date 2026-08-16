"""BM25 lexical search store."""

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
from konte.integrity import SIGNATURE_SUFFIX, sign, verify
from konte.models import ContextualizedChunk, MetadataFilter
from konte.storage import atomic_writer
from konte.stores.base import matches_filter_value

logger = structlog.get_logger()

# Returns the chunks an index was built over, in index order.
Corpus = Callable[[], Sequence[ContextualizedChunk]]

INDEX_FILENAME = "bm25.npz"

LEGACY_CHUNKS_FILENAME = "bm25_chunks.json"
LEGACY_INDEX_FILENAME = "bm25.pkl"

SIGNED_FILENAMES = (INDEX_FILENAME,)

# Bump on any change to the arrays save() writes.
_FORMAT_VERSION = 1

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


_NO_DOCUMENTS = np.empty(0, dtype=np.int32)


class _Model(NamedTuple):
    """The ranking state, in the form the index file stores it.

    Postings are term-major: a term's id spans `documents` and `frequencies`
    between consecutive `offsets`, ascending by document. BM25Okapi instead
    walks the whole corpus once per query token, and bigrams make a CJK query
    several times longer.
    """

    terms: dict[str, int]  # term -> its id, an index into idf and offsets
    offsets: np.ndarray
    documents: np.ndarray
    frequencies: np.ndarray
    idf: np.ndarray
    doc_len: np.ndarray
    avgdl: float
    k1: float
    b: float

    @property
    def size(self) -> int:
        """int: How many documents the model ranks."""
        return int(self.doc_len.size)

    def documents_for(self, term: str) -> np.ndarray:
        """Return the documents holding a term, ascending, empty when unindexed."""
        term_id = self.terms.get(term)
        if term_id is None:
            return _NO_DOCUMENTS
        return self.documents[self.offsets[term_id] : self.offsets[term_id + 1]]


def _holds(documents: np.ndarray, position: int) -> bool:
    """Test one document against a term's ascending span."""
    found = int(np.searchsorted(documents, position))
    return found < documents.size and int(documents[found]) == position


def _invert(index: BM25Okapi) -> _Model:
    """Regroup a freshly built index's per-document frequencies by term."""
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
    offsets = np.empty(len(grouped) + 1, dtype=np.int64)
    idf = np.empty(len(grouped), dtype=np.float64)
    terms: dict[str, int] = {}

    start = 0
    for term_id, (term, (term_documents, term_counts)) in enumerate(grouped.items()):
        stop = start + len(term_documents)
        documents[start:stop] = term_documents
        counts[start:stop] = term_counts
        offsets[term_id] = start
        idf[term_id] = index.idf.get(term, 0.0)
        terms[term] = term_id
        start = stop
    offsets[len(grouped)] = start

    return _Model(
        terms=terms,
        offsets=offsets,
        documents=documents,
        frequencies=counts,
        idf=idf,
        doc_len=np.array(index.doc_len, dtype=np.int32),
        avgdl=float(index.avgdl),
        k1=float(index.k1),
        b=float(index.b),
    )


def _encode_terms(terms: dict[str, int]) -> np.ndarray:
    """Pack the vocabulary into one array, in term id order.

    Newline separates the terms; no token _tokenize emits can contain one.
    """
    ordered = [""] * len(terms)
    for term, term_id in terms.items():
        ordered[term_id] = term
    return np.frombuffer("\n".join(ordered).encode("utf-8"), dtype=np.uint8)


def _decode_terms(vocabulary: np.ndarray) -> dict[str, int]:
    """Unpack the vocabulary written by _encode_terms()."""
    blob = vocabulary.tobytes().decode("utf-8")
    if not blob:
        return {}
    return {term: term_id for term_id, term in enumerate(blob.split("\n"))}


def _score(model: _Model, tokens: Sequence[str], length_norm: np.ndarray) -> np.ndarray:
    """Score every document against the query terms.

    Scores identical to BM25Okapi.get_scores; keep them that way.
    """
    scores = np.zeros(model.size)
    saturation = model.k1 + 1

    for token, count in Counter(tokens).items():
        term_id = model.terms.get(token)
        if term_id is None:
            continue
        idf = float(model.idf[term_id])
        if not idf:  # floored to zero weight
            continue
        start, stop = model.offsets[term_id], model.offsets[term_id + 1]
        documents = model.documents[start:stop]
        matched = model.frequencies[start:stop].astype(np.float64)
        scores[documents] += (count * idf) * (
            matched * saturation / (matched + length_norm[documents])
        )

    return scores


def _term_weights(model: _Model, tokens: Sequence[str], unseen: float) -> dict[str, float]:
    """Weight each distinct query term by how much it narrows the corpus.

    An unindexed term weighs as much as the rarest indexed one, so naming
    something absent cannot read as a full match. Negative IDF — a term in
    most documents, or any term of a corpus too small to separate — clamps away.
    """
    terms = model.terms
    return {
        token: max(float(model.idf[terms[token]]) if token in terms else unseen, 0.0)
        for token in dict.fromkeys(tokens)
    }


def _coverage(model: _Model, weights: dict[str, float], positions: Sequence[int]) -> list[float]:
    """Score how much of the weighted query each of the given documents carries."""
    if not weights:
        return [0.0] * len(positions)

    total = sum(weights.values())
    if total <= 0:  # no weight survived the clamp; fall back to counting terms
        weights = dict.fromkeys(weights, 1.0)
        total = float(len(weights))

    carried = [(weight, model.documents_for(term)) for term, weight in weights.items()]
    return [
        sum(weight for weight, documents in carried if _holds(documents, position)) / total
        for position in positions
    ]


class BM25Store:
    """BM25 store for lexical search on contextualized chunks."""

    def __init__(self):
        """Initialize BM25 store."""
        self._model: _Model | None = None
        self._corpus: Corpus = list  # nothing indexed yet
        self._unseen_idf = 0.0
        self._length_norm = np.zeros(0)
        self._filters: _FilterIndex | None = None

    def _attach(self, model: _Model, corpus: Corpus) -> None:
        """Bind a ranking model to the chunks it was built over.

        The rarest term's IDF and the length normalization are derived here
        rather than per query: both scan a structure the size of the corpus.
        """
        self._model = model
        self._corpus = corpus
        self._unseen_idf = float(model.idf.max()) if model.idf.size else 0.0
        self._filters = None
        doc_len = model.doc_len.astype(np.float64)
        self._length_norm = model.k1 * (1 - model.b + model.b * doc_len / model.avgdl)

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build BM25 index from contextualized chunks.

        Args:
            chunks: List of contextualized chunks to index.
        """
        if not chunks:
            logger.warning("bm25_build_empty_chunks")
            return

        index = BM25Okapi([_tokenize(c.contextualized_content) for c in chunks])
        self._attach(_invert(index), lambda: chunks)

        logger.info("bm25_index_built", num_chunks=len(chunks))

    def save(self, directory: Path) -> None:
        """Save the BM25 ranking model to disk.

        Plain arrays rather than a pickled rank_bm25 model: reading a pickle
        runs whatever it holds. The corpus is not stored — ranking reads the
        term frequencies alone.

        Args:
            directory: Directory to save index files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model = self._model
        if model is None:
            logger.warning("bm25_save_no_index")
            return

        with atomic_writer(directory / INDEX_FILENAME) as handle:
            np.savez(
                handle,
                format=np.array([_FORMAT_VERSION]),
                tokenizer=np.array([_TOKENIZER_VERSION]),
                vocabulary=_encode_terms(model.terms),
                offsets=model.offsets,
                documents=model.documents,
                frequencies=model.frequencies,
                idf=model.idf,
                doc_len=model.doc_len,
                params=np.array([model.avgdl, model.k1, model.b]),
            )

        sign(directory, SIGNED_FILENAMES)

        # Earlier versions kept a second copy of the corpus, then a pickle.
        for stale in (
            LEGACY_CHUNKS_FILENAME,
            LEGACY_INDEX_FILENAME,
            f"{LEGACY_INDEX_FILENAME}{SIGNATURE_SUFFIX}",
        ):
            (directory / stale).unlink(missing_ok=True)

        logger.info("bm25_index_saved", directory=str(directory))

    def load(self, directory: Path, corpus: Corpus) -> None:
        """Load the BM25 ranking model from disk and bind it to a corpus.

        Args:
            directory: Directory containing index files.
            corpus: Returns the chunks the index was built over, in index
                order. Called on the first query, not here.

        Raises:
            FileNotFoundError: If the index file is missing.
            IntegrityError: If the index does not match what was recorded for it.
            ValueError: If the index was written by a different format or
                tokenizer version.
        """
        directory = Path(directory)
        index_path = directory / INDEX_FILENAME

        if not index_path.exists():
            if (directory / LEGACY_INDEX_FILENAME).exists():
                raise ValueError(
                    f"{directory / LEGACY_INDEX_FILENAME} holds a pickled model, and "
                    f"reading a pickle runs whatever it holds, so it is never loaded. "
                    f"Rebuild the project to write {INDEX_FILENAME} instead."
                )
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        verify(directory, SIGNED_FILENAMES)

        # allow_pickle stays off: an array claiming to hold objects would be
        # executed on the way in.
        with np.load(index_path, allow_pickle=False) as data:
            self._reject_stale(index_path, data)
            avgdl, k1, b = (float(value) for value in data["params"])
            model = _Model(
                terms=_decode_terms(data["vocabulary"]),
                offsets=data["offsets"],
                documents=data["documents"],
                frequencies=data["frequencies"],
                idf=data["idf"],
                doc_len=data["doc_len"],
                avgdl=avgdl,
                k1=k1,
                b=b,
            )

        self._attach(model, cache(corpus))

        logger.info("bm25_index_loaded", directory=str(directory))

    @staticmethod
    def _reject_stale(index_path: Path, data: Any) -> None:
        """Refuse an index this version would read as something it is not."""
        written_by = int(data["format"][0])
        if written_by != _FORMAT_VERSION:
            raise ValueError(
                f"BM25 index at {index_path} is format v{written_by}, but this version "
                f"reads v{_FORMAT_VERSION}. Rebuild the project to rewrite it."
            )

        tokenized_by = int(data["tokenizer"][0])
        if tokenized_by != _TOKENIZER_VERSION:
            raise ValueError(
                f"BM25 index at {index_path} was built by tokenizer v{tokenized_by}, "
                f"but this version indexes v{_TOKENIZER_VERSION} terms. "
                "Rebuild the project to reindex it."
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
        model = self._model
        if model is None or not chunks:
            logger.warning("bm25_query_empty_index")
            return LexicalResults([], {})

        candidates = self._candidates(chunks, metadata_filter, source_filter)
        if candidates is not None and not candidates.size:
            return LexicalResults([], {})

        # IDF is computed over the whole corpus, so scoring cannot be restricted
        # to the candidates; only the ranking that follows is.
        tokens = _tokenize(query)
        scores = _score(model, tokens, self._length_norm)
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
        weights = _term_weights(model, tokens, self._unseen_idf)
        return LexicalResults(
            results,
            {
                chunk.chunk.chunk_id: matched
                for (chunk, _), matched in zip(
                    results, _coverage(model, weights, positions), strict=True
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
        return self._model is None
