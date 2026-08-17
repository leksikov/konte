"""The chunks the indexes rank over, read once and filtered once.

Both stores address the corpus by position, so one reading and one set of
postings serve them all — and a filter cannot mean one thing in the vector
index and another in the lexical one.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from konte.domain.models import ContextualizedChunk, MetadataFilter
from konte.index.filter_index import FilterIndex, matches_filter_value

# Returns the chunks an index was built over, in index order.
ChunkLoader = Callable[[], Sequence[ContextualizedChunk]]

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


def _filter_fields(chunk: ContextualizedChunk) -> Iterator[tuple[str, Any]]:
    """Yield the fields one chunk is filtered on, its own columns first.

    Metadata of a column's name is left out: _chunk_field reads the column, so
    posting the metadata would filter on something no scan agrees with.
    """
    yield "source", chunk.chunk.source
    yield "segment_idx", chunk.chunk.segment_idx
    yield "chunk_idx", chunk.chunk.chunk_idx
    for key, value in chunk.chunk.metadata.items():
        if key not in _CHUNK_COLUMNS:
            yield key, value


def _filter_entries(chunks: Sequence[ContextualizedChunk]) -> Iterator[tuple[int, Any]]:
    """Yield each chunk's corpus position and the fields to post it on."""
    return ((position, _filter_fields(chunk)) for position, chunk in enumerate(chunks))


class ChunkSource:
    """The corpus an index ranks over, and the postings a filter resolves through.

    Args:
        load: Returns the chunks, in index order. Called on the first query
            that needs them, not here.
    """

    __slots__ = ("_chunks", "_filters", "_load")

    def __init__(self, load: ChunkLoader) -> None:
        self._load = load
        self._chunks: Sequence[ContextualizedChunk] | None = None
        self._filters: FilterIndex | None = None

    @classmethod
    def holding(cls, chunks: Sequence[ContextualizedChunk] = ()) -> "ChunkSource":
        """Return a source over chunks already in memory, empty by default."""
        source = cls(lambda: chunks)
        source._chunks = chunks
        return source

    @property
    def chunks(self) -> Sequence[ContextualizedChunk]:
        """Sequence[ContextualizedChunk]: The corpus, read the first time it is asked for."""
        chunks = self._chunks
        if chunks is None:
            self._chunks = chunks = self._load()
        return chunks

    def select(
        self,
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> np.ndarray | None:
        """Resolve both filters to corpus positions, or None where nothing filters.

        Postings are built on the first filtered query; a field whose values
        cannot be hashed falls back to a scan.

        Args:
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            The matching positions, ascending; None where neither filter was given.
        """
        if not metadata_filter and not source_filter:
            return None

        chunks = self.chunks
        if self._filters is None:
            self._filters = FilterIndex(_filter_entries(chunks))
        selected = self._filters.select(metadata_filter, source_filter)
        if selected is None:
            return _filter_indices(chunks, metadata_filter, source_filter)
        return selected
