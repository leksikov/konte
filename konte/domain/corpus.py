"""The corpus a project holds, and the parse each artifact may still be waiting on."""

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from konte.domain.models import Chunk, ContextualizedChunk, SegmentKey

_T = TypeVar("_T")

_ARTIFACTS = ("chunks", "segments", "contextualized_chunks")


class _Cell(Generic[_T]):
    """One artifact, held either as a value or as the call that produces it."""

    __slots__ = ("_load", "_value")

    def __init__(self, value: _T) -> None:
        self._value = value
        self._load: Callable[[], _T] | None = None

    @property
    def loaded(self) -> bool:
        """bool: Whether the value is in memory rather than waiting to be parsed."""
        return self._load is None

    def get(self) -> _T:
        """Return the value, running a pending parse the first time only."""
        load = self._load
        if load is not None:
            self._load = None
            self._value = load()
        return self._value

    def set(self, value: _T) -> None:
        """Replace the value, discarding a parse that has not run yet."""
        self._value = value
        self._load = None

    def defer(self, load: Callable[[], _T]) -> None:
        """Arrange for the value to be parsed the first time it is read."""
        self._load = load


class Corpus:
    """A project's chunks, the segments they were cut from, and their contexts.

    Each artifact may be left to a deferred parse. Only building and saving read
    the corpus-sized ones, so a query-only caller never pays to parse them, at
    the cost of a damaged artifact surfacing at first use rather than at load.
    """

    __slots__ = ("_chunks", "_contextualized_chunks", "_segments")

    def __init__(
        self,
        chunks: Sequence[Chunk] | None = None,
        segments: dict[SegmentKey, str] | None = None,
        contextualized_chunks: Sequence[ContextualizedChunk] | None = None,
    ) -> None:
        """Hold the artifacts passed in, each defaulting to an empty one.

        Args:
            chunks: The raw corpus.
            segments: Segment texts, keyed by (source, segment index).
            contextualized_chunks: What the last build contextualized.
        """
        self._chunks: _Cell[list[Chunk]] = _Cell(list(chunks or ()))
        self._segments: _Cell[dict[SegmentKey, str]] = _Cell(dict(segments or {}))
        self._contextualized_chunks: _Cell[list[ContextualizedChunk]] = _Cell(
            list(contextualized_chunks or ())
        )

    @classmethod
    def deferred(
        cls,
        *,
        chunks: Callable[[], list[Chunk]],
        segments: Callable[[], dict[SegmentKey, str]],
        contextualized_chunks: Callable[[], list[ContextualizedChunk]],
    ) -> "Corpus":
        """Build a corpus whose every artifact is parsed the first time it is read.

        Args:
            chunks: Called to parse the raw corpus.
            segments: Called to parse the segment texts.
            contextualized_chunks: Called to parse the contextualized chunks.

        Returns:
            An empty-looking corpus that fills itself in on demand.
        """
        corpus = cls()
        corpus._chunks.defer(chunks)
        corpus._segments.defer(segments)
        corpus._contextualized_chunks.defer(contextualized_chunks)
        return corpus

    @property
    def chunks(self) -> list[Chunk]:
        """list[Chunk]: The raw corpus, in the order its documents were added."""
        return self._chunks.get()

    @chunks.setter
    def chunks(self, value: list[Chunk]) -> None:
        self._chunks.set(value)

    @property
    def segments(self) -> dict[SegmentKey, str]:
        """dict[SegmentKey, str]: Segment texts each chunk's context is written against."""
        return self._segments.get()

    @segments.setter
    def segments(self, value: dict[SegmentKey, str]) -> None:
        self._segments.set(value)

    @property
    def contextualized_chunks(self) -> list[ContextualizedChunk]:
        """list[ContextualizedChunk]: What the last build contextualized."""
        return self._contextualized_chunks.get()

    @contextualized_chunks.setter
    def contextualized_chunks(self, value: list[ContextualizedChunk]) -> None:
        self._contextualized_chunks.set(value)

    @property
    def loaded(self) -> frozenset[str]:
        """frozenset[str]: Which of `_ARTIFACTS` are in memory rather than pending."""
        cells = (self._chunks, self._segments, self._contextualized_chunks)
        return frozenset(name for name, cell in zip(_ARTIFACTS, cells, strict=True) if cell.loaded)
