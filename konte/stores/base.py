"""Shared protocol and filter semantics for retrieval stores."""

from pathlib import Path
from typing import Any, Protocol

from konte.models import ContextualizedChunk, MetadataFilter


def matches_filter_value(actual: Any, expected: Any) -> bool:
    """Check one metadata field against one filter value.

    A list on the filter side means "match any"; any other value is compared for
    equality. Both stores route through this so a filter cannot mean one thing
    in the vector index and another in the lexical one.

    Args:
        actual: Value read from the chunk.
        expected: Value supplied in the filter.

    Returns:
        True if the field satisfies the filter.
    """
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


class Store(Protocol):
    """Protocol defining the interface for retrieval stores.

    Loading is outside it: a store may need more than a directory to come back.
    """

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build index from contextualized chunks."""
        ...

    def save(self, directory: Path) -> None:
        """Save index to disk."""
        ...

    def query(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_filter: str | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the index, return (chunk, score) tuples.

        Args:
            query: Query string.
            top_k: Number of results to return.
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.
        """
        ...

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        ...
