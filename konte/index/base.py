"""Shared protocol for retrieval stores."""

from pathlib import Path
from typing import Protocol

from konte.domain.models import ContextualizedChunk, MetadataFilter


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
