"""Base protocol for retrieval stores."""

from pathlib import Path
from typing import Any, Protocol

from konte.models import ContextualizedChunk


class Store(Protocol):
    """Protocol defining the interface for retrieval stores."""

    def build_index(self, chunks: list[ContextualizedChunk]) -> None:
        """Build index from contextualized chunks."""
        ...

    def save(self, directory: Path) -> None:
        """Save index to disk."""
        ...

    def load(self, directory: Path) -> None:
        """Load index from disk."""
        ...

    def query(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[ContextualizedChunk, float]]:
        """Query the index, return (chunk, score) tuples.

        Args:
            query: Query string.
            top_k: Number of results to return.
            metadata_filter: Filter results by metadata (equality match, AND logic).
        """
        ...

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        ...
