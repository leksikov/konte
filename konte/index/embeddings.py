"""Query vectors, cached for the process.

Ranking is an in-memory index lookup, so embedding the query is the one network
round trip a semantic search pays for. Only queries are cached: document vectors
are asked for once per build, and holding a corpus of them would cost more than
the index they are packed into.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = structlog.get_logger()

# 1536 float32 per entry under the default model: a few megabytes when full.
_CACHE_SIZE = 512


class _VectorSlot:
    """Holder for one query's vector.

    The cache stores the slot rather than the vector, so a failed request leaves
    an empty slot to ask again instead of an entry remembering the failure.
    """

    __slots__ = ("vector",)

    def __init__(self) -> None:
        self.vector: np.ndarray | None = None


@lru_cache(maxsize=_CACHE_SIZE)
def _vector_slot(model: str, query: str) -> _VectorSlot:
    """Return the slot for one model and query, allocating it on first sight."""
    return _VectorSlot()


def clear_query_embedding_cache() -> None:
    """Forget every query vector cached so far.

    Entries are keyed by model name and query text, so a process repointed at
    another endpoint under the same model name keeps serving the vectors it
    already holds until the cache is emptied.
    """
    _vector_slot.cache_clear()


def query_vector(embeddings: "Embeddings", model: str, query: str) -> np.ndarray:
    """Return the row FAISS searches with for one query, embedding it once.

    Args:
        embeddings: Client that answers the request on a miss.
        model: Embedding model the vector belongs to. Part of the key, since
            the same text embeds differently under a different model.
        query: Query text.

    Returns:
        A read-only (1, dimensions) float32 row, shared with every other caller
        holding the same model and query.
    """
    slot = _vector_slot(model, query)
    vector = slot.vector
    if vector is not None:
        return vector

    vector = np.asarray(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    vector.flags.writeable = False  # shared with every later search of this query

    slot.vector = vector
    logger.debug("query_embedded", model=model, dimensions=vector.shape[1])
    return vector
