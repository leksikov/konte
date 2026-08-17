"""Combining ranked lists, and the readings that survive the combination."""

from collections.abc import Sequence
from typing import NamedTuple

from konte.domain.models import ContextualizedChunk

ScoredChunks = list[tuple[ContextualizedChunk, float]]

RRF_K = 60


class Ranked(NamedTuple):
    """A ranked list, and how well each of its chunks matched, by chunk id.

    Fusion and lexical normalization both scale their winner to 1.0, so a
    ranking score cannot say whether a chunk answered the query or was merely
    the least bad. `absolute` can: vector similarity, lexical coverage, or the
    reranker's score. Only it may reach the confidence hints.
    """

    results: ScoredChunks
    absolute: dict[str, float]


def reciprocal_rank_fusion(
    results_list: list[ScoredChunks],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> ScoredChunks:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(w_i / (k + rank_i)) for each result list

    Args:
        results_list: List of result lists, each containing (chunk, score) tuples.
        k: Constant to prevent high ranks from dominating (default 60).
        weights: How much a rank in each list counts, aligned with
            results_list. Only their ratio matters. Defaults to equal.

    Returns:
        Combined list of (chunk, score) tuples sorted by RRF score, rescaled so
        the best result scores 1.0. Ranking only: the rescaling erases how well
        anything matched. See Ranked.
    """
    fused: dict[str, tuple[ContextualizedChunk, float]] = {}
    if weights is None:
        weights = (1.0,) * len(results_list)

    for results, weight in zip(results_list, weights, strict=True):
        for rank, (chunk, _) in enumerate(results):
            chunk_id = chunk.chunk.chunk_id
            contribution = weight / (k + rank + 1)  # rank is 0-indexed
            existing = fused.get(chunk_id)
            if existing is None:
                fused[chunk_id] = (chunk, contribution)
            else:
                fused[chunk_id] = (existing[0], existing[1] + contribution)

    ranked = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    if not ranked:
        return []

    top_score = ranked[0][1]
    if top_score <= 0:
        return ranked
    return [(chunk, score / top_score) for chunk, score in ranked]


def by_chunk_id(results: ScoredChunks) -> dict[str, float]:
    """Read ranking scores as absolute ones, for the indexes where they are."""
    return {chunk.chunk.chunk_id: score for chunk, score in results}


def merge_absolute(*readings: dict[str, float]) -> dict[str, float]:
    """Keep the strongest reading per chunk, so fusion cannot penalize agreement."""
    merged: dict[str, float] = {}
    for reading in readings:
        for chunk_id, score in reading.items():
            if score > merged.get(chunk_id, 0.0):
                merged[chunk_id] = score
    return merged


def drop_redundant(results: ScoredChunks, limit: int) -> ScoredChunks:
    """Keep the best `limit` of `results`, best first, dropping repeated text.

    Overlapping segments cut a boundary passage twice, so it ranks twice under
    two chunk ids. The two cuts share a boundary, so a copy is a kept result's
    whole, head, or tail; a repeat buried inside a longer chunk is left alone.
    A lower-ranked superset survives — dropping it would lose the text it adds.
    """
    kept: ScoredChunks = []
    texts: list[str] = []

    for chunk, score in results:
        content = chunk.chunk.content
        if any(seen.endswith(content) or seen.startswith(content) for seen in texts):
            continue
        kept.append((chunk, score))
        texts.append(content)
        if len(kept) == limit:
            break

    return kept
