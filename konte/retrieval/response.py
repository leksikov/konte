"""Wrapping a ranked list as the response a caller reads."""

import random

from konte.domain.models import RetrievalResponse, RetrievalResult, SuggestedAction
from konte.retrieval.fusion import Ranked
from konte.runtime.settings import settings

_INJECTED_EVIDENCE_SCORE = 0.95


def _clamp_unit(score: float) -> float:
    """Hold a score inside the 0-1 range the response models declare.

    A reranker scores on whatever scale its head was trained on — logits either
    side of zero, or a sigmoid landing a hair past 1.0 — and only the ordering
    reaches here, so clamping moves no result. max() first folds NaN to 0.0.
    """
    return min(1.0, max(0.0, score))


def suggested_action(top_score: float) -> SuggestedAction:
    """Determine suggested action based on top score.

    Args:
        top_score: The highest retrieval score.

    Returns:
        Suggested action for the agent.
    """
    if top_score >= settings.CONFIDENCE_THRESHOLD:
        return "deliver"
    if top_score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
        return "query_more"
    return "refine_query"


def build_response(query: str, ranked: Ranked, top_k: int) -> RetrievalResponse:
    """Build RetrievalResponse from retrieval results.

    The hints read `ranked.absolute`, never the ranking scores: those are
    rescaled per response, so a threshold on them fires on everything.

    Args:
        query: The original query.
        ranked: Ranked results and their absolute match readings.
        top_k: Number of results to include.

    Returns:
        RetrievalResponse with agent decision hints.
    """
    results = ranked.results[:top_k]
    measured = [_clamp_unit(ranked.absolute.get(chunk.chunk.chunk_id, 0.0)) for chunk, _ in results]

    top_score = max(measured, default=0.0)
    bottom_score = min(measured, default=0.0)

    return RetrievalResponse(
        results=[
            RetrievalResult(
                content=chunk.chunk.content,
                context=chunk.context,
                score=_clamp_unit(score),
                source=chunk.chunk.source,
                chunk_id=chunk.chunk.chunk_id,
                metadata=chunk.chunk.metadata,
            )
            for chunk, score in results
        ],
        query=query,
        total_found=len(results),
        top_score=top_score,
        score_spread=top_score - bottom_score,
        has_high_confidence=top_score >= settings.CONFIDENCE_THRESHOLD,
        suggested_action=suggested_action(top_score),
    )


def inject_evidence(
    response: RetrievalResponse,
    evidence: str,
    position: int | None = None,
) -> RetrievalResponse:
    """Inject evidence at specified position for ablation study.

    Args:
        response: Original retrieval response.
        evidence: Evidence text to inject.
        position: Position to insert (0=top, None=random).

    Returns:
        New RetrievalResponse with evidence injected.
    """
    evidence_result = RetrievalResult(
        content=evidence,
        context="[Ablation study: injected evidence]",
        score=_INJECTED_EVIDENCE_SCORE,
        source="ablation_study",
        chunk_id="INJECTED_EVIDENCE",
        metadata={"injected": True},
    )

    if position is None:
        insert_at = random.randint(0, len(response.results))
    else:
        insert_at = min(position, len(response.results))

    new_results = list(response.results)
    new_results.insert(insert_at, evidence_result)

    return RetrievalResponse(
        results=new_results,
        query=response.query,
        total_found=len(new_results),
        top_score=_INJECTED_EVIDENCE_SCORE,
        score_spread=response.score_spread,
        has_high_confidence=True,
        suggested_action="deliver",
    )
