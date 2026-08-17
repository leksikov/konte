"""The chunks a project holds and the shape of a retrieval against them."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

MetadataFilter = dict[str, Any]  # simple equality filter, AND logic across keys
RetrievalMode = Literal["hybrid", "semantic", "lexical"]
SuggestedAction = Literal["deliver", "query_more", "refine_query"]
SegmentKey = tuple[str, int]

_SEGMENT_KEY_SEPARATOR = "|"


def encode_segment_key(key: SegmentKey) -> str:
    """Flatten a (source, segment index) key into a JSON-safe string."""
    source, index = key
    return f"{source}{_SEGMENT_KEY_SEPARATOR}{index}"


def decode_segment_key(raw: str) -> SegmentKey:
    """Parse a stored segment key.

    Keys with no separator were written by an older format that recorded only
    the segment index, so their source document is unrecoverable.
    """
    source, separator, index = raw.rpartition(_SEGMENT_KEY_SEPARATOR)
    if not separator:
        return ("unknown", int(raw))
    return (source, int(index))


class Chunk(BaseModel):
    """A chunk of text from a document."""

    chunk_id: str
    content: str
    source: str
    segment_idx: int
    chunk_idx: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextualizedChunk(BaseModel):
    """A chunk with LLM-generated context prepended."""

    chunk: Chunk
    context: str

    @computed_field
    @property
    def contextualized_content(self) -> str:
        """Return context + original content."""
        if not self.context:
            return self.chunk.content
        return f"{self.context} {self.chunk.content}"

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialize to the JSON shape shared by every on-disk artifact.

        `contextualized_content` is deliberately excluded: it is derived from the
        other two fields, so persisting it would only let the copy on disk drift
        away from the definition above.

        Returns:
            Mapping with a nested `chunk` dict and a `context` string.
        """
        return {"chunk": self.chunk.model_dump(), "context": self.context}

    @classmethod
    def from_storage_dict(cls, data: dict[str, Any]) -> "ContextualizedChunk":
        """Rebuild an instance from the shape written by to_storage_dict().

        Args:
            data: Mapping with a nested `chunk` dict and a `context` string.

        Returns:
            The reconstructed chunk.
        """
        return cls(chunk=Chunk(**data["chunk"]), context=data["context"])


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Everything one retrieval is asked for.

    A dataclass rather than a model: it is handed from the project down to the
    stores rather than serialized, and carrying it whole is what keeps the
    retrieval path from repeating these ten parameters at every step.

    Args:
        query: Query string, as the caller wrote it.
        mode: Retrieval mode - "hybrid", "semantic", or "lexical".
        top_k: Number of results. None follows settings.DEFAULT_TOP_K.
        use_keyword_extraction: Extract keywords before BM25 search, at the
            cost of one LLM call. None follows settings.BM25_KEYWORD_EXTRACTION,
            except where `keyword_extraction` below says otherwise.
        metadata_filter: Filter results by metadata (equality match, AND logic).
            Example: {"source": "doc.pdf", "company": "ACME", "year": 2024}
        source_filter: Substring match on chunk source field.
            Example: "JOHNSON" matches "JOHNSON_JOHNSON_2022_10K.md"
        rerank: Rerank via the configured RERANKER_BASE_URL endpoint. Only the
            async entry point can serve this.
        rerank_initial_k: Candidates to retrieve before reranking.
        inject_evidence: For ablation study - inject this text.
        inject_position: Position to inject (0=top, None=random).
    """

    query: str
    mode: RetrievalMode = "hybrid"
    top_k: int | None = None
    use_keyword_extraction: bool | None = None
    metadata_filter: MetadataFilter | None = None
    source_filter: str | None = None
    rerank: bool = False
    rerank_initial_k: int = 50
    inject_evidence: str | None = None
    inject_position: int | None = None

    @property
    def keyword_extraction(self) -> bool | None:
        """bool | None: The extraction flag the lexical index is actually read with.

        Deliberate: reranked hybrid has always fed BM25 the raw query, so an
        unset flag stays off there instead of following the configured default.
        Enabling it would shift every reranked hybrid result.
        """
        if self.use_keyword_extraction is None and self.rerank and self.mode == "hybrid":
            return False
        return self.use_keyword_extraction


class RetrievalResult(BaseModel):
    """A single retrieval result with score."""

    content: str
    context: str
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Ranking score within this response. Fusion and lexical "
            "normalization scale the winner to 1.0, so it does not compare "
            "across queries; read top_score for that."
        ),
    )
    source: str
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Response from retrieval with agent decision hints."""

    results: list[RetrievalResult]
    query: str
    total_found: int
    top_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the best result matches the query: vector similarity, "
            "lexical query coverage, or the reranker's score. Comparable "
            "across queries, unlike RetrievalResult.score."
        ),
    )
    score_spread: float = Field(
        ge=0.0,
        description="Best minus worst result on that same measure.",
    )
    has_high_confidence: bool
    suggested_action: SuggestedAction

    @classmethod
    def empty(cls, query: str) -> "RetrievalResponse":
        """Build the response for a query that matched nothing.

        A zero top score sits below every confidence threshold, so the caller is
        told to refine the query rather than to deliver an empty answer.

        Args:
            query: The query that produced no results.

        Returns:
            An empty response carrying the `refine_query` hint.
        """
        return cls(
            results=[],
            query=query,
            total_found=0,
            top_score=0.0,
            score_spread=0.0,
            has_high_confidence=False,
            suggested_action="refine_query",
        )
