"""Pydantic models for Konte contextual RAG library."""

from pathlib import Path, PurePath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

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


class BuildCheckpoint(BaseModel):
    """Checkpoint state for build process resumption."""

    completed_segments: list[str] = Field(default_factory=list)
    contextualized_chunks: list[dict[str, Any]] = Field(default_factory=list)


def validate_project_name(name: str) -> str:
    """Return the name if it names one directory under the storage root.

    Callers join the name onto the storage path, so one the filesystem reads
    as more than a single component reaches outside the root — where
    delete_project() would remove whatever it found.

    Raises:
        ValueError: If the name is not a single path component.
    """
    if not name or name == ".." or "\0" in name or PurePath(name).name != name:
        raise ValueError(
            f"Invalid project name: {name!r}. A project name is one directory "
            "name under the storage root: no path separators, no '..', not absolute."
        )
    return name


class ProjectConfig(BaseModel):
    """Configuration for a project."""

    name: str
    storage_path: Path

    # Segmentation
    segment_size: int = 8000
    segment_overlap: int = 800

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 80

    # Context
    context_prompt_path: Path | None = None  # Per-project prompt override

    # Models
    embedding_model: str = "text-embedding-3-small"
    context_model: str | None = None  # Defaults to settings.LLM_MODEL or CONTEXT_MODEL

    # Index options
    enable_faiss: bool = True
    enable_bm25: bool = True

    # Fusion weights. Only their ratio matters, not their scale.
    fusion_weight_semantic: float = Field(default=0.5, ge=0.0)
    fusion_weight_lexical: float = Field(default=0.5, ge=0.0)

    model_config = ConfigDict(ser_json_path="str")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """A name read back from config.json is checked like any other."""
        return validate_project_name(value)
