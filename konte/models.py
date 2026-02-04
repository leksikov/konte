"""Pydantic models for Konte contextual RAG library."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Type alias for metadata filtering (simple equality filter with AND logic)
MetadataFilter = dict[str, Any]


class Chunk(BaseModel):
    """A chunk of text from a document."""

    chunk_id: str
    content: str
    source: str
    segment_idx: int
    chunk_idx: int
    metadata: dict = Field(default_factory=dict)


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


class RetrievalResult(BaseModel):
    """A single retrieval result with score."""

    content: str
    context: str
    score: float = Field(ge=0.0, le=1.0)
    source: str
    chunk_id: str
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Response from retrieval with agent decision hints."""

    results: list[RetrievalResult]
    query: str
    total_found: int
    top_score: float = Field(ge=0.0, le=1.0)
    score_spread: float = Field(ge=0.0)
    has_high_confidence: bool
    suggested_action: Literal["deliver", "query_more", "refine_query"]


class BuildCheckpoint(BaseModel):
    """Checkpoint state for build process resumption."""

    completed_segments: list[str] = Field(default_factory=list)
    contextualized_chunks: list[dict] = Field(default_factory=list)


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
    context_min_tokens: int = 100
    context_max_tokens: int = 200

    # Models
    embedding_model: str = "text-embedding-3-small"
    context_model: str | None = None  # Use Backend.AI or settings.CONTEXT_MODEL

    # Index options
    enable_faiss: bool = True
    enable_bm25: bool = True

    # Fusion weights
    fusion_weight_semantic: float = 0.5
    fusion_weight_lexical: float = 0.5

    model_config = ConfigDict(ser_json_path="str")
