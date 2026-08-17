"""The objects the rest of the library is written in terms of."""

from konte.domain.config import ProjectConfig, validate_project_name
from konte.domain.corpus import Corpus
from konte.domain.models import (
    Chunk,
    ContextualizedChunk,
    MetadataFilter,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    SegmentKey,
    SuggestedAction,
    decode_segment_key,
    encode_segment_key,
)

__all__ = [
    "Chunk",
    "ContextualizedChunk",
    "Corpus",
    "MetadataFilter",
    "ProjectConfig",
    "RetrievalMode",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievalResult",
    "SegmentKey",
    "SuggestedAction",
    "decode_segment_key",
    "encode_segment_key",
    "validate_project_name",
]
