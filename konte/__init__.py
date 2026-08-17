"""Konte - Contextual RAG library with hybrid retrieval."""

from importlib.metadata import PackageNotFoundError, version

from konte.answer import GeneratedAnswer, generate_answer
from konte.domain.config import ContextStrategy, ProjectConfig
from konte.domain.corpus import Corpus
from konte.domain.models import (
    Chunk,
    ContextualizedChunk,
    MetadataFilter,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)
from konte.index.embeddings import clear_query_embedding_cache
from konte.manager import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    project_exists,
    trust_project,
)
from konte.persistence.checkpoint import BuildCheckpoint
from konte.persistence.integrity import IntegrityError
from konte.project import Project
from konte.retrieval.query_processor import (
    clear_keyword_cache,
    extract_search_keywords,
    extract_search_keywords_async,
)
from konte.runtime.cache import (
    clear_project_cache,
    get_shared_project,
    invalidate_project,
    preload_projects,
)
from konte.runtime.settings import settings

# Single source of truth is pyproject.toml, read from installed metadata
try:
    __version__ = version("konte")
except PackageNotFoundError:  # running from an uninstalled source tree
    __version__ = "0.0.0+unknown"
__all__ = [
    # Settings
    "settings",
    # Models
    "BuildCheckpoint",
    "Chunk",
    "ContextStrategy",
    "ContextualizedChunk",
    "Corpus",
    "MetadataFilter",
    "RetrievalMode",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalResponse",
    "ProjectConfig",
    "GeneratedAnswer",
    # Project
    "Project",
    # Manager functions
    "create_project",
    "list_projects",
    "get_project",
    "delete_project",
    "project_exists",
    "trust_project",
    # Index integrity
    "IntegrityError",
    # Serving cache
    "get_shared_project",
    "preload_projects",
    "invalidate_project",
    "clear_project_cache",
    # Generator
    "generate_answer",
    # Query processor
    "extract_search_keywords",
    "extract_search_keywords_async",
    "clear_keyword_cache",
    # Query embeddings
    "clear_query_embedding_cache",
]
