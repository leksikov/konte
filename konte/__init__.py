"""Konte - Contextual RAG library with hybrid retrieval."""

from importlib.metadata import PackageNotFoundError, version

from konte.config import settings
from konte.generator import GeneratedAnswer, generate_answer
from konte.manager import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    project_exists,
)
from konte.models import (
    BuildCheckpoint,
    Chunk,
    ContextualizedChunk,
    MetadataFilter,
    ProjectConfig,
    RetrievalMode,
    RetrievalResponse,
    RetrievalResult,
)
from konte.project import Project
from konte.query_processor import extract_search_keywords

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
    "ContextualizedChunk",
    "MetadataFilter",
    "RetrievalMode",
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
    # Generator
    "generate_answer",
    # Query processor
    "extract_search_keywords",
]
