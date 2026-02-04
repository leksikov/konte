"""Konte - Contextual RAG library with hybrid retrieval."""

from konte.config import settings
from konte.generator import GeneratedAnswer, generate_answer
from konte.query_processor import extract_search_keywords
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
    RetrievalResponse,
    RetrievalResult,
)
from konte.project import Project

__version__ = "0.1.0"
__all__ = [
    # Settings
    "settings",
    # Models
    "BuildCheckpoint",
    "Chunk",
    "ContextualizedChunk",
    "MetadataFilter",
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
