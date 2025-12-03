"""Konte - Contextual RAG library with hybrid retrieval."""

from konte.manager import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    project_exists,
)
from konte.models import (
    Chunk,
    ContextualizedChunk,
    ProjectConfig,
    RetrievalResponse,
    RetrievalResult,
)
from konte.project import Project
from konte.settings import settings

__version__ = "0.1.0"
__all__ = [
    # Settings
    "settings",
    # Models
    "Chunk",
    "ContextualizedChunk",
    "RetrievalResult",
    "RetrievalResponse",
    "ProjectConfig",
    # Project
    "Project",
    # Manager functions
    "create_project",
    "list_projects",
    "get_project",
    "delete_project",
    "project_exists",
]
