"""Konte - Contextual RAG library with hybrid retrieval."""

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

# Which module each exported name lives in. Resolved on first access, so that
# importing this package does not pull in faiss and the OpenAI client stack.
_EXPORTS = {
    "GeneratedAnswer": "konte.answer",
    "generate_answer": "konte.answer",
    "ContextStrategy": "konte.domain.config",
    "ProjectConfig": "konte.domain.config",
    "Corpus": "konte.domain.corpus",
    "Chunk": "konte.domain.models",
    "ContextualizedChunk": "konte.domain.models",
    "MetadataFilter": "konte.domain.models",
    "RetrievalMode": "konte.domain.models",
    "RetrievalRequest": "konte.domain.models",
    "RetrievalResponse": "konte.domain.models",
    "RetrievalResult": "konte.domain.models",
    "clear_query_embedding_cache": "konte.index.embeddings",
    "create_project": "konte.manager",
    "delete_project": "konte.manager",
    "get_project": "konte.manager",
    "list_projects": "konte.manager",
    "project_exists": "konte.manager",
    "trust_project": "konte.manager",
    "BuildCheckpoint": "konte.persistence.checkpoint",
    "IntegrityError": "konte.persistence.integrity",
    "Project": "konte.project",
    "clear_keyword_cache": "konte.retrieval.query_processor",
    "extract_search_keywords": "konte.retrieval.query_processor",
    "extract_search_keywords_async": "konte.retrieval.query_processor",
    "clear_project_cache": "konte.runtime.cache",
    "get_shared_project": "konte.runtime.cache",
    "invalidate_project": "konte.runtime.cache",
    "preload_projects": "konte.runtime.cache",
    "settings": "konte.runtime.settings",
}


def _read_version() -> str:
    """Read this distribution's version, whose source of truth is pyproject.toml."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("konte")
    except PackageNotFoundError:  # running from an uninstalled source tree
        return "0.0.0+unknown"


def __getattr__(name: str) -> Any:
    """Resolve an exported name, or a submodule, on first access."""
    if name == "__version__":
        value: Any = _read_version()
    elif name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
    elif find_spec(f"{__name__}.{name}") is not None:
        # So that `import konte` alone is enough to reach konte.project.
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List what this module exports, resolved or not."""
    from pkgutil import iter_modules

    submodules = (module.name for module in iter_modules(__path__))
    return sorted({*globals(), *_EXPORTS, *submodules, "__version__"})


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
