"""Project manager for CRUD operations on projects."""

import shutil
from pathlib import Path

import structlog

from konte.cache import invalidate_project
from konte.config import settings
from konte.integrity import sign
from konte.models import validate_project_name
from konte.project import Project
from konte.stores.bm25_store import SIGNED_FILENAMES as _BM25_FILENAMES
from konte.stores.faiss_store import SIGNED_FILENAMES as _FAISS_FILENAMES

logger = structlog.get_logger()

_INDEX_FILENAMES = (*_FAISS_FILENAMES, *_BM25_FILENAMES)


def _locate(name: str, storage_path: Path | None) -> tuple[Path, Path]:
    """Return the storage root and the directory the named project sits in."""
    path = storage_path or settings.STORAGE_PATH
    return path, path / validate_project_name(name)


def create_project(
    name: str,
    storage_path: Path | None = None,
    **kwargs,
) -> Project:
    """Create a new project.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.
        **kwargs: Additional ProjectConfig parameters.

    Returns:
        New Project instance.

    Raises:
        ValueError: If the name is not a single path component, or a project of
            that name already exists.
    """
    path, project_dir = _locate(name, storage_path)

    if project_dir.exists():
        raise ValueError(f"Project already exists: {name}")

    project = Project.create(name=name, storage_path=path, **kwargs)
    logger.info("project_created", name=name, path=str(project_dir))
    return project


def list_projects(storage_path: Path | None = None) -> list[str]:
    """List all projects.

    Args:
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        List of project names.
    """
    path = storage_path or settings.STORAGE_PATH

    if not path.exists():
        return []

    projects = []
    for item in path.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            projects.append(item.name)

    return sorted(projects)


def get_project(
    name: str,
    storage_path: Path | None = None,
) -> Project:
    """Get an existing project.

    Each call opens its own instance, free to mutate. Code that only queries
    wants konte.get_shared_project() instead.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        Loaded Project instance.

    Raises:
        ValueError: If the name is not a single path component.
        FileNotFoundError: If project doesn't exist.
    """
    return Project.open(name=name, storage_path=storage_path)


def delete_project(
    name: str,
    storage_path: Path | None = None,
) -> None:
    """Delete a project and all its data.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Raises:
        ValueError: If the name is not a single path component.
        FileNotFoundError: If project doesn't exist.
    """
    path, project_dir = _locate(name, storage_path)

    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {name}")

    shutil.rmtree(project_dir)
    invalidate_project(name, storage_path=path)
    logger.info("project_deleted", name=name)


def trust_project(
    name: str,
    storage_path: Path | None = None,
) -> list[str]:
    """Record the index files a project already has on disk as trusted.

    Recording says the files are trusted as they stand, which trusts anything
    that reached the directory unnoticed as well. Rebuilding is the answer
    wherever that is in doubt.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        The names of the files recorded.

    Raises:
        ValueError: If the name is not a single path component.
        FileNotFoundError: If project doesn't exist.
    """
    _, project_dir = _locate(name, storage_path)

    if not (project_dir / "config.json").exists():
        raise FileNotFoundError(f"Project not found: {name}")

    signed = [filename for filename in _INDEX_FILENAMES if (project_dir / filename).exists()]
    sign(project_dir, signed)

    logger.info("project_trusted", name=name, files=signed)
    return signed


def project_exists(
    name: str,
    storage_path: Path | None = None,
) -> bool:
    """Check if a project exists.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        True if project exists. A name that reaches outside the storage root
        is absent rather than an error, so a router can answer on it.
    """
    try:
        _, project_dir = _locate(name, storage_path)
    except ValueError:
        return False
    return project_dir.exists() and (project_dir / "config.json").exists()
