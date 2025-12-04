"""Project manager for CRUD operations on projects."""

import shutil
from pathlib import Path

import structlog

from konte.config import settings
from konte.models import ProjectConfig
from konte.project import Project

logger = structlog.get_logger()


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
        ValueError: If project already exists.
    """
    path = storage_path or settings.STORAGE_PATH
    project_dir = path / name

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

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        Loaded Project instance.

    Raises:
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
        FileNotFoundError: If project doesn't exist.
    """
    path = storage_path or settings.STORAGE_PATH
    project_dir = path / name

    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {name}")

    shutil.rmtree(project_dir)
    logger.info("project_deleted", name=name)


def project_exists(
    name: str,
    storage_path: Path | None = None,
) -> bool:
    """Check if a project exists.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        True if project exists.
    """
    path = storage_path or settings.STORAGE_PATH
    project_dir = path / name
    return project_dir.exists() and (project_dir / "config.json").exists()
