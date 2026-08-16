"""Process-level cache of projects opened for querying."""

import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import structlog

from konte.config import settings
from konte.models import validate_project_name
from konte.project import Project

logger = structlog.get_logger()

_CONFIG_FILENAME = "config.json"

# Everything Project.load() reads. A rebuild changes the size or modification
# time of at least one of them, which is what marks a loaded instance stale.
_ARTIFACT_FILENAMES = (
    _CONFIG_FILENAME,
    "raw_chunks.json",
    "segments.json",
    "chunks.json",
    "faiss.faiss",
    "faiss.pkl",
    "bm25.pkl",
)

Fingerprint = tuple[tuple[str, int, int], ...]


class _Key(NamedTuple):
    """What makes two requests refer to the same project."""

    name: str
    storage_path: Path


class _Entry(NamedTuple):
    """A loaded project and the on-disk state it was loaded from."""

    project: Project
    fingerprint: Fingerprint


def _fingerprint(directory: Path) -> Fingerprint | None:
    """Summarize a project's files, or return None when it is not a project.

    Args:
        directory: Candidate project directory.

    Returns:
        One (name, modification time, size) triple per artifact present, or
        None when the directory holds no config.json.
    """
    if not (directory / _CONFIG_FILENAME).exists():
        return None

    marks = []
    for filename in _ARTIFACT_FILENAMES:
        try:
            status = (directory / filename).stat()
        except OSError:
            continue
        marks.append((filename, status.st_mtime_ns, status.st_size))
    return tuple(marks)


class ProjectCache:
    """Keeps recently used projects open, one shared instance each.

    Instances handed out here are shared: several callers may query one
    concurrently, which the retrieval path supports, but none may mutate it.
    Use konte.get_project() for anything that adds documents, builds, or saves.
    """

    def __init__(self, max_size: int | None = None) -> None:
        """Create an empty cache.

        Args:
            max_size: How many projects to hold at once. None follows
                settings.PROJECT_CACHE_SIZE on every insert.
        """
        self._max_size = max_size
        self._lock = threading.Lock()
        self._entries: OrderedDict[_Key, _Entry] = OrderedDict()
        self._load_locks: dict[_Key, threading.Lock] = {}

    def get(self, name: str, storage_path: Path | None = None) -> Project:
        """Return the shared instance of a project, opening it if needed.

        Args:
            name: Project name.
            storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

        Returns:
            A loaded project, shared with every other caller holding it.

        Raises:
            ValueError: If the name is not a single path component.
            FileNotFoundError: If no project of that name exists.
        """
        # Checked before the fingerprint, which would otherwise stat outside the root.
        key = _Key(validate_project_name(name), Path(storage_path or settings.STORAGE_PATH))
        directory = key.storage_path / key.name

        fingerprint = _fingerprint(directory)
        if fingerprint is None:
            raise FileNotFoundError(f"Project config not found: {directory / _CONFIG_FILENAME}")

        cached = self._hit(key, fingerprint)
        if cached is not None:
            return cached

        # Opening is slow, so it runs outside the cache lock and under one lock
        # per project: concurrent first requests wait on a single load.
        with self._load_lock(key):
            cached = self._hit(key, fingerprint)
            if cached is not None:
                return cached

            logger.info("project_cache_miss", project=name)
            project = Project.open(name=key.name, storage_path=key.storage_path)

            # Deliberately the fingerprint taken *before* the load: a rebuild
            # that lands while it runs must leave the instance looking stale.
            self._store(key, _Entry(project, fingerprint))
            return project

    def invalidate(self, name: str, storage_path: Path | None = None) -> None:
        """Drop one project, so the next request for it opens the files again."""
        key = _Key(name, Path(storage_path or settings.STORAGE_PATH))
        with self._lock:
            self._entries.pop(key, None)
            self._load_locks.pop(key, None)

    def clear(self) -> None:
        """Forget every cached project."""
        with self._lock:
            self._entries.clear()
            self._load_locks.clear()

    @property
    def size(self) -> int:
        """int: How many projects the cache holds."""
        with self._lock:
            return len(self._entries)

    @property
    def _limit(self) -> int:
        """int: How many projects may be held at once, never fewer than one."""
        configured = self._max_size if self._max_size is not None else settings.PROJECT_CACHE_SIZE
        return max(1, configured)

    def _hit(self, key: _Key, fingerprint: Fingerprint) -> Project | None:
        """Return the cached project when it still matches what is on disk."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.fingerprint != fingerprint:
                return None
            self._entries.move_to_end(key)
            return entry.project

    def _store(self, key: _Key, entry: _Entry) -> None:
        """Insert as most recently used, evicting the oldest past the limit."""
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._limit:
                evicted, _ = self._entries.popitem(last=False)
                logger.info("project_cache_evicted", project=evicted.name)

    def _load_lock(self, key: _Key) -> threading.Lock:
        """Return the lock serializing loads of one project."""
        with self._lock:
            return self._load_locks.setdefault(key, threading.Lock())


_cache = ProjectCache()


def get_shared_project(name: str, storage_path: Path | None = None) -> Project:
    """Get a project for querying, reusing the one already in memory.

    The returned instance is shared across the process. Query it, from several
    threads if the server works that way, but do not add documents, build, or
    save through it — use konte.get_project() for those.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        The shared, loaded project.

    Raises:
        ValueError: If the name is not a single path component.
        FileNotFoundError: If no project of that name exists.
    """
    return _cache.get(name, storage_path)


def invalidate_project(name: str, storage_path: Path | None = None) -> None:
    """Forget one cached project.

    A rebuild is picked up on its own; this is for the cases that leave nothing
    to compare against, such as deleting a project.

    Args:
        name: Project name.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.
    """
    _cache.invalidate(name, storage_path)


def clear_project_cache() -> None:
    """Forget every cached project, releasing the memory they hold."""
    _cache.clear()


def preload_projects(
    names: Sequence[str],
    storage_path: Path | None = None,
) -> list[str]:
    """Open projects ahead of the first request that needs them.

    Args:
        names: Projects to open.
        storage_path: Base storage path. Defaults to settings.STORAGE_PATH.

    Returns:
        The names actually opened; one that fails is logged and skipped.
    """
    loaded = []
    for name in names:
        try:
            get_shared_project(name, storage_path)
        except Exception as error:
            logger.warning("project_preload_failed", project=name, error=str(error))
            continue
        loaded.append(name)

    return loaded
