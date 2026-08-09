"""Unit tests for the process-level project cache."""

import threading
from unittest.mock import patch

import pytest

from konte.cache import ProjectCache, clear_project_cache, get_shared_project, invalidate_project
from konte.project import Project


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Keep the module-level cache from leaking between tests."""
    clear_project_cache()
    yield
    clear_project_cache()


def _make_project(storage_path, name="proj", **kwargs):
    """Write a queryable, index-free project to disk."""
    project = Project.create(name=name, storage_path=storage_path, **kwargs)
    project.save()
    return project


def _counting_open():
    """Patch Project.open so tests can count how often a project is read."""
    calls = []
    original = Project.open

    def spy(*args, **kwargs):
        calls.append(kwargs.get("name") or args[0])
        return original(*args, **kwargs)

    return patch.object(Project, "open", spy), calls


@pytest.mark.unit
class TestProjectCacheHits:
    """Test that a project is opened once and then reused."""

    def test_repeated_gets_return_the_same_instance(self, tmp_path):
        """Test that the second request does not reopen the project."""
        _make_project(tmp_path)
        cache = ProjectCache()

        first = cache.get("proj", tmp_path)
        second = cache.get("proj", tmp_path)

        assert first is second

    def test_disk_is_read_once(self, tmp_path):
        """Test that repeated queries cost no further reads."""
        _make_project(tmp_path)
        cache = ProjectCache()
        patcher, calls = _counting_open()

        with patcher:
            for _ in range(5):
                cache.get("proj", tmp_path)

        assert len(calls) == 1

    def test_projects_are_kept_apart(self, tmp_path):
        """Test that two names never resolve to one instance."""
        _make_project(tmp_path, name="a")
        _make_project(tmp_path, name="b")
        cache = ProjectCache()

        assert cache.get("a", tmp_path) is not cache.get("b", tmp_path)

    def test_same_name_under_different_storage_paths(self, tmp_path):
        """Test that the storage path is part of a project's identity."""
        first_root = tmp_path / "one"
        second_root = tmp_path / "two"
        _make_project(first_root)
        _make_project(second_root)
        cache = ProjectCache()

        first = cache.get("proj", first_root)
        second = cache.get("proj", second_root)

        assert first is not second
        assert first.project_dir != second.project_dir


@pytest.mark.unit
class TestProjectCacheStaleness:
    """Test that a rebuilt project is not served from before the rebuild."""

    def test_rebuilt_project_is_reopened(self, tmp_path):
        """Test that saving over a project invalidates the cached instance."""
        project = _make_project(tmp_path)
        cache = ProjectCache()
        first = cache.get("proj", tmp_path)

        from konte.models import Chunk

        project._chunks = [
            Chunk(chunk_id="c1", content="new", source="doc.pdf", segment_idx=0, chunk_idx=0)
        ]
        project.save()

        second = cache.get("proj", tmp_path)

        assert second is not first
        assert len(second._chunks) == 1

    def test_untouched_project_is_not_reopened(self, tmp_path):
        """Test that merely checking for staleness does not reopen anything."""
        _make_project(tmp_path)
        cache = ProjectCache()
        patcher, calls = _counting_open()

        with patcher:
            cache.get("proj", tmp_path)
            cache.get("proj", tmp_path)

        assert len(calls) == 1

    def test_missing_project_raises(self, tmp_path):
        """Test that a name with no project behind it is an error, not a hit."""
        cache = ProjectCache()

        with pytest.raises(FileNotFoundError):
            cache.get("absent", tmp_path)


@pytest.mark.unit
class TestProjectCacheEviction:
    """Test the bound on how much the cache holds."""

    def test_least_recently_used_is_evicted(self, tmp_path):
        """Test that the cache does not grow past its limit."""
        _make_project(tmp_path, name="a")
        _make_project(tmp_path, name="b")
        cache = ProjectCache(max_size=1)

        first_a = cache.get("a", tmp_path)
        cache.get("b", tmp_path)

        assert cache.size == 1
        assert cache.get("a", tmp_path) is not first_a

    def test_use_refreshes_recency(self, tmp_path):
        """Test that the project still in use is not the one evicted."""
        for name in ("a", "b", "c"):
            _make_project(tmp_path, name=name)
        cache = ProjectCache(max_size=2)

        first_a = cache.get("a", tmp_path)
        cache.get("b", tmp_path)
        cache.get("a", tmp_path)  # a is now the most recently used
        cache.get("c", tmp_path)  # evicts b, not a

        assert cache.get("a", tmp_path) is first_a

    def test_limit_follows_settings(self, tmp_path, monkeypatch):
        """Test that a cache without an explicit size follows configuration."""
        from konte.config import settings

        monkeypatch.setattr(settings, "PROJECT_CACHE_SIZE", 1)
        _make_project(tmp_path, name="a")
        _make_project(tmp_path, name="b")
        cache = ProjectCache()

        cache.get("a", tmp_path)
        cache.get("b", tmp_path)

        assert cache.size == 1


@pytest.mark.unit
class TestProjectCacheInvalidation:
    """Test explicit removal from the cache."""

    def test_invalidate_forces_a_reopen(self, tmp_path):
        """Test that an invalidated project is opened again on next use."""
        _make_project(tmp_path)
        cache = ProjectCache()

        first = cache.get("proj", tmp_path)
        cache.invalidate("proj", tmp_path)

        assert cache.get("proj", tmp_path) is not first

    def test_clear_empties_the_cache(self, tmp_path):
        """Test that clearing releases everything held."""
        _make_project(tmp_path, name="a")
        _make_project(tmp_path, name="b")
        cache = ProjectCache()
        cache.get("a", tmp_path)
        cache.get("b", tmp_path)

        cache.clear()

        assert cache.size == 0

    def test_delete_project_invalidates(self, tmp_path):
        """Test that a deleted project stops being served from memory."""
        from konte.manager import delete_project

        _make_project(tmp_path)
        get_shared_project("proj", tmp_path)

        delete_project("proj", storage_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            get_shared_project("proj", tmp_path)

    def test_invalidating_an_uncached_project_is_harmless(self, tmp_path):
        """Test that invalidation does not require a cached entry."""
        invalidate_project("never-cached", tmp_path)


@pytest.mark.unit
class TestProjectCacheConcurrency:
    """Test the cache under the worker threads a server calls it from."""

    def test_concurrent_first_requests_open_once(self, tmp_path):
        """Test that a cold project is not opened once per waiting request."""
        _make_project(tmp_path)
        cache = ProjectCache()

        calls = []
        original = Project.open
        start = threading.Barrier(4)

        def slow_open(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        results = []

        def worker():
            start.wait(timeout=5)
            results.append(cache.get("proj", tmp_path))

        with patch.object(Project, "open", slow_open):
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

        assert len(calls) == 1
        assert len(results) == 4
        assert all(result is results[0] for result in results)
