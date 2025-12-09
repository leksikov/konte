"""Integration tests for project manager."""

import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.mark.integration
class TestCreateProject:
    """Test project creation via manager."""

    def test_create_new_project(self, tmp_path):
        """Test creating a new project."""
        from konte.manager import create_project

        project = create_project(name="test", storage_path=tmp_path)

        assert project.config.name == "test"
        assert project.config.storage_path == tmp_path

    def test_create_duplicate_raises(self, tmp_path):
        """Test that creating duplicate project raises error."""
        from konte.manager import create_project

        create_project(name="test", storage_path=tmp_path)
        project_dir = tmp_path / "test"
        project_dir.mkdir(exist_ok=True)

        with pytest.raises(ValueError, match="already exists"):
            create_project(name="test", storage_path=tmp_path)


@pytest.mark.integration
class TestListProjects:
    """Test listing projects."""

    def test_list_empty(self, tmp_path):
        """Test listing projects in empty directory."""
        from konte.manager import list_projects

        projects = list_projects(storage_path=tmp_path)
        assert projects == []

    async def test_list_multiple_projects(self, tmp_path):
        """Test listing multiple projects."""
        from konte.manager import create_project, list_projects

        # Create projects
        p1 = create_project(name="alpha", storage_path=tmp_path)
        p2 = create_project(name="beta", storage_path=tmp_path)

        # Add documents and save to create config files
        p1.add_documents([FIXTURES_DIR / "sample.txt"])
        await p1.build(skip_context=False)
        p1.save()

        p2.add_documents([FIXTURES_DIR / "sample.txt"])
        await p2.build(skip_context=False)
        p2.save()

        projects = list_projects(storage_path=tmp_path)

        assert "alpha" in projects
        assert "beta" in projects
        assert len(projects) == 2

    def test_list_nonexistent_path(self, tmp_path):
        """Test listing in nonexistent path returns empty."""
        from konte.manager import list_projects

        projects = list_projects(storage_path=tmp_path / "nonexistent")
        assert projects == []


@pytest.mark.integration
class TestGetProject:
    """Test getting projects."""

    async def test_get_existing_project(self, tmp_path):
        """Test getting an existing project."""
        from konte.manager import create_project, get_project

        project1 = create_project(name="test", storage_path=tmp_path)
        project1.add_documents([FIXTURES_DIR / "sample.txt"])
        await project1.build(skip_context=False)
        project1.save()

        project2 = get_project(name="test", storage_path=tmp_path)

        assert project2.config.name == "test"

    def test_get_nonexistent_raises(self, tmp_path):
        """Test that getting nonexistent project raises error."""
        from konte.manager import get_project

        with pytest.raises(FileNotFoundError):
            get_project(name="nonexistent", storage_path=tmp_path)


@pytest.mark.integration
class TestDeleteProject:
    """Test deleting projects."""

    async def test_delete_project(self, tmp_path):
        """Test deleting a project."""
        from konte.manager import create_project, delete_project, list_projects

        project = create_project(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=False)
        project.save()

        delete_project(name="test", storage_path=tmp_path)

        projects = list_projects(storage_path=tmp_path)
        assert "test" not in projects

    def test_delete_nonexistent_raises(self, tmp_path):
        """Test that deleting nonexistent project raises error."""
        from konte.manager import delete_project

        with pytest.raises(FileNotFoundError):
            delete_project(name="nonexistent", storage_path=tmp_path)


@pytest.mark.integration
class TestProjectExists:
    """Test project existence check."""

    async def test_exists_true(self, tmp_path):
        """Test that exists returns True for existing project."""
        from konte.manager import create_project, project_exists

        project = create_project(name="test", storage_path=tmp_path)
        project.add_documents([FIXTURES_DIR / "sample.txt"])
        await project.build(skip_context=False)
        project.save()

        assert project_exists(name="test", storage_path=tmp_path) is True

    def test_exists_false(self, tmp_path):
        """Test that exists returns False for nonexistent project."""
        from konte.manager import project_exists

        assert project_exists(name="nonexistent", storage_path=tmp_path) is False
