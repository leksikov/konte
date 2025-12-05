"""Unit tests for CLI module."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

runner = CliRunner()


@pytest.mark.unit
class TestCLIHelp:
    """Test CLI help and basic functionality."""

    def test_app_has_help(self):
        """Test that app shows help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Contextual RAG CLI" in result.stdout

    def test_create_command_has_help(self):
        """Test create command help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0
        assert "Create a new project" in result.stdout

    def test_list_command_has_help(self):
        """Test list command help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "List all projects" in result.stdout

    def test_delete_command_has_help(self):
        """Test delete command help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["delete", "--help"])
        assert result.exit_code == 0
        assert "Delete a project" in result.stdout

    def test_add_command_has_help(self):
        """Test add command help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["add", "--help"])
        assert result.exit_code == 0
        assert "Add documents to a project" in result.stdout

    def test_build_command_has_help(self):
        """Test build command help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["build", "--help"])
        assert result.exit_code == 0
        assert "Build indexes" in result.stdout

    def test_query_command_has_help(self):
        """Test query command help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "Query a project" in result.stdout

    def test_info_command_has_help(self):
        """Test info command help."""
        from konte.cli.app import app

        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "Show project information" in result.stdout


@pytest.mark.unit
class TestCLICreateCommand:
    """Test CLI create command with mocked dependencies."""

    def test_create_project_success(self, tmp_path):
        """Test creating a project successfully."""
        from konte.cli.app import app

        result = runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Created project" in result.stdout
        assert "test_project" in result.stdout

    def test_create_duplicate_project_fails(self, tmp_path):
        """Test creating duplicate project fails."""
        from konte.cli.app import app

        # Create first
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        # Try to create again
        result = runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "already exists" in result.stdout


@pytest.mark.unit
class TestCLIListCommand:
    """Test CLI list command."""

    def test_list_empty(self, tmp_path):
        """Test listing with no projects."""
        from konte.cli.app import app

        result = runner.invoke(app, ["list", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "No projects found" in result.stdout

    def test_list_with_projects(self, tmp_path):
        """Test listing existing projects."""
        from konte.cli.app import app

        # Create projects
        runner.invoke(app, ["create", "project1", "--storage", str(tmp_path)])
        runner.invoke(app, ["create", "project2", "--storage", str(tmp_path)])

        result = runner.invoke(app, ["list", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "project1" in result.stdout
        assert "project2" in result.stdout


@pytest.mark.unit
class TestCLIDeleteCommand:
    """Test CLI delete command."""

    def test_delete_project_with_force(self, tmp_path):
        """Test deleting a project with --force."""
        from konte.cli.app import app

        # Create project
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        # Delete with force
        result = runner.invoke(app, ["delete", "test_project", "--storage", str(tmp_path), "--force"])
        assert result.exit_code == 0
        assert "Deleted project" in result.stdout

    def test_delete_nonexistent_fails(self, tmp_path):
        """Test deleting non-existent project fails."""
        from konte.cli.app import app

        result = runner.invoke(app, ["delete", "nonexistent", "--storage", str(tmp_path), "--force"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_delete_cancelled_without_force(self, tmp_path):
        """Test delete is cancelled without confirmation."""
        from konte.cli.app import app

        # Create project
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        # Try to delete without force, provide 'n' for no
        result = runner.invoke(app, ["delete", "test_project", "--storage", str(tmp_path)], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.stdout


@pytest.mark.unit
class TestCLIInfoCommand:
    """Test CLI info command."""

    def test_info_existing_project(self, tmp_path):
        """Test showing info for existing project."""
        from konte.cli.app import app

        # Create project
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        result = runner.invoke(app, ["info", "test_project", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "test_project" in result.stdout
        assert "Segment Size" in result.stdout
        assert "Chunk Size" in result.stdout

    def test_info_nonexistent_fails(self, tmp_path):
        """Test info for non-existent project fails."""
        from konte.cli.app import app

        result = runner.invoke(app, ["info", "nonexistent", "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "not found" in result.stdout


@pytest.mark.unit
class TestCLIAddCommand:
    """Test CLI add command."""

    def test_add_nonexistent_project_fails(self, tmp_path):
        """Test adding to non-existent project fails."""
        from konte.cli.app import app

        result = runner.invoke(app, ["add", "nonexistent", str(tmp_path / "doc.txt"), "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_add_nonexistent_file_fails(self, tmp_path):
        """Test adding non-existent file fails."""
        from konte.cli.app import app

        # Create project
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        result = runner.invoke(app, ["add", "test_project", str(tmp_path / "nonexistent.txt"), "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "File not found" in result.stdout


@pytest.mark.unit
class TestCLIQueryCommand:
    """Test CLI query command."""

    def test_query_nonexistent_project_fails(self, tmp_path):
        """Test querying non-existent project fails."""
        from konte.cli.app import app

        result = runner.invoke(app, ["query", "nonexistent", "test query", "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_query_invalid_mode_fails(self, tmp_path):
        """Test query with invalid mode fails."""
        from konte.cli.app import app

        # Create project
        runner.invoke(app, ["create", "test_project", "--storage", str(tmp_path)])

        result = runner.invoke(app, ["query", "test_project", "test", "--mode", "invalid", "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "Invalid mode" in result.stdout


@pytest.mark.unit
class TestCLIBuildCommand:
    """Test CLI build command."""

    def test_build_nonexistent_project_fails(self, tmp_path):
        """Test building non-existent project fails."""
        from konte.cli.app import app

        result = runner.invoke(app, ["build", "nonexistent", "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "not found" in result.stdout
