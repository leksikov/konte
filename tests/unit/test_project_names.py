"""Unit tests for keeping a project name inside its storage root."""

import json

import pytest

from konte.domain import ProjectConfig, validate_project_name
from konte.manager import (
    create_project,
    delete_project,
    get_project,
    project_exists,
    trust_project,
)
from konte.project import Project
from konte.runtime.cache import ProjectCache

ESCAPING_NAMES = [
    "../SECRET",
    "..",
    "../../..",
    "sub/../..",
    ".",
    "",
    "a/b",
    "nested/",
    "/etc",
    "x\0y",
]


@pytest.fixture
def storage(tmp_path):
    """A storage root with a directory beside it that must survive."""
    root = tmp_path / "storage"
    root.mkdir()
    outside = tmp_path / "SECRET"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep me", encoding="utf-8")
    return root


@pytest.mark.unit
class TestNameValidation:
    """Test which names a project may carry."""

    @pytest.mark.parametrize("name", ESCAPING_NAMES)
    def test_escaping_names_are_refused(self, name):
        """Test that a name reaching past one directory is refused."""
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name(name)

    @pytest.mark.parametrize(
        "name",
        ["proj", "my_project", "v1.2", ".hidden", "..leading", "프로젝트", "a b", "trailing."],
    )
    def test_ordinary_names_are_kept(self, name):
        """Test that a name of one directory is returned untouched."""
        assert validate_project_name(name) == name

    @pytest.mark.parametrize("name", ESCAPING_NAMES)
    def test_config_refuses_an_escaping_name(self, name, tmp_path):
        """Test that no project can be configured with one, however it is built."""
        with pytest.raises(ValueError):
            ProjectConfig(name=name, storage_path=tmp_path)


@pytest.mark.unit
class TestDeleteStaysInsideTheRoot:
    """Test that deleting a project cannot remove anything else."""

    def test_parent_directory_is_not_removed(self, storage):
        """Test that '../SECRET' does not delete the sibling of the root."""
        outside = storage.parent / "SECRET"

        with pytest.raises(ValueError, match="Invalid project name"):
            delete_project("../SECRET", storage_path=storage)

        assert (outside / "keep.txt").exists()

    def test_an_absolute_name_is_not_removed(self, storage, tmp_path):
        """Test that an absolute name does not delete the directory it names."""
        target = tmp_path / "elsewhere"
        target.mkdir()

        with pytest.raises(ValueError, match="Invalid project name"):
            delete_project(str(target), storage_path=storage)

        assert target.exists()

    @pytest.mark.parametrize("name", ["", "."])
    def test_the_storage_root_itself_is_not_removed(self, name, storage):
        """Test that a name resolving to the root does not delete every project."""
        create_project("keeper", storage_path=storage).save()

        with pytest.raises(ValueError, match="Invalid project name"):
            delete_project(name, storage_path=storage)

        assert (storage / "keeper" / "config.json").exists()

    def test_a_real_project_still_deletes(self, storage):
        """Test that the check leaves ordinary deletion working."""
        create_project("gone", storage_path=storage).save()

        delete_project("gone", storage_path=storage)

        assert not (storage / "gone").exists()


@pytest.mark.unit
class TestWritesStayInsideTheRoot:
    """Test that creating a project cannot write outside the root."""

    def test_create_refuses_an_escaping_name(self, storage):
        """Test that nothing is written beside the root."""
        with pytest.raises(ValueError, match="Invalid project name"):
            create_project("../outside", storage_path=storage)

        assert not (storage.parent / "outside").exists()

    def test_trust_refuses_an_escaping_name(self, storage):
        """Test that index records are only written for a project of the root."""
        with pytest.raises(ValueError, match="Invalid project name"):
            trust_project("../SECRET", storage_path=storage)


@pytest.mark.unit
class TestReadsStayInsideTheRoot:
    """Test that a project outside the root cannot be addressed by name."""

    @pytest.fixture
    def foreign(self, tmp_path):
        """A project directory outside the storage root."""
        directory = tmp_path / "other" / "foreign"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps({"name": "foreign", "storage_path": "."}), encoding="utf-8"
        )
        return directory

    def test_open_refuses_before_reading_the_config(self, storage, foreign):
        """Test that the name is checked ahead of any file it points at."""
        with pytest.raises(ValueError, match="Invalid project name"):
            Project.open("../other/foreign", storage_path=storage)

    def test_get_project_refuses(self, storage, foreign):
        """Test that the manager entry point is checked too."""
        with pytest.raises(ValueError, match="Invalid project name"):
            get_project("../other/foreign", storage_path=storage)

    def test_the_shared_cache_refuses(self, storage, foreign):
        """Test that the serving path is checked before it stats anything."""
        with pytest.raises(ValueError, match="Invalid project name"):
            ProjectCache().get("../other/foreign", storage)

    def test_existence_is_reported_absent_not_raised(self, storage, foreign):
        """Test that a router asking about an escaping name is answered, not broken."""
        assert project_exists("../other/foreign", storage_path=storage) is False

    def test_a_tampered_config_is_refused(self, storage):
        """Test that a name reaching outside is refused on the way in from disk."""
        directory = storage / "proj"
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"name": "../evil", "storage_path": "."}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="Invalid project name"):
            Project.open("proj", storage_path=storage)
