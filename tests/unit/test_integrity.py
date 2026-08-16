"""Unit tests for the signatures that guard deserializing an index."""

import pickle
import stat
from pathlib import Path

import pytest

from konte.config import settings
from konte.integrity import IntegrityError, verify
from konte.manager import trust_project
from konte.models import Chunk, ContextualizedChunk
from konte.stores import BM25Store
from konte.stores.bm25_store import SIGNED_FILENAMES


def _record_execution(marker: str) -> None:
    """Leave proof that a pickle got to run something."""
    Path(marker).write_text("executed", encoding="utf-8")


class _Payload:
    """A pickle that runs code the moment it is read."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (_record_execution, (str(self.marker),))


@pytest.fixture
def chunks():
    """A small corpus to build an index from."""
    return [
        ContextualizedChunk(
            chunk=Chunk(
                chunk_id=f"test_s0_c{i}",
                content=content,
                source="test.txt",
                segment_idx=0,
                chunk_idx=i,
            ),
            context="This section covers tariff classification.",
        )
        for i, content in enumerate(
            [
                "Tariff code 8542.31 applies to electronic integrated circuits.",
                "Import duty rate is 5% for MFN countries.",
                "Customs valuation follows WTO Agreement rules.",
            ]
        )
    ]


@pytest.fixture
def project_dir(tmp_path):
    """A project directory under its own storage root."""
    directory = tmp_path / "storage" / "proj"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def _saved_index(chunks, directory: Path) -> None:
    """Build and save a signed index in the directory."""
    store = BM25Store()
    store.build_index(chunks)
    store.save(directory)


@pytest.mark.unit
class TestIndexSignatures:
    """Test that only an index this installation signed is deserialized."""

    def test_a_signed_index_loads(self, chunks, project_dir):
        """Test that the signature written on save is the one load accepts."""
        _saved_index(chunks, project_dir)

        store = BM25Store()
        store.load(project_dir, lambda: chunks)

        assert not store.is_empty

    def test_a_swapped_index_never_runs(self, chunks, project_dir, tmp_path):
        """Test that a replaced pickle is refused before it is deserialized."""
        marker = tmp_path / "executed.txt"
        _saved_index(chunks, project_dir)

        with (project_dir / "bm25.pkl").open("wb") as handle:
            pickle.dump({"index": _Payload(marker), "tokenizer": 2}, handle)

        with pytest.raises(IntegrityError, match="does not match its signature"):
            BM25Store().load(project_dir, lambda: chunks)

        assert not marker.exists()

    def test_an_unsigned_index_is_refused(self, chunks, project_dir):
        """Test that an index without a signature is not read on faith."""
        _saved_index(chunks, project_dir)
        (project_dir / "bm25.pkl.sig").unlink()

        with pytest.raises(IntegrityError, match="is not signed"):
            BM25Store().load(project_dir, lambda: chunks)

    def test_an_index_from_another_installation_is_refused(
        self, chunks, project_dir, monkeypatch
    ):
        """Test that a directory signed elsewhere does not verify here."""
        monkeypatch.setattr(settings, "INDEX_SIGNING_KEY", "the-key-that-signed-it")
        _saved_index(chunks, project_dir)

        monkeypatch.setattr(settings, "INDEX_SIGNING_KEY", "this-installation")
        with pytest.raises(IntegrityError, match="does not match its signature"):
            BM25Store().load(project_dir, lambda: chunks)

    def test_a_signature_does_not_carry_to_another_name(self, chunks, project_dir):
        """Test that a signature cannot be transplanted onto a different file."""
        _saved_index(chunks, project_dir)

        (project_dir / "other.pkl").write_bytes((project_dir / "bm25.pkl").read_bytes())
        (project_dir / "other.pkl.sig").write_bytes(
            (project_dir / "bm25.pkl.sig").read_bytes()
        )

        with pytest.raises(IntegrityError):
            verify(project_dir, ["other.pkl"])

    def test_the_key_stays_outside_the_directory_it_signs(self, chunks, project_dir):
        """Test that a copied project directory carries no key to verify itself with."""
        _saved_index(chunks, project_dir)

        assert not any(path.name == ".signing-key" for path in project_dir.iterdir())
        assert (project_dir.parent / ".signing-key").exists()

    def test_the_key_is_readable_only_by_its_owner(self, chunks, project_dir):
        """Test that the local key is not left world-readable."""
        _saved_index(chunks, project_dir)

        mode = (project_dir.parent / ".signing-key").stat().st_mode
        assert stat.S_IMODE(mode) == 0o600

    def test_an_unsigned_faiss_docstore_never_runs(self, project_dir, tmp_path, monkeypatch):
        """Test that the docstore pickle is refused before LangChain unpickles it."""
        from konte.stores import FAISSStore

        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
        marker = tmp_path / "faiss_executed.txt"

        (project_dir / "faiss.faiss").write_bytes(b"not an index")
        with (project_dir / "faiss.pkl").open("wb") as handle:
            pickle.dump(_Payload(marker), handle)

        with pytest.raises(IntegrityError, match="is not signed"):
            FAISSStore().load(project_dir)

        assert not marker.exists()

    def test_a_configured_key_leaves_no_key_file(self, chunks, project_dir, monkeypatch):
        """Test that INDEX_SIGNING_KEY keeps the secret off the storage root."""
        monkeypatch.setattr(settings, "INDEX_SIGNING_KEY", "from-the-environment")

        _saved_index(chunks, project_dir)
        BM25Store().load(project_dir, lambda: chunks)

        assert not (project_dir.parent / ".signing-key").exists()


@pytest.mark.unit
class TestTrustProject:
    """Test adopting index files that were written before they were signed."""

    def test_trust_signs_what_is_on_disk(self, chunks, project_dir):
        """Test that a legacy index loads again once it is signed."""
        _saved_index(chunks, project_dir)
        (project_dir / "bm25.pkl.sig").unlink()

        signed = trust_project("proj", storage_path=project_dir.parent)

        assert signed == list(SIGNED_FILENAMES)
        BM25Store().load(project_dir, lambda: chunks)

    def test_trust_reports_an_index_free_project(self, project_dir):
        """Test that a project with nothing to sign says so instead of failing."""
        assert trust_project("proj", storage_path=project_dir.parent) == []

    def test_trust_requires_the_project(self, tmp_path):
        """Test that signing a project that does not exist is an error."""
        with pytest.raises(FileNotFoundError):
            trust_project("missing", storage_path=tmp_path)
