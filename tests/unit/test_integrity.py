"""Unit tests for the records that guard reading an index."""

import json
import pickle
import stat
from pathlib import Path

import pytest

from konte.domain import Chunk, ContextualizedChunk
from konte.index import BM25Store, ChunkSource
from konte.index.bm25_store import INDEX_FILENAME, LEGACY_INDEX_FILENAME, SIGNED_FILENAMES
from konte.manager import trust_project
from konte.persistence.integrity import IntegrityError, verify
from konte.runtime import settings


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


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    """A manifest path, standing in for one committed to a source repository."""
    path = tmp_path / "repo" / "konte-index.lock"
    monkeypatch.setattr(settings, "INDEX_MANIFEST", path)
    return path


def _saved_index(chunks, directory: Path) -> None:
    """Build and save a recorded index in the directory."""
    store = BM25Store()
    store.build_index(chunks)
    store.save(directory)


def _tamper(directory: Path) -> None:
    """Replace the index with a file of the same shape but different contents."""
    path = directory / INDEX_FILENAME
    path.write_bytes(path.read_bytes() + b"\0")


@pytest.mark.unit
class TestIndexSignatures:
    """Test that only an index this installation recorded is read."""

    def test_a_signed_index_loads(self, chunks, project_dir):
        """Test that the signature written on save is the one load accepts."""
        _saved_index(chunks, project_dir)

        store = BM25Store()
        store.load(project_dir, ChunkSource(lambda: chunks))

        assert not store.is_empty

    def test_a_swapped_index_is_refused(self, chunks, project_dir):
        """Test that a replaced index is refused before it is read."""
        _saved_index(chunks, project_dir)
        _tamper(project_dir)

        with pytest.raises(IntegrityError, match="does not match its signature"):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_an_unsigned_index_is_refused(self, chunks, project_dir):
        """Test that an index without a signature is not read on faith."""
        _saved_index(chunks, project_dir)
        (project_dir / f"{INDEX_FILENAME}.sig").unlink()

        with pytest.raises(IntegrityError, match="is not signed"):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_a_corrupt_signature_is_refused_as_a_signature(self, chunks, project_dir):
        """Test that an unparseable signature file fails the check, not the parser."""
        _saved_index(chunks, project_dir)
        (project_dir / f"{INDEX_FILENAME}.sig").write_text("{not json", encoding="utf-8")

        with pytest.raises(IntegrityError):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_an_index_from_another_installation_is_refused(self, chunks, project_dir, monkeypatch):
        """Test that a directory signed elsewhere does not verify here."""
        monkeypatch.setattr(settings, "INDEX_SIGNING_KEY", "the-key-that-signed-it")
        _saved_index(chunks, project_dir)

        monkeypatch.setattr(settings, "INDEX_SIGNING_KEY", "this-installation")
        with pytest.raises(IntegrityError, match="does not match its signature"):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_a_signature_does_not_carry_to_another_name(self, chunks, project_dir):
        """Test that a signature cannot be transplanted onto a different file."""
        _saved_index(chunks, project_dir)

        (project_dir / "other.npz").write_bytes((project_dir / INDEX_FILENAME).read_bytes())
        (project_dir / "other.npz.sig").write_bytes(
            (project_dir / f"{INDEX_FILENAME}.sig").read_bytes()
        )

        with pytest.raises(IntegrityError):
            verify(project_dir, ["other.npz"])

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

    def test_an_unsigned_faiss_docstore_is_refused(self, project_dir, monkeypatch):
        """Test that the docstore is refused before the vector store is built from it."""
        from konte.index import FAISSStore

        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

        (project_dir / "faiss.faiss").write_bytes(b"not an index")
        (project_dir / "faiss_docstore.json").write_text("{}", encoding="utf-8")

        with pytest.raises(IntegrityError, match="is not signed"):
            FAISSStore().load(project_dir, ChunkSource.holding())

    def test_a_configured_key_leaves_no_key_file(self, chunks, project_dir, monkeypatch):
        """Test that INDEX_SIGNING_KEY keeps the secret off the storage root."""
        monkeypatch.setattr(settings, "INDEX_SIGNING_KEY", "from-the-environment")

        _saved_index(chunks, project_dir)
        BM25Store().load(project_dir, ChunkSource(lambda: chunks))

        assert not (project_dir.parent / ".signing-key").exists()


@pytest.mark.unit
class TestIntegrityMode:
    """Test turning the check down to a warning, or off."""

    def test_off_skips_the_check(self, chunks, project_dir, monkeypatch):
        """Test that an index enforce refuses loads once the check is turned off."""
        _saved_index(chunks, project_dir)
        (project_dir / f"{INDEX_FILENAME}.sig").unlink()

        with pytest.raises(IntegrityError):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "off")
        store = BM25Store()
        store.load(project_dir, ChunkSource(lambda: chunks))

        assert not store.is_empty

    def test_off_writes_no_record(self, chunks, project_dir, monkeypatch):
        """Test that saving with the check off leaves no signature behind."""
        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "off")

        _saved_index(chunks, project_dir)

        assert not (project_dir / f"{INDEX_FILENAME}.sig").exists()
        BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_warn_loads_an_unrecorded_index(self, chunks, project_dir, monkeypatch):
        """Test that an unsigned index loads under warn, where enforce refuses it."""
        _saved_index(chunks, project_dir)
        (project_dir / f"{INDEX_FILENAME}.sig").unlink()

        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "warn")
        store = BM25Store()
        store.load(project_dir, ChunkSource(lambda: chunks))

        assert not store.is_empty

    def test_warn_still_records_what_it_writes(self, chunks, project_dir, monkeypatch):
        """Test that a project built under warn verifies under enforce."""
        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "warn")
        _saved_index(chunks, project_dir)

        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "enforce")
        BM25Store().load(project_dir, ChunkSource(lambda: chunks))


@pytest.mark.unit
class TestIndexManifest:
    """Test the digest manifest that travels with the indexes it covers."""

    def test_a_pinned_index_loads(self, chunks, project_dir, manifest):
        """Test that the digest written on save is the one load accepts."""
        _saved_index(chunks, project_dir)

        store = BM25Store()
        store.load(project_dir, ChunkSource(lambda: chunks))

        assert not store.is_empty
        assert manifest.exists()

    def test_the_manifest_holds_the_files_own_digest(self, chunks, project_dir, manifest):
        """Test that a pinned digest is the plain SHA-256 of the file's contents."""
        import hashlib

        _saved_index(chunks, project_dir)

        pinned = json.loads(manifest.read_text(encoding="utf-8"))["projects"]["proj"]
        expected = hashlib.sha256((project_dir / INDEX_FILENAME).read_bytes()).hexdigest()
        assert pinned[INDEX_FILENAME] == expected

    def test_a_swapped_index_is_refused(self, chunks, project_dir, manifest):
        """Test that an index edited after it was pinned does not load."""
        _saved_index(chunks, project_dir)
        _tamper(project_dir)

        with pytest.raises(IntegrityError, match="does not match the digest pinned"):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_an_unpinned_index_is_refused(self, chunks, project_dir, manifest, monkeypatch):
        """Test that an index the manifest never recorded is not read on faith."""
        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "off")
        _saved_index(chunks, project_dir)

        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "enforce")
        with pytest.raises(IntegrityError, match="is not pinned"):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_a_pinned_index_needs_no_local_key(self, chunks, project_dir, manifest):
        """Test that verifying against the manifest generates no signing key."""
        _saved_index(chunks, project_dir)
        BM25Store().load(project_dir, ChunkSource(lambda: chunks))

        assert not (project_dir.parent / ".signing-key").exists()
        assert not (project_dir / f"{INDEX_FILENAME}.sig").exists()

    def test_another_machine_verifies_the_same_manifest(self, chunks, project_dir, manifest):
        """Test that a checkout with no local state loads what the manifest pins."""
        _saved_index(chunks, project_dir)
        (project_dir.parent / ".signing-key").unlink(missing_ok=True)

        elsewhere = project_dir.parent.parent / "checkout" / "proj"
        elsewhere.mkdir(parents=True)
        (elsewhere / INDEX_FILENAME).write_bytes((project_dir / INDEX_FILENAME).read_bytes())

        store = BM25Store()
        store.load(elsewhere, lambda: chunks)

        assert not store.is_empty

    def test_pinning_one_project_leaves_the_others(self, chunks, project_dir, manifest):
        """Test that rebuilding a project does not drop its neighbours' digests."""
        sibling = project_dir.parent / "other"
        sibling.mkdir()
        (sibling / "config.json").write_text("{}", encoding="utf-8")

        _saved_index(chunks, sibling)
        _saved_index(chunks, project_dir)

        pinned = json.loads(manifest.read_text(encoding="utf-8"))["projects"]
        assert set(pinned) == {"other", "proj"}

    def test_a_corrupt_manifest_refuses_rather_than_raises(self, chunks, project_dir, manifest):
        """Test that an unparseable manifest fails the check, not the parser."""
        _saved_index(chunks, project_dir)
        manifest.write_text("{not json", encoding="utf-8")

        with pytest.raises(IntegrityError, match="is not pinned"):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))


@pytest.mark.unit
class TestPickleRefusal:
    """Test that an index a previous version pickled is never read."""

    def test_a_pickled_bm25_index_is_refused_by_name(self, chunks, project_dir):
        """Test that the legacy index file is reported rather than deserialized."""
        marker = project_dir / "executed.txt"

        with (project_dir / LEGACY_INDEX_FILENAME).open("wb") as handle:
            pickle.dump({"index": _Payload(marker), "tokenizer": 2}, handle)

        with pytest.raises(ValueError, match="pickled model"):
            BM25Store().load(project_dir, ChunkSource(lambda: chunks))

        assert not marker.exists()

    def test_a_pickled_faiss_docstore_is_refused_by_name(self, project_dir, monkeypatch):
        """Test that the legacy docstore is reported rather than deserialized."""
        from konte.index import FAISSStore

        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

        (project_dir / "faiss.faiss").write_bytes(b"not an index")
        (project_dir / "faiss.pkl").write_bytes(b"not read either")

        with pytest.raises(ValueError, match="pickled docstore"):
            FAISSStore().load(project_dir, ChunkSource.holding())

    def test_rebuilding_clears_the_legacy_index(self, chunks, project_dir):
        """Test that a saved project leaves no pickle behind for anything to find."""
        (project_dir / LEGACY_INDEX_FILENAME).write_bytes(b"stale")

        _saved_index(chunks, project_dir)

        assert not (project_dir / LEGACY_INDEX_FILENAME).exists()


@pytest.mark.unit
class TestTrustProject:
    """Test adopting index files that were written before they were recorded."""

    def test_trust_records_what_is_on_disk(self, chunks, project_dir):
        """Test that an unrecorded index loads again once it is trusted."""
        _saved_index(chunks, project_dir)
        (project_dir / f"{INDEX_FILENAME}.sig").unlink()

        recorded = trust_project("proj", storage_path=project_dir.parent)

        assert recorded == list(SIGNED_FILENAMES)
        BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_trust_pins_into_the_manifest(self, chunks, project_dir, manifest, monkeypatch):
        """Test that trusting writes the digest when a manifest anchors the record."""
        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "off")
        _saved_index(chunks, project_dir)

        monkeypatch.setattr(settings, "INDEX_INTEGRITY", "enforce")
        trust_project("proj", storage_path=project_dir.parent)

        BM25Store().load(project_dir, ChunkSource(lambda: chunks))

    def test_trust_reports_an_index_free_project(self, project_dir):
        """Test that a project with nothing to record says so instead of failing."""
        assert trust_project("proj", storage_path=project_dir.parent) == []

    def test_trust_requires_the_project(self, tmp_path):
        """Test that recording a project that does not exist is an error."""
        with pytest.raises(FileNotFoundError):
            trust_project("missing", storage_path=tmp_path)
