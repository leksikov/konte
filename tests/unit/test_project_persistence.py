"""Unit tests for saving, opening, and querying a project (BM25 only, no API)."""

import pytest

from konte.models import Chunk
from konte.project import Project

_CORPUS_ARTIFACTS = ("_chunks", "_segments", "_contextualized_chunks")


def _chunk(index: int, content: str) -> Chunk:
    """Build one chunk of a single-segment document."""
    return Chunk(
        chunk_id=f"id{index}",
        content=content,
        source="doc.txt",
        segment_idx=0,
        chunk_idx=index,
    )


async def _built_project(tmp_path, name="persist_test"):
    """Create, build and save a lexical-only project."""
    project = Project.create(name=name, storage_path=tmp_path, enable_faiss=False)
    project._chunks = [
        _chunk(0, "Import duty rates for electronic integrated circuits"),
        _chunk(1, "Tariff classification of clothing dryers"),
    ]
    project._segments = {("doc.txt", 0): "Full segment text about tariffs"}

    await project.build(skip_context=True, enable_faiss=False)
    project.save()
    return project


@pytest.mark.unit
class TestRoundTrip:
    """Test that a saved project comes back queryable."""

    async def test_reopened_project_answers_queries(self, tmp_path):
        """Test that opening a saved project restores its lexical index."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        response = reopened.query("integrated circuits", mode="lexical")

        assert response.total_found > 0
        assert "integrated circuits" in response.results[0].content

    async def test_reopened_project_restores_the_corpus(self, tmp_path):
        """Test that the stored chunks survive the round trip."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)

        assert [c.chunk_id for c in reopened._chunks] == ["id0", "id1"]
        assert reopened._segments == {("doc.txt", 0): "Full segment text about tariffs"}
        assert len(reopened._contextualized_chunks) == 2


@pytest.mark.unit
class TestDeferredArtifacts:
    """Test that opening a project does not parse what it is not asked for."""

    async def test_open_does_not_parse_the_corpus(self, tmp_path):
        """Test that the corpus-sized artifacts stay unparsed until used."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)

        assert not any(name in reopened.__dict__ for name in _CORPUS_ARTIFACTS)

    async def test_querying_parses_only_the_index_payload(self, tmp_path):
        """Test that a lexical query reads its own chunks and nothing else."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        reopened.query("tariff", mode="lexical")

        assert "_contextualized_chunks" in reopened.__dict__
        assert "_chunks" not in reopened.__dict__
        assert "_segments" not in reopened.__dict__

    async def test_the_corpus_is_parsed_once_for_every_query(self, tmp_path):
        """Test that repeated queries do not re-read the chunks each time."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        reopened.query("tariff", mode="lexical")
        parsed = reopened._contextualized_chunks

        reopened.query("circuits", mode="lexical")

        assert reopened._contextualized_chunks is parsed

    async def test_reading_an_artifact_parses_only_that_one(self, tmp_path):
        """Test that touching one artifact does not drag in the others."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        assert len(reopened._chunks) == 2

        assert "_chunks" in reopened.__dict__
        assert "_contextualized_chunks" not in reopened.__dict__

    async def test_parsing_happens_once(self, tmp_path):
        """Test that the deferred parse is not repeated on every read."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)

        assert reopened._chunks is reopened._chunks

    async def test_assignment_wins_over_a_pending_parse(self, tmp_path):
        """Test that writing an artifact before it is read discards the file."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        reopened._chunks = []

        assert reopened._chunks == []

    async def test_a_damaged_artifact_surfaces_at_first_use(self, tmp_path):
        """Test that a build-only artifact is not parsed, damaged or not."""
        await _built_project(tmp_path)
        (tmp_path / "persist_test" / "raw_chunks.json").write_text("{ttorn", encoding="utf-8")

        reopened = Project.open("persist_test", storage_path=tmp_path)

        assert reopened.query("tariff", mode="lexical").total_found > 0
        with pytest.raises(ValueError):
            _ = reopened._chunks


@pytest.mark.unit
class TestSaveDurability:
    """Test what save() leaves on disk."""

    async def test_no_temporary_files_remain(self, tmp_path):
        """Test that the atomic writes clean up after themselves."""
        await _built_project(tmp_path)

        leftovers = [p.name for p in (tmp_path / "persist_test").iterdir() if ".tmp" in p.name]

        assert leftovers == []

    async def test_rewriting_a_project_replaces_it_wholesale(self, tmp_path):
        """Test that a second save leaves no trace of the first."""
        project = await _built_project(tmp_path)

        project._chunks = [_chunk(0, "Only chunk that remains")]
        project.save()

        reopened = Project.open("persist_test", storage_path=tmp_path)
        assert [c.chunk_id for c in reopened._chunks] == ["id0"]
        assert reopened._chunks[0].content == "Only chunk that remains"
