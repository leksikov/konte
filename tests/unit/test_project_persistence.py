"""Unit tests for saving, opening, and querying a project (BM25 only, no API)."""

import pytest

from konte.domain import Chunk
from konte.project import Project

_CORPUS_ARTIFACTS = frozenset({"chunks", "segments", "contextualized_chunks"})


def _chunk(index: int, content: str) -> Chunk:
    """Build one chunk of a single-segment document."""
    return Chunk(
        chunk_id=f"id{index}",
        content=content,
        source="doc.txt",
        segment_idx=0,
        chunk_idx=index,
    )


async def _built_project(tmp_path, name="persist_test", **config):
    """Create, build and save a lexical-only project."""
    project = Project.create(name=name, storage_path=tmp_path, enable_faiss=False, **config)
    project.corpus.chunks = [
        _chunk(0, "Import duty rates for electronic integrated circuits"),
        _chunk(1, "Tariff classification of clothing dryers"),
    ]
    project.corpus.segments = {("doc.txt", 0): "Full segment text about tariffs"}

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

        assert [c.chunk_id for c in reopened.corpus.chunks] == ["id0", "id1"]
        assert reopened.corpus.segments == {("doc.txt", 0): "Full segment text about tariffs"}
        assert len(reopened.corpus.contextualized_chunks) == 2

    async def test_configured_fusion_weights_reach_the_retriever(self, tmp_path):
        """Test the stored fusion weights reach the retriever."""
        await _built_project(
            tmp_path,
            name="weighted",
            fusion_weight_semantic=0.8,
            fusion_weight_lexical=0.2,
        )

        reopened = Project.open("weighted", storage_path=tmp_path)

        assert reopened._indexes._retriever._fusion_weights == (0.8, 0.2)


@pytest.mark.unit
class TestDeferredArtifacts:
    """Test that opening a project does not parse what it is not asked for."""

    async def test_open_does_not_parse_the_corpus(self, tmp_path):
        """Test that the corpus-sized artifacts stay unparsed until used."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)

        assert not reopened.corpus.loaded & _CORPUS_ARTIFACTS

    async def test_querying_parses_only_the_index_payload(self, tmp_path):
        """Test that a lexical query reads its own chunks and nothing else."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        reopened.query("tariff", mode="lexical")

        assert reopened.corpus.loaded == {"contextualized_chunks"}

    async def test_the_corpus_is_parsed_once_for_every_query(self, tmp_path):
        """Test that repeated queries do not re-read the chunks each time."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        reopened.query("tariff", mode="lexical")
        parsed = reopened.corpus.contextualized_chunks

        reopened.query("circuits", mode="lexical")

        assert reopened.corpus.contextualized_chunks is parsed

    async def test_reading_an_artifact_parses_only_that_one(self, tmp_path):
        """Test that touching one artifact does not drag in the others."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        assert len(reopened.corpus.chunks) == 2

        assert reopened.corpus.loaded == {"chunks"}

    async def test_parsing_happens_once(self, tmp_path):
        """Test that the deferred parse is not repeated on every read."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)

        assert reopened.corpus.chunks is reopened.corpus.chunks

    async def test_assignment_wins_over_a_pending_parse(self, tmp_path):
        """Test that writing an artifact before it is read discards the file."""
        await _built_project(tmp_path)

        reopened = Project.open("persist_test", storage_path=tmp_path)
        reopened.corpus.chunks = []

        assert reopened.corpus.chunks == []

    async def test_a_damaged_artifact_surfaces_at_first_use(self, tmp_path):
        """Test that a build-only artifact is not parsed, damaged or not."""
        await _built_project(tmp_path)
        (tmp_path / "persist_test" / "raw_chunks.json").write_text("{ttorn", encoding="utf-8")

        reopened = Project.open("persist_test", storage_path=tmp_path)

        assert reopened.query("tariff", mode="lexical").total_found > 0
        with pytest.raises(ValueError):
            _ = reopened.corpus.chunks


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

        project.corpus.chunks = [_chunk(0, "Only chunk that remains")]
        project.save()

        reopened = Project.open("persist_test", storage_path=tmp_path)
        assert [c.chunk_id for c in reopened.corpus.chunks] == ["id0"]
        assert reopened.corpus.chunks[0].content == "Only chunk that remains"


def _document(directory, name, body):
    """Write a document long enough to segment, and return its path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{body}\n\n" * 40, encoding="utf-8")
    return path


@pytest.mark.unit
class TestDocumentsAreNamedApart:
    """Test that two documents cannot be filed under one source name.

    A shared name let the second document overwrite the first's segments and
    let fusion merge their chunks into one result.
    """

    def test_same_filename_from_two_directories(self, tmp_path):
        """Test the second `report.md` is filed under enough of its path to differ."""
        project = Project.create(name="named", storage_path=tmp_path / "store")
        project.add_documents(
            [
                _document(tmp_path / "kr", "report.md", "Hwaseong output rose."),
                _document(tmp_path / "us", "report.md", "Taylor output fell."),
            ]
        )

        assert {chunk.source for chunk in project.corpus.chunks} == {"report.md", "us/report.md"}

    def test_chunk_ids_stay_unique(self, tmp_path):
        """Test no two chunks share an id, which fusion reads as one chunk."""
        project = Project.create(name="named", storage_path=tmp_path / "store")
        project.add_documents(
            [
                _document(tmp_path / "kr", "report.md", "Hwaseong output rose."),
                _document(tmp_path / "us", "report.md", "Taylor output fell."),
            ]
        )

        ids = [chunk.chunk_id for chunk in project.corpus.chunks]
        assert len(ids) == len(set(ids))

    def test_neither_document_loses_its_segments(self, tmp_path):
        """Test each document's chunks are contextualized against its own text."""
        project = Project.create(name="named", storage_path=tmp_path / "store")
        project.add_documents(
            [
                _document(tmp_path / "kr", "report.md", "Hwaseong output rose."),
                _document(tmp_path / "us", "report.md", "Taylor output fell."),
            ]
        )

        assert "Hwaseong" in project.corpus.segments[("report.md", 0)]
        assert "Taylor" in project.corpus.segments[("us/report.md", 0)]

    def test_a_lone_filename_is_left_alone(self, tmp_path):
        """Test the common case still files under the bare name callers filter on."""
        project = Project.create(name="named", storage_path=tmp_path / "store")
        project.add_documents([_document(tmp_path / "kr", "report.md", "Hwaseong output rose.")])

        assert {chunk.source for chunk in project.corpus.chunks} == {"report.md"}

    def test_a_reopened_project_still_names_documents_apart(self, tmp_path):
        """Test the names already on disk are what a later add avoids."""
        project = Project.create(name="named", storage_path=tmp_path / "store")
        project.add_documents([_document(tmp_path / "kr", "report.md", "Hwaseong output rose.")])
        project.save()

        reopened = Project.open("named", storage_path=tmp_path / "store")
        reopened.add_documents([_document(tmp_path / "us", "report.md", "Taylor output fell.")])

        assert {chunk.source for chunk in reopened.corpus.chunks} == {"report.md", "us/report.md"}


@pytest.mark.unit
class TestReAddingADocumentIsRefused:
    """Test that the same text cannot enter a project twice, silently."""

    def test_the_same_file_twice(self, tmp_path):
        """Test a repeated `konte add` is reported instead of doubling the corpus."""
        project = Project.create(name="readd", storage_path=tmp_path / "store")
        path = _document(tmp_path / "kr", "report.md", "Hwaseong output rose.")
        project.add_documents([path])

        with pytest.raises(ValueError, match="report.md"):
            project.add_documents([path])

    def test_a_copy_under_another_name(self, tmp_path):
        """Test identity is the text, not the path it arrived by."""
        project = Project.create(name="readd", storage_path=tmp_path / "store")
        project.add_documents([_document(tmp_path / "kr", "report.md", "Hwaseong output rose.")])

        with pytest.raises(ValueError):
            project.add_documents([_document(tmp_path / "us", "copy.md", "Hwaseong output rose.")])

    def test_the_refused_document_leaves_nothing_behind(self, tmp_path):
        """Test the rejected text is not half-added."""
        project = Project.create(name="readd", storage_path=tmp_path / "store")
        path = _document(tmp_path / "kr", "report.md", "Hwaseong output rose.")
        added = project.add_documents([path])

        with pytest.raises(ValueError):
            project.add_documents([path])

        assert len(project.corpus.chunks) == added
        assert {chunk.source for chunk in project.corpus.chunks} == {"report.md"}

    def test_a_different_document_is_still_accepted(self, tmp_path):
        """Test the check refuses copies, not same-named neighbours."""
        project = Project.create(name="readd", storage_path=tmp_path / "store")
        project.add_documents([_document(tmp_path / "kr", "report.md", "Hwaseong output rose.")])

        added = project.add_documents(
            [_document(tmp_path / "us", "report.md", "Taylor output fell.")]
        )

        assert added > 0
