"""Unit tests for the FAISS store's persistence, over a stub embedder."""

import pytest
from langchain_core.embeddings import Embeddings

from konte.models import Chunk, ContextualizedChunk
from konte.stores import FAISSStore
from konte.stores.faiss_store import DOCSTORE_FILENAME, INDEX_FILENAME

_DIMENSIONS = 8


class _StubEmbeddings(Embeddings):
    """Deterministic vectors, so a query reaches the text it came from."""

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * _DIMENSIONS
        for token in text.lower().split():
            vector[sum(ord(character) for character in token) % _DIMENSIONS] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        return self._vector(text)


@pytest.fixture
def chunks():
    """A small corpus carrying metadata worth round-tripping."""
    rows = [
        ("ADOBE_2022_10K.md", "Adobe revenue grew in fiscal 2022.", {"company": "ADOBE"}),
        ("ADOBE_2023_10K.md", "Adobe cloud segment led the growth.", {"company": "ADOBE"}),
        ("3M_2022_10K.md", "3M industrial results for the year.", {"company": "3M"}),
    ]
    return [
        ContextualizedChunk(
            chunk=Chunk(
                chunk_id=f"{source}_s0_c{i}",
                content=content,
                source=source,
                segment_idx=0,
                chunk_idx=i,
                metadata=metadata,
            ),
            context="This section covers annual results.",
        )
        for i, (source, content, metadata) in enumerate(rows)
    ]


@pytest.fixture
def store(monkeypatch):
    """A store whose embeddings never leave the process."""
    monkeypatch.setattr(
        "konte.stores.faiss_store.OpenAIEmbeddings",
        lambda **kwargs: _StubEmbeddings(),
    )
    return FAISSStore()


@pytest.mark.unit
class TestFAISSStorePersistence:
    """Test that an index survives the trip through disk intact."""

    def test_save_and_load(self, store, chunks, tmp_path):
        """Test that a saved index loads and holds the same vectors."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path)

        assert not reloaded.is_empty
        assert reloaded._vectorstore.index.ntotal == len(chunks)

    def test_save_writes_no_pickle(self, store, chunks, tmp_path):
        """Test that nothing the index writes is read back through pickle."""
        store.build_index(chunks)
        store.save(tmp_path)

        assert not list(tmp_path.glob("*.pkl"))
        assert (tmp_path / INDEX_FILENAME).exists()
        assert (tmp_path / DOCSTORE_FILENAME).exists()

    def test_chunks_survive_a_reload(self, store, chunks, tmp_path):
        """Test that a reloaded document rebuilds the chunk it was flattened from."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path)
        found = reloaded.query("Adobe revenue", top_k=3)

        rebuilt = {chunk.chunk.chunk_id: chunk for chunk, _ in found}
        assert set(rebuilt) == {chunk.chunk.chunk_id for chunk in chunks}
        for original in chunks:
            restored = rebuilt[original.chunk.chunk_id]
            assert restored.chunk.content == original.chunk.content
            assert restored.chunk.source == original.chunk.source
            assert restored.context == original.context
            assert restored.chunk.metadata == original.chunk.metadata

    def test_query_results_same_after_reload(self, store, chunks, tmp_path):
        """Test that reloading changes neither the ranking nor the scores."""
        store.build_index(chunks)
        before = store.query("Adobe cloud segment", top_k=3)

        store.save(tmp_path)
        reloaded = FAISSStore()
        reloaded.load(tmp_path)
        after = reloaded.query("Adobe cloud segment", top_k=3)

        assert [c.chunk.chunk_id for c, _ in after] == [c.chunk.chunk_id for c, _ in before]
        assert [score for _, score in after] == pytest.approx([score for _, score in before])

    def test_filters_still_select_after_reload(self, store, chunks, tmp_path):
        """Test that the id mapping the filtered path walks survives the round trip."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path)
        found = reloaded.query("results", top_k=10, metadata_filter={"company": "ADOBE"})

        assert {chunk.chunk.source for chunk, _ in found} == {
            "ADOBE_2022_10K.md",
            "ADOBE_2023_10K.md",
        }

    def test_load_reports_a_missing_docstore(self, store, chunks, tmp_path):
        """Test that an index without its docstore is named, not half-loaded."""
        store.build_index(chunks)
        store.save(tmp_path)
        (tmp_path / DOCSTORE_FILENAME).unlink()

        with pytest.raises(FileNotFoundError, match="docstore"):
            FAISSStore().load(tmp_path)

    def test_load_rejects_a_docstore_of_another_version(self, store, chunks, tmp_path):
        """Test that a payload this version does not read is refused."""
        import json

        from konte.integrity import sign
        from konte.stores.faiss_store import SIGNED_FILENAMES

        store.build_index(chunks)
        store.save(tmp_path)

        payload = json.loads((tmp_path / DOCSTORE_FILENAME).read_text(encoding="utf-8"))
        payload["version"] = 999
        (tmp_path / DOCSTORE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        sign(tmp_path, SIGNED_FILENAMES)  # the version check, not the record

        with pytest.raises(ValueError, match="version"):
            FAISSStore().load(tmp_path)
