"""Unit tests for the FAISS store's persistence, over a stub embedder."""

import pytest
from langchain_core.embeddings import Embeddings

from konte.domain import Chunk, ContextualizedChunk
from konte.index import ChunkSource, FAISSStore
from konte.index.faiss_store import DOCSTORE_FILENAME, INDEX_FILENAME

_DIMENSIONS = 8


class _StubEmbeddings(Embeddings):
    """Deterministic vectors, so a query reaches the text it came from."""

    def __init__(self):
        self.queries: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * _DIMENSIONS
        for token in text.lower().split():
            vector[sum(ord(character) for character in token) % _DIMENSIONS] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query, recording the request."""
        self.queries.append(text)
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
def embeddings(monkeypatch):
    """Point every store built in a test at one in-process embedder."""
    stub = _StubEmbeddings()
    monkeypatch.setattr("konte.index.faiss_store.OpenAIEmbeddings", lambda **kwargs: stub)
    return stub


@pytest.fixture
def store(embeddings):
    """A store whose embeddings never leave the process."""
    return FAISSStore()


@pytest.mark.unit
class TestFAISSStorePersistence:
    """Test that an index survives the trip through disk intact."""

    def test_save_and_load(self, store, chunks, tmp_path):
        """Test that a saved index loads and holds the same vectors."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path, ChunkSource.holding(chunks))

        assert not reloaded.is_empty
        assert reloaded._index.ntotal == len(chunks)

    def test_save_writes_no_pickle(self, store, chunks, tmp_path):
        """Test that nothing the index writes is read back through pickle."""
        store.build_index(chunks)
        store.save(tmp_path)

        assert not list(tmp_path.glob("*.pkl"))
        assert (tmp_path / INDEX_FILENAME).exists()
        assert (tmp_path / DOCSTORE_FILENAME).exists()

    def test_chunks_survive_a_reload(self, store, chunks, tmp_path):
        """Test that a reloaded index still resolves a hit to the chunk behind it."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path, ChunkSource.holding(chunks))
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
        reloaded.load(tmp_path, ChunkSource.holding(chunks))
        after = reloaded.query("Adobe cloud segment", top_k=3)

        assert [c.chunk.chunk_id for c, _ in after] == [c.chunk.chunk_id for c, _ in before]
        assert [score for _, score in after] == pytest.approx([score for _, score in before])

    def test_filters_still_select_after_reload(self, store, chunks, tmp_path):
        """Test that the id mapping the filtered path walks survives the round trip."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path, ChunkSource.holding(chunks))
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
            FAISSStore().load(tmp_path, ChunkSource.holding(chunks))

    def test_load_rejects_a_docstore_of_another_version(self, store, chunks, tmp_path):
        """Test that a payload this version does not read is refused."""
        import json

        from konte.index.faiss_store import SIGNED_FILENAMES
        from konte.persistence.integrity import sign

        store.build_index(chunks)
        store.save(tmp_path)

        payload = json.loads((tmp_path / DOCSTORE_FILENAME).read_text(encoding="utf-8"))
        payload["version"] = 999
        (tmp_path / DOCSTORE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        sign(tmp_path, SIGNED_FILENAMES)  # the version check, not the record

        with pytest.raises(ValueError, match="version"):
            FAISSStore().load(tmp_path, ChunkSource.holding(chunks))


@pytest.mark.unit
class TestFAISSStoreKeepsNoCorpus:
    """Test that the index carries vectors and the corpus carries the text."""

    def test_the_saved_payload_holds_no_chunk_text(self, store, chunks, tmp_path):
        """Test that nothing beside the vectors repeats what a chunk says."""
        import json

        from konte.index.faiss_store import _DOCSTORE_VERSION

        store.build_index(chunks)
        store.save(tmp_path)

        payload = json.loads((tmp_path / DOCSTORE_FILENAME).read_text(encoding="utf-8"))

        assert payload == {"version": _DOCSTORE_VERSION, "count": len(chunks)}

    def test_load_defers_reading_the_corpus(self, store, chunks, tmp_path):
        """Test that loading an index does not pull its chunks in with it."""
        store.build_index(chunks)
        store.save(tmp_path)

        reads = []

        def corpus():
            reads.append(1)
            return chunks

        reloaded = FAISSStore()
        reloaded.load(tmp_path, ChunkSource(corpus))
        assert reads == []

        reloaded.query("Adobe revenue", top_k=2)
        reloaded.query("Adobe cloud", top_k=2)
        assert reads == [1]

    def test_two_stores_share_one_reading_of_the_corpus(self, store, chunks, tmp_path):
        """Test a hybrid project parses its chunks once, not once per index."""
        from konte.index import BM25Store

        store.build_index(chunks)
        store.save(tmp_path)
        built = BM25Store()
        built.build_index(chunks)
        built.save(tmp_path)

        reads = []

        def corpus():
            reads.append(1)
            return chunks

        shared = ChunkSource(corpus)
        semantic = FAISSStore()
        semantic.load(tmp_path, shared)
        lexical = BM25Store()
        lexical.load(tmp_path, shared)

        semantic.query("Adobe revenue", top_k=2)
        lexical.query("Adobe revenue", top_k=2)

        assert reads == [1]

    def test_a_bundle_binds_both_indexes_to_one_corpus(self, store, chunks, tmp_path):
        """Test the wiring a project opens through reads the corpus once."""
        from konte.domain.config import ProjectConfig
        from konte.domain.corpus import Corpus
        from konte.domain.models import RetrievalRequest
        from konte.index import BM25Store
        from konte.retrieval.bundle import IndexBundle

        store.build_index(chunks)
        store.save(tmp_path)
        lexical = BM25Store()
        lexical.build_index(chunks)
        lexical.save(tmp_path)

        reads = []

        def read():
            reads.append(1)
            return chunks

        corpus = Corpus.deferred(chunks=list, segments=dict, contextualized_chunks=read)
        bundle = IndexBundle.load(
            tmp_path,
            ProjectConfig(name="bundled", storage_path=tmp_path),
            corpus,
        )
        found = bundle.retrieve(
            RetrievalRequest(query="Adobe revenue", mode="hybrid", top_k=2, source_filter="ADOBE")
        )

        assert found.total_found > 0
        assert reads == [1]

    def test_a_corpus_of_another_length_is_refused(self, store, chunks, tmp_path):
        """Test a vector never names whatever chunk happens to sit at its position."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path, ChunkSource.holding(chunks[:-1]))

        with pytest.raises(ValueError, match="Rebuild the project"):
            reloaded.query("Adobe revenue", top_k=2)

    def test_an_absent_corpus_answers_empty(self, store, chunks, tmp_path):
        """Test a directory missing its chunks opens, the way the others do."""
        store.build_index(chunks)
        store.save(tmp_path)

        reloaded = FAISSStore()
        reloaded.load(tmp_path, ChunkSource.holding())

        assert reloaded.query("Adobe revenue", top_k=2) == []

    def test_a_docstore_of_the_previous_version_still_loads(self, store, chunks, tmp_path):
        """Test an index written before this stops repeating the corpus is not re-embedded."""
        import json

        from konte.index.faiss_store import SIGNED_FILENAMES
        from konte.persistence.integrity import sign

        store.build_index(chunks)
        store.save(tmp_path)

        legacy = {
            "version": 1,
            "index_to_id": {str(position): f"id{position}" for position in range(len(chunks))},
            "documents": {
                f"id{position}": {
                    "content": f"{chunk.context} {chunk.chunk.content}",
                    "metadata": {"chunk_id": chunk.chunk.chunk_id},
                }
                for position, chunk in enumerate(chunks)
            },
        }
        (tmp_path / DOCSTORE_FILENAME).write_text(json.dumps(legacy), encoding="utf-8")
        sign(tmp_path, SIGNED_FILENAMES)

        reloaded = FAISSStore()
        reloaded.load(tmp_path, ChunkSource.holding(chunks))
        found = reloaded.query("Adobe revenue", top_k=3)

        assert {chunk.chunk.chunk_id for chunk, _ in found} == {
            chunk.chunk.chunk_id for chunk in chunks
        }


@pytest.mark.unit
class TestFAISSStoreQueryEmbedding:
    """Test that ranking asks for a query's vector once, not once per search."""

    def test_repeated_query_embeds_once(self, store, embeddings, chunks):
        """Test that a repeated search costs the round trip once."""
        store.build_index(chunks)
        embeddings.queries.clear()

        for _ in range(8):
            store.query("Adobe revenue", top_k=2)

        assert embeddings.queries == ["Adobe revenue"]

    def test_filtered_query_shares_the_cache(self, store, embeddings, chunks):
        """Test that adding a filter does not put the same query back on the wire."""
        store.build_index(chunks)
        embeddings.queries.clear()

        store.query("Adobe revenue", top_k=2)
        store.query("Adobe revenue", top_k=2, metadata_filter={"company": "ADOBE"})
        store.query("Adobe revenue", top_k=2, source_filter="ADOBE")

        assert embeddings.queries == ["Adobe revenue"]

    def test_separate_stores_embed_once(self, store, embeddings, chunks):
        """Test that one query fanned out across projects embeds once."""
        store.build_index(chunks)
        other = FAISSStore()
        other.build_index(chunks)
        embeddings.queries.clear()

        store.query("Adobe revenue", top_k=2)
        other.query("Adobe revenue", top_k=2)

        assert embeddings.queries == ["Adobe revenue"]

    def test_cached_query_ranks_the_same(self, store, embeddings, chunks):
        """Test that the second search of a query returns what the first did."""
        store.build_index(chunks)

        first = store.query("Adobe cloud segment", top_k=3)
        second = store.query("Adobe cloud segment", top_k=3)

        assert [chunk.chunk.chunk_id for chunk, _ in second] == [
            chunk.chunk.chunk_id for chunk, _ in first
        ]
        assert [score for _, score in second] == pytest.approx([score for _, score in first])
