"""Unit tests for BM25 store (no external API required)."""

import pytest

from konte.domain import Chunk, ContextualizedChunk


@pytest.fixture
def sample_chunks():
    """Create sample contextualized chunks for testing."""
    chunks = []
    contents = [
        ("Tariff code 8542.31 applies to electronic integrated circuits.",
         "This section covers semiconductor devices under Chapter 85."),
        ("Import duty rate is 5% for MFN countries.",
         "This discusses duty rates for electronics from WTO member nations."),
        ("Anti-dumping duties may apply to certain imports.",
         "This explains special duties imposed on unfairly priced goods."),
        ("The Harmonized System uses 6-digit codes internationally.",
         "This describes the international tariff classification standard."),
        ("Customs valuation follows WTO Agreement rules.",
         "This covers how import values are determined for duty calculation."),
    ]
    for i, (content, context) in enumerate(contents):
        chunk = Chunk(
            chunk_id=f"test_s0_c{i}",
            content=content,
            source="test.txt",
            segment_idx=0,
            chunk_idx=i,
        )
        chunks.append(ContextualizedChunk(chunk=chunk, context=context))
    return chunks


@pytest.mark.unit
class TestBM25StoreBuild:
    """Test BM25 store build functionality."""

    def test_build_index_creates_index(self, sample_chunks):
        """Test that build_index creates a searchable index."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        assert not store.is_empty

    def test_build_empty_chunks(self):
        """Test building index with empty chunks list."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index([])

        assert store.is_empty


@pytest.mark.unit
class TestBM25StoreQuery:
    """Test BM25 store query functionality."""

    def test_query_returns_results(self, sample_chunks):
        """Test that query returns relevant results."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.query("tariff code electronics", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r[0], ContextualizedChunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_query_scores_in_range(self, sample_chunks):
        """Test that query scores are between 0 and 1."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.query("tariff classification", top_k=5)

        for _chunk, score in results:
            assert 0.0 <= score <= 1.0

    def test_query_exact_match_scores_high(self, sample_chunks):
        """Test that exact keyword matches score higher."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        results = store.query("8542.31", top_k=3)

        # The chunk with exact tariff code should be first
        top_content = results[0][0].chunk.content.lower()
        assert "8542.31" in top_content

    def test_query_empty_index(self):
        """Test querying empty index returns empty list."""
        from konte.index import BM25Store

        store = BM25Store()
        results = store.query("test query")

        assert results == []

    def test_query_no_matches(self, sample_chunks):
        """Test query with no matching terms still returns results (ranked by BM25)."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        # Query with terms not in corpus
        results = store.query("xyz123 completely unrelated", top_k=3)

        # BM25 will return results with 0 scores
        assert len(results) == 3


@pytest.mark.unit
class TestBM25Tokenization:
    """Test that terms survive the punctuation and particles attached to them."""

    @staticmethod
    def _store(contents):
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(
            [
                ContextualizedChunk(
                    chunk=Chunk(
                        chunk_id=f"c{i}",
                        content=text,
                        source="t.txt",
                        segment_idx=0,
                        chunk_idx=i,
                    ),
                    context="",
                )
                for i, text in enumerate(contents)
            ]
        )
        return store

    def test_punctuation_does_not_hide_a_term(self):
        """Test a term ending a clause is the same term as one that does not."""
        from konte.index.bm25_store import _tokenize

        assert _tokenize("Revenue grew 12% in FY2022, driven by cloud.") == [
            "revenue",
            "grew",
            "12",
            "in",
            "fy2022",
            "driven",
            "by",
            "cloud",
        ]

    def test_codes_and_contractions_stay_whole(self):
        """Test punctuation between two alphanumerics is kept, so codes survive."""
        from konte.index.bm25_store import _tokenize

        assert _tokenize("Heading 8542.31 (circuits), $27.5 billion, don't") == [
            "heading",
            "8542.31",
            "circuits",
            "27.5",
            "billion",
            "don't",
        ]

    def test_korean_is_indexed_below_the_particle(self):
        """Test the noun's terms are a subset of the terms it carries a particle in."""
        from konte.index.bm25_store import _tokenize

        assert set(_tokenize("탈수기는")) == {"탈수", "수기", "기는"}
        assert set(_tokenize("탈수기")) < set(_tokenize("탈수기는"))

    def test_sentence_final_term_is_retrievable(self):
        """Test a query reaches the document that ended a sentence with the term."""
        store = self._store(
            [
                "PayPal reported positive working capital in FY2022.",
                "Adobe cloud segment grew during the period.",
                "Anti-dumping duties may apply to certain imports.",
            ]
        )

        found = store.query_with_coverage("working capital FY2022", top_k=1)

        assert found.results[0][0].chunk.chunk_id == "c0"
        assert found.coverage["c0"] == pytest.approx(1.0)

    def test_korean_query_reaches_an_inflected_document(self):
        """Test the noun matches the document that wrote it with a particle attached."""
        store = self._store(
            [
                "의류 탈수기는 제8450호에 분류됩니다.",
                "관세평가협정에 따라 과세가격을 결정합니다.",
                "반덤핑관세는 수입물품에 부과되는 특별관세입니다.",
            ]
        )

        found = store.query_with_coverage("의류 탈수기", top_k=1)

        assert found.results[0][0].chunk.chunk_id == "c0"
        assert found.coverage["c0"] == pytest.approx(1.0)

    def test_scoring_matches_the_reference_implementation(self, sample_chunks):
        """Test skipping unindexed terms and scaling repeats leaves scores unchanged."""
        import numpy as np
        from rank_bm25 import BM25Okapi

        from konte.index import BM25Store
        from konte.index.bm25_store import _score, _tokenize

        store = BM25Store()
        store.build_index(sample_chunks)
        reference = BM25Okapi([_tokenize(c.contextualized_content) for c in sample_chunks])

        for query in ("duty duty rate", "zirconium", "customs valuation 8542.31", ""):
            tokens = _tokenize(query)
            assert np.allclose(
                _score(store._model, tokens, store._length_norm),
                reference.get_scores(tokens),
            )


@pytest.mark.unit
class TestBM25StoreCoverage:
    """Test the absolute reading that survives comparison across queries."""

    def test_ranking_score_saturates_where_coverage_does_not(self, sample_chunks):
        """Test the top chunk always ranks 1.0 while coverage tracks the match."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        matched = store.query_with_coverage("tariff code 8542.31", top_k=3)
        weak = store.query_with_coverage("the", top_k=3)

        assert matched.results[0][1] == 1.0
        assert weak.results[0][1] == 1.0  # normalization cannot tell these apart
        assert matched.coverage[matched.results[0][0].chunk.chunk_id] == pytest.approx(1.0)

    def test_absent_query_term_lowers_coverage(self, sample_chunks):
        """Test naming something the corpus never indexed cannot read as a full match."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        present = store.query_with_coverage("8542.31", top_k=1)
        half = store.query_with_coverage("8542.31 zirconium", top_k=1)

        assert present.coverage[present.results[0][0].chunk.chunk_id] == pytest.approx(1.0)
        assert half.coverage[half.results[0][0].chunk.chunk_id] < 0.6

    def test_unmatched_query_covers_nothing(self, sample_chunks):
        """Test a query the corpus cannot answer reads 0.0, not 1.0."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        found = store.query_with_coverage("xyz123 completely unrelated gibberish", top_k=3)

        assert len(found.results) == 3
        assert all(value == 0.0 for value in found.coverage.values())

    def test_empty_query_covers_nothing(self, sample_chunks):
        """Test an empty query is not treated as fully matched."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        found = store.query_with_coverage("", top_k=2)

        assert all(value == 0.0 for value in found.coverage.values())

    def test_corpus_too_small_for_idf_counts_terms(self):
        """Test a corpus whose IDF is negative throughout still separates matches."""
        from konte.index import BM25Store

        chunk = Chunk(
            chunk_id="only",
            content="Customs valuation follows WTO Agreement rules.",
            source="test.txt",
            segment_idx=0,
            chunk_idx=0,
        )
        store = BM25Store()
        store.build_index([ContextualizedChunk(chunk=chunk, context="")])

        assert store.query_with_coverage("customs valuation").coverage["only"] == pytest.approx(1.0)
        assert store.query_with_coverage("customs zirconium").coverage["only"] == pytest.approx(0.5)
        assert store.query_with_coverage("zirconium").coverage["only"] == 0.0

    def test_coverage_survives_a_reload(self, sample_chunks, tmp_path):
        """Test the reading is the same after the index round-trips through disk."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)
        store.save(tmp_path)

        reloaded = BM25Store()
        reloaded.load(tmp_path, lambda: sample_chunks)

        for query in ("tariff code 8542.31", "8542.31 zirconium", "xyz123"):
            assert reloaded.query_with_coverage(query, top_k=3).coverage == pytest.approx(
                store.query_with_coverage(query, top_k=3).coverage
            )

    def test_coverage_is_reported_for_filtered_candidates(self, sample_chunks):
        """Test filtering does not misalign coverage with the chunks it describes."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)

        found = store.query_with_coverage("duty rate", top_k=2, source_filter="test.txt")

        assert set(found.coverage) == {chunk.chunk.chunk_id for chunk, _ in found.results}
        assert found.coverage[found.results[0][0].chunk.chunk_id] > 0.0


@pytest.mark.unit
class TestBM25StorePersistence:
    """Test BM25 store save/load functionality."""

    def test_save_and_load(self, sample_chunks, tmp_path):
        """Test that saved index can be loaded and queried."""
        from konte.index import BM25Store

        # Build and save
        store1 = BM25Store()
        store1.build_index(sample_chunks)
        store1.save(tmp_path)

        # Load into new store
        store2 = BM25Store()
        store2.load(tmp_path, lambda: sample_chunks)

        assert not store2.is_empty

        # Query should work
        results = store2.query("import duty rate", top_k=3)
        assert len(results) == 3

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading from nonexistent path raises error."""
        from konte.index import BM25Store

        store = BM25Store()
        with pytest.raises(FileNotFoundError):
            store.load(tmp_path / "nonexistent", list)

    def test_load_rejects_an_index_from_another_tokenizer(self, sample_chunks, tmp_path):
        """Test a stale index is refused rather than silently matching nothing."""
        import numpy as np

        from konte.index import BM25Store
        from konte.index.bm25_store import (
            _TOKENIZER_VERSION,
            INDEX_FILENAME,
            SIGNED_FILENAMES,
        )
        from konte.persistence.integrity import sign

        store = BM25Store()
        store.build_index(sample_chunks)
        store.save(tmp_path)

        with np.load(tmp_path / INDEX_FILENAME) as data:
            arrays = dict(data.items())
        arrays["tokenizer"] = np.array([_TOKENIZER_VERSION - 1])
        np.savez(tmp_path / INDEX_FILENAME, **arrays)
        sign(tmp_path, SIGNED_FILENAMES)  # the tokenizer check, not the signature

        with pytest.raises(ValueError, match="tokenizer"):
            BM25Store().load(tmp_path, lambda: sample_chunks)

    def test_save_writes_no_chunk_payload(self, sample_chunks, tmp_path):
        """Test that the lexical index stores no second copy of the corpus."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)
        store.save(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["bm25.npz", "bm25.npz.sig"]

    def test_save_stores_no_pickle(self, sample_chunks, tmp_path):
        """Test that nothing the index writes is read back through pickle."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(sample_chunks)
        store.save(tmp_path)

        assert not list(tmp_path.glob("*.pkl"))

    def test_save_removes_a_legacy_chunk_payload(self, sample_chunks, tmp_path):
        """Test that rebuilding drops the copy an earlier version left behind."""
        from konte.index import BM25Store
        from konte.index.bm25_store import LEGACY_CHUNKS_FILENAME

        (tmp_path / LEGACY_CHUNKS_FILENAME).write_text("[]", encoding="utf-8")

        store = BM25Store()
        store.build_index(sample_chunks)
        store.save(tmp_path)

        assert not (tmp_path / LEGACY_CHUNKS_FILENAME).exists()

    def test_load_defers_reading_the_corpus(self, sample_chunks, tmp_path):
        """Test that loading an index does not pull its chunks in with it."""
        from konte.index import BM25Store

        store1 = BM25Store()
        store1.build_index(sample_chunks)
        store1.save(tmp_path)

        reads = []

        def corpus():
            reads.append(1)
            return sample_chunks

        store2 = BM25Store()
        store2.load(tmp_path, corpus)
        assert reads == []

        store2.query("import duty rate", top_k=1)
        store2.query("import duty rate", top_k=1)
        assert reads == [1]

    def test_saved_chunks_preserved(self, sample_chunks, tmp_path):
        """Test that chunk data is preserved after save/load."""
        from konte.index import BM25Store

        store1 = BM25Store()
        store1.build_index(sample_chunks)
        store1.save(tmp_path)

        store2 = BM25Store()
        store2.load(tmp_path, lambda: sample_chunks)

        results = store2.query("test", top_k=1)
        chunk = results[0][0]

        # Verify chunk structure preserved
        assert chunk.chunk.chunk_id is not None
        assert chunk.chunk.source == "test.txt"

    def test_query_results_same_after_reload(self, sample_chunks, tmp_path):
        """Test that query results are consistent before and after reload."""
        from konte.index import BM25Store

        store1 = BM25Store()
        store1.build_index(sample_chunks)

        # Query before save
        results1 = store1.query("electronic integrated circuits", top_k=3)

        store1.save(tmp_path)

        store2 = BM25Store()
        store2.load(tmp_path, lambda: sample_chunks)

        # Query after load
        results2 = store2.query("electronic integrated circuits", top_k=3)

        # Same chunks should be returned
        ids1 = [r[0].chunk.chunk_id for r in results1]
        ids2 = [r[0].chunk.chunk_id for r in results2]
        assert ids1 == ids2


@pytest.fixture
def multi_source_chunks():
    """Create chunks from multiple sources for source_filter testing."""
    chunks = []
    sources = [
        ("ADOBE_2022_10K.md", "Adobe revenue grew 15% in FY2022."),
        ("ADOBE_2022_10K.md", "Adobe cloud segment leads growth."),
        ("JOHNSON_JOHNSON_2022_10K.md", "Johnson & Johnson pharmaceutical division."),
        ("JOHNSON_JOHNSON_2022_10K.md", "Johnson & Johnson medical devices revenue."),
        ("3M_2018_10K.md", "3M industrial segment performance."),
    ]
    for i, (source, content) in enumerate(sources):
        chunk = Chunk(
            chunk_id=f"{source}_s0_c{i}",
            content=content,
            source=source,
            segment_idx=0,
            chunk_idx=i,
        )
        chunks.append(ContextualizedChunk(chunk=chunk, context=""))
    return chunks


@pytest.fixture
def metadata_chunks():
    """Create chunks with custom metadata for metadata_filter testing."""
    data = [
        ("ADOBE_2022_10K.md", "Adobe revenue report.", {"company": "ADOBE", "year": "2022"}),
        ("ADOBE_2023_10K.md", "Adobe growth analysis.", {"company": "ADOBE", "year": "2023"}),
        ("3M_2022_10K.md", "3M industrial results.", {"company": "3M", "year": "2022"}),
        ("3M_2023_10K.md", "3M segment performance.", {"company": "3M", "year": "2023"}),
        ("JOHNSON_2022_10K.md", "Johnson pharma division.", {"company": "JOHNSON", "year": "2022"}),
    ]
    chunks = []
    for i, (source, content, meta) in enumerate(data):
        chunk = Chunk(
            chunk_id=f"{source}_s0_c{i}",
            content=content,
            source=source,
            segment_idx=0,
            chunk_idx=i,
            metadata=meta,
        )
        chunks.append(ContextualizedChunk(chunk=chunk, context=""))
    return chunks


@pytest.mark.unit
class TestBM25StoreSourceFilter:
    """Test BM25 store source_filter functionality."""

    def test_source_filter_substring_match(self, multi_source_chunks):
        """Test that source_filter returns only matching sources."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(multi_source_chunks)

        results = store.query("revenue", top_k=10, source_filter="ADOBE")

        assert len(results) > 0
        for chunk, _ in results:
            assert "ADOBE" in chunk.chunk.source

    def test_source_filter_multi_word(self, multi_source_chunks):
        """Test source_filter with multi-word company name."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(multi_source_chunks)

        results = store.query("revenue", top_k=10, source_filter="JOHNSON_JOHNSON")

        assert len(results) > 0
        for chunk, _ in results:
            assert "JOHNSON_JOHNSON" in chunk.chunk.source

    def test_source_filter_no_match(self, multi_source_chunks):
        """Test source_filter with no matching source."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(multi_source_chunks)

        results = store.query("revenue", top_k=10, source_filter="NONEXISTENT")

        assert results == []

    def test_source_filter_combined_with_metadata_filter(self, multi_source_chunks):
        """Test source_filter works alongside metadata_filter."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(multi_source_chunks)

        results = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"source": "ADOBE_2022_10K.md"},
            source_filter="ADOBE",
        )

        assert len(results) > 0
        for chunk, _ in results:
            assert chunk.chunk.source == "ADOBE_2022_10K.md"


@pytest.fixture
def filter_edge_chunks():
    """Chunks covering every field shape a posted filter has to answer for."""
    rows = [
        ("ADOBE_2022_10K.md", 0, 0, {"company": "ADOBE", "year": "2022", "pages": 10}),
        ("ADOBE_2022_10K.md", 0, 1, {"company": "ADOBE", "year": None}),
        ("ADOBE_2023_10K.md", 1, 0, {"company": "ADOBE"}),  # carries no year at all
        ("3M_2022_10K.md", 0, 0, {"company": "3M", "year": "2022"}),
        ("3M_2023_10K.md", 1, 2, {"company": "3M", "year": 2023}),  # a number, not a string
        ("3M_2023_10K.md", 1, 3, {"company": "3M", "tags": ["duty", "quota"]}),
        ("3M_2023_10K.md", 1, 4, {"company": "3M", "tags": "duty"}),
        ("NOTES.md", 0, 0, {"source": "SHADOW.md", "segment_idx": 99}),
        ("NOTES.md", 0, 1, {}),
    ]
    return [
        ContextualizedChunk(
            chunk=Chunk(
                chunk_id=f"{source}_s{segment_idx}_c{chunk_idx}",
                content=f"Duty rate and tariff notes {i}.",
                source=source,
                segment_idx=segment_idx,
                chunk_idx=chunk_idx,
                metadata=metadata,
            ),
            context="",
        )
        for i, (source, segment_idx, chunk_idx, metadata) in enumerate(rows)
    ]


FILTER_CASES = [
    ({"company": "ADOBE"}, None),
    ({"company": ["ADOBE", "3M"]}, None),
    ({"company": None}, None),
    ({"year": None}, None),
    ({"year": 2023}, None),
    ({"year": "2023"}, None),
    ({"year": ["2022", None]}, None),
    ({"pages": 10}, None),
    ({"unposted": "value"}, None),
    ({"unposted": None}, None),
    ({"company": "ADOBE", "year": "2022"}, None),
    ({"segment_idx": 0}, None),
    ({"segment_idx": 99}, None),
    ({"chunk_idx": 4}, None),
    ({"source": "NOTES.md"}, None),
    ({"source": "SHADOW.md"}, None),
    ({"tags": "duty"}, None),
    ({"tags": ["duty", "quota"]}, None),
    ({"company": "3M", "tags": "duty"}, None),
    (None, "ADOBE"),
    (None, "10K"),
    (None, "SHADOW"),
    (None, "MISSING"),
    ({"company": "3M"}, "2023"),
    ({}, "NOTES"),
]


@pytest.mark.unit
class TestBM25StoreFilterIndex:
    """Test that posted filters select exactly what walking the corpus selects."""

    def test_posted_filters_select_what_a_scan_selects(self, filter_edge_chunks):
        """Test every filter shape resolves to the same positions, in the same order."""
        from konte.index.bm25_store import _filter_entries, _filter_indices
        from konte.index.filter_index import FilterIndex

        index = FilterIndex(_filter_entries(filter_edge_chunks))

        for metadata_filter, source_filter in FILTER_CASES:
            scanned = _filter_indices(filter_edge_chunks, metadata_filter, source_filter)
            selected = index.select(metadata_filter, source_filter)
            if selected is None:  # unpostable field; the store scans instead
                continue
            assert list(selected) == list(scanned), (metadata_filter, source_filter)

    def test_a_query_returns_what_a_scan_would_have(self, filter_edge_chunks):
        """Test the store reaches the same chunks through the index it reached before."""
        from konte.index import BM25Store
        from konte.index.bm25_store import _filter_indices

        store = BM25Store()
        store.build_index(filter_edge_chunks)

        for metadata_filter, source_filter in FILTER_CASES:
            expected = {
                filter_edge_chunks[position].chunk.chunk_id
                for position in _filter_indices(filter_edge_chunks, metadata_filter, source_filter)
            }
            found = store.query(
                "duty rate tariff",
                top_k=len(filter_edge_chunks),
                metadata_filter=metadata_filter,
                source_filter=source_filter,
            )
            assert {chunk.chunk.chunk_id for chunk, _ in found} == expected, (
                metadata_filter,
                source_filter,
            )

    def test_an_unpostable_value_is_scanned_rather_than_dropped(self, filter_edge_chunks):
        """Test a field one chunk holds a list in still filters the rest of the corpus."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(filter_edge_chunks)

        found = store.query("duty rate tariff", top_k=10, metadata_filter={"tags": "duty"})

        assert [chunk.chunk.chunk_id for chunk, _ in found] == ["3M_2023_10K.md_s1_c4"]

    def test_metadata_cannot_shadow_the_chunk_it_belongs_to(self, filter_edge_chunks):
        """Test a source filter reads the chunk's own source, not metadata named alike."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(filter_edge_chunks)

        assert store.query("duty", top_k=10, source_filter="SHADOW") == []
        assert len(store.query("duty", top_k=10, metadata_filter={"source": "NOTES.md"})) == 2

    def test_a_rebuild_rereads_the_corpus(self, filter_edge_chunks, sample_chunks):
        """Test the positions posted for one corpus never answer for the next."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(filter_edge_chunks)
        store.query("duty", top_k=10, metadata_filter={"company": "ADOBE"})

        store.build_index(sample_chunks)
        found = store.query("tariff duty", top_k=10, source_filter="test.txt")

        assert len(found) == len(sample_chunks)


@pytest.mark.unit
class TestBM25StoreListValueFilter:
    """Test BM25 store metadata_filter with list values."""

    def test_metadata_filter_list_values(self, metadata_chunks):
        """Test filtering with list values returns chunks matching any value."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(metadata_chunks)

        results = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"company": ["ADOBE", "3M"]},
        )

        assert len(results) == 4
        companies = {r[0].chunk.metadata["company"] for r in results}
        assert companies == {"ADOBE", "3M"}

    def test_metadata_filter_list_single_item(self, metadata_chunks):
        """Test list with single item behaves same as scalar."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(metadata_chunks)

        results_list = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"year": ["2022"]},
        )
        results_scalar = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"year": "2022"},
        )

        ids_list = sorted(r[0].chunk.chunk_id for r in results_list)
        ids_scalar = sorted(r[0].chunk.chunk_id for r in results_scalar)
        assert ids_list == ids_scalar

    def test_metadata_filter_mixed_list_and_scalar(self, metadata_chunks):
        """Test combining list and scalar filters (AND logic)."""
        from konte.index import BM25Store

        store = BM25Store()
        store.build_index(metadata_chunks)

        results = store.query(
            "revenue",
            top_k=10,
            metadata_filter={"company": ["ADOBE", "3M"], "year": "2022"},
        )

        assert len(results) == 2
        for chunk, _ in results:
            assert chunk.chunk.metadata["company"] in ["ADOBE", "3M"]
            assert chunk.chunk.metadata["year"] == "2022"
