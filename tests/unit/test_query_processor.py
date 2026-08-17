"""Unit tests for query preprocessing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from konte.retrieval.query_processor import (
    ExtractedKeywords,
    clear_keyword_cache,
    extract_search_keywords,
    extract_search_keywords_async,
)


@pytest.fixture(autouse=True)
def _empty_cache():
    """Keep the module-level extraction cache from leaking between tests."""
    clear_keyword_cache()
    yield
    clear_keyword_cache()


@pytest.fixture
def structured_llm():
    """Patch the extraction client and yield the structured-output stub."""
    with patch("konte.retrieval.query_processor.get_llm") as get_llm:
        stub = MagicMock()
        stub.invoke.return_value = ExtractedKeywords(keywords=["HS 코드", "분류"])
        stub.ainvoke = AsyncMock(return_value=ExtractedKeywords(keywords=["HS 코드", "분류"]))
        get_llm.return_value.with_structured_output.return_value = stub
        stub.get_llm = get_llm
        yield stub


@pytest.mark.unit
class TestExtractionRequest:
    """Test how the extraction client is configured."""

    def test_runs_on_a_short_budget(self, structured_llm):
        """Extraction must not inherit the batch-sized context timeout."""
        from konte.runtime import settings

        extract_search_keywords("의류 탈수기는 어느 HS 코드에 분류되나요?")

        kwargs = structured_llm.get_llm.call_args.kwargs
        assert kwargs["timeout"] == settings.KEYWORD_EXTRACTION_TIMEOUT
        assert kwargs["timeout"] <= 5.0, "a query blocks on this call"

    def test_does_not_retry(self, structured_llm):
        """Retries multiply the delay a dead endpoint imposes on every query."""
        extract_search_keywords("test query")

        assert structured_llm.get_llm.call_args.kwargs["max_retries"] == 0


@pytest.mark.unit
class TestKeywordCache:
    """Test that repeated queries do not repeat the round trip."""

    def test_repeated_query_hits_the_cache(self, structured_llm):
        """Test the same query is extracted once, however often it is asked."""
        for _ in range(3):
            assert extract_search_keywords("same query") == ["HS 코드", "분류"]

        assert structured_llm.invoke.call_count == 1

    def test_distinct_queries_are_extracted_separately(self, structured_llm):
        """Test the cache keys on the query text."""
        extract_search_keywords("first query")
        extract_search_keywords("second query")

        assert structured_llm.invoke.call_count == 2

    async def test_async_shares_the_sync_cache(self, structured_llm):
        """Test either path can satisfy the other; the LLM is called once."""
        extract_search_keywords("shared query")
        assert await extract_search_keywords_async("shared query") == ["HS 코드", "분류"]

        assert structured_llm.invoke.call_count == 1
        structured_llm.ainvoke.assert_not_called()

    async def test_sync_reuses_an_async_extraction(self, structured_llm):
        """Test the sharing holds in the other direction too."""
        await extract_search_keywords_async("shared query")
        assert extract_search_keywords("shared query") == ["HS 코드", "분류"]

        assert structured_llm.ainvoke.call_count == 1
        structured_llm.invoke.assert_not_called()

    def test_clear_forces_re_extraction(self, structured_llm):
        """Test clear_keyword_cache drops entries a new prompt would invalidate."""
        extract_search_keywords("same query")
        clear_keyword_cache()
        extract_search_keywords("same query")

        assert structured_llm.invoke.call_count == 2

    def test_caller_cannot_corrupt_the_cache(self, structured_llm):
        """Test each caller gets its own list, not the cached tuple's contents."""
        first = extract_search_keywords("same query")
        first.append("injected")

        assert extract_search_keywords("same query") == ["HS 코드", "분류"]


@pytest.mark.unit
class TestExtractionFailure:
    """Test the fallback path around a failing endpoint."""

    def test_falls_back_to_tokenization(self, structured_llm):
        """Test a failed call degrades to stopword filtering."""
        structured_llm.invoke.side_effect = RuntimeError("endpoint down")

        keywords = extract_search_keywords("What is the working capital of Paypal?")

        assert "the" not in keywords
        assert "working" in keywords

    async def test_async_falls_back_to_tokenization(self, structured_llm):
        """Test the async path degrades the same way."""
        structured_llm.ainvoke.side_effect = RuntimeError("endpoint down")

        assert await extract_search_keywords_async("What is the tariff") == ["tariff"]

    def test_failure_is_not_cached(self, structured_llm):
        """Test a recovered endpoint is used again rather than the fallback."""
        structured_llm.invoke.side_effect = [
            RuntimeError("endpoint down"),
            ExtractedKeywords(keywords=["HS 코드"]),
        ]

        assert extract_search_keywords("HS 코드는?") == ["HS", "코드는?"]
        assert extract_search_keywords("HS 코드는?") == ["HS 코드"]
