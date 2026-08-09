"""Unit tests for the chat client factory."""

from unittest.mock import patch

import pytest

from konte import llm


@pytest.fixture(autouse=True)
def _empty_client_cache():
    """Keep cached clients from leaking between tests."""
    llm._client_cache.clear()
    yield
    llm._client_cache.clear()


@pytest.fixture
def chat_openai():
    """Patch the client class so no credentials or network are involved."""
    with patch("konte.llm.ChatOpenAI") as cls:
        cls.side_effect = lambda **kwargs: kwargs
        yield cls


@pytest.mark.unit
class TestGetLLM:
    """Test client construction and caching."""

    def test_defaults_to_the_batch_retry_budget(self, chat_openai):
        """Test callers that say nothing keep the previous behavior."""
        client = llm.get_llm()

        assert client["max_retries"] == llm.CLIENT_MAX_RETRIES
        assert client["timeout"] == 120.0

    def test_retry_budget_is_forwarded(self, chat_openai):
        """Test a latency-bound caller can make the request single-shot."""
        assert llm.get_llm(max_retries=0)["max_retries"] == 0

    def test_identical_arguments_reuse_one_client(self, chat_openai):
        """Test the cache still works for the common case."""
        assert llm.get_llm(timeout=5.0) is llm.get_llm(timeout=5.0)
        assert chat_openai.call_count == 1

    def test_retry_budget_keys_the_cache(self, chat_openai):
        """Test a cached client cannot hand its retry budget to the next caller.

        Extraction and context generation share this factory, so without the
        key whichever ran first decides how long the other blocks for.
        """
        patient = llm.get_llm(timeout=120.0, max_retries=2)
        impatient = llm.get_llm(timeout=120.0, max_retries=0)

        assert patient["max_retries"] == 2
        assert impatient["max_retries"] == 0
