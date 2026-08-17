"""Unit tests for the chat client factory."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from konte.runtime import llm


@pytest.fixture(autouse=True)
def _empty_client_cache():
    """Keep cached clients from leaking between tests."""
    llm._client_cache.clear()
    yield
    llm._client_cache.clear()


@pytest.fixture
def chat_openai():
    """Patch the client class so no credentials or network are involved."""
    with patch("konte.runtime.llm.ChatOpenAI") as cls:
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


@pytest.mark.unit
class TestResponseText:
    """Test that an answer is read whatever shape the provider returns it in."""

    def test_plain_string_is_stripped(self):
        """Test the ordinary case is untouched."""
        assert llm.response_text(AIMessage(content="  hello  ")) == "hello"

    def test_content_blocks_are_read(self):
        """Test a block-shaped answer is not discarded as unreadable."""
        response = AIMessage(
            content=[
                {"type": "text", "text": "This passage is from"},
                {"type": "text", "text": " Samsung's Q3 report."},
            ]
        )

        assert llm.response_text(response) == "This passage is from Samsung's Q3 report."

    def test_string_blocks_are_read(self):
        """Test the list-of-strings shape is joined too."""
        assert llm.response_text(AIMessage(content=["one ", "two"])) == "one two"

    def test_non_text_blocks_are_dropped(self):
        """Test a reasoning trace is not indexed as if it were the answer."""
        response = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "the user wants context"},
                {"type": "text", "text": "The chunk sits in the memory section."},
            ]
        )

        assert llm.response_text(response) == "The chunk sits in the memory section."

    def test_blocks_carrying_no_text_read_empty(self):
        """Test a response with nothing to index still reports nothing."""
        assert llm.response_text(AIMessage(content=[{"type": "tool_use", "id": "1"}])) == ""


@pytest.mark.unit
class TestWasTruncated:
    """Test detection of a response cut off at the token ceiling."""

    def test_length_finish_reason_is_truncation(self):
        """Test the ceiling is reported."""
        response = AIMessage(content="half a sen", response_metadata={"finish_reason": "length"})

        assert llm.was_truncated(response) is True

    def test_normal_stop_is_not_truncation(self):
        """Test a complete response is not flagged."""
        response = AIMessage(content="whole", response_metadata={"finish_reason": "stop"})

        assert llm.was_truncated(response) is False

    def test_absent_metadata_is_not_truncation(self):
        """Test an endpoint that reports no reason is taken at face value."""
        assert llm.was_truncated(AIMessage(content="whole")) is False
