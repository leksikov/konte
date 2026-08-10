"""Unit tests for context generation accounting (no network)."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from openai import APITimeoutError

from konte.context import generate_contexts_batch
from konte.models import Chunk

PROMPT = "{segment} :: {chunk}"


def _chunks(count):
    """Build `count` chunks of one segment."""
    return [
        Chunk(
            chunk_id=f"s0_c{i}",
            content=f"content {i}",
            source="doc.md",
            segment_idx=0,
            chunk_idx=i,
        )
        for i in range(count)
    ]


def _replies(*responses):
    """Patch the chat client so each chunk gets the next scripted response."""
    client = AsyncMock()
    client.ainvoke.side_effect = list(responses)
    return patch("konte.context.get_llm", return_value=client)


def _answers(reply):
    """Patch the chat client so `reply` decides each prompt's response.

    Retries are cut to one attempt, the backoff between ten being real time.
    """
    client = AsyncMock()
    client.ainvoke.side_effect = lambda prompt: reply(prompt)
    return patch("konte.context.get_llm", return_value=client), patch(
        "konte.context.MAX_RETRIES", 1
    )


@pytest.mark.unit
class TestBatchAccounting:
    """Test that a batch reports what its requests actually produced."""

    async def test_every_context_generated_reports_nothing_lost(self):
        """Test the ordinary case counts no failures."""
        with _replies(AIMessage(content="ctx a"), AIMessage(content="ctx b")):
            batch = await generate_contexts_batch(
                segment="segment", chunks=_chunks(2), prompt_template=PROMPT
            )

        assert [c.context for c in batch.chunks] == ["ctx a", "ctx b"]
        assert batch.failed == 0
        assert batch.truncated == 0

    async def test_a_failed_request_is_counted_not_just_emptied(self):
        """Test a chunk left without context is countable, not just emptied."""

        def reply(prompt):
            if "content 0" in prompt:
                raise APITimeoutError(request=None)
            return AIMessage(content="ctx b")

        client, retries = _answers(reply)
        with client, retries:
            batch = await generate_contexts_batch(
                segment="segment",
                chunks=_chunks(2),
                prompt_template=PROMPT,
            )

        assert len(batch.chunks) == 2
        assert batch.chunks[0].context == ""
        assert batch.chunks[1].context == "ctx b"
        assert batch.failed == 1

    async def test_a_response_with_no_usable_text_counts_as_failed(self):
        """Test a success carrying nothing indexable is a failure, not a context."""
        with _replies(AIMessage(content=[{"type": "reasoning", "reasoning": "..."}])):
            batch = await generate_contexts_batch(
                segment="segment", chunks=_chunks(1), prompt_template=PROMPT
            )

        assert batch.chunks[0].context == ""
        assert batch.failed == 1

    async def test_a_truncated_context_is_kept_and_counted(self):
        """Test a context cut off at the ceiling is indexed but reported."""
        cut_off = AIMessage(
            content="This passage is from the Semiconductor Divi",
            response_metadata={"finish_reason": "length"},
        )

        with _replies(cut_off):
            batch = await generate_contexts_batch(
                segment="segment", chunks=_chunks(1), prompt_template=PROMPT
            )

        assert batch.chunks[0].context.startswith("This passage")
        assert batch.truncated == 1
        assert batch.failed == 0

    async def test_skip_context_reports_no_failures(self):
        """Test standard-RAG mode does not read as a corpus that lost every context."""
        batch = await generate_contexts_batch(
            segment="segment", chunks=_chunks(3), skip_context=True
        )

        assert all(c.context == "" for c in batch.chunks)
        assert batch.failed == 0
