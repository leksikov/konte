"""Unit tests for context generation accounting (no network)."""

import re
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from openai import APITimeoutError

from konte.contextualize.generator import generate_contexts_batch
from konte.domain import Chunk

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
    return patch("konte.contextualize.generator.get_llm", return_value=client)


def _answers(reply):
    """Patch the chat client so `reply` decides each prompt's response.

    Retries are cut to one attempt, the backoff between ten being real time.
    """
    client = AsyncMock()
    client.ainvoke.side_effect = lambda prompt: reply(prompt)
    return patch("konte.contextualize.generator.get_llm", return_value=client), patch(
        "konte.contextualize.generator.MAX_RETRIES", 1
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


BATCH_PROMPT = "{segment}\n\nEmit [[1]] to [[{count}]]."

SEGMENT = "Alpha opens the section. Beta follows it. Gamma closes it."


def _segment_chunks():
    """Three chunks that are verbatim slices of SEGMENT, in order."""
    return [
        Chunk(
            chunk_id=f"s0_c{i}",
            content=content,
            source="doc.md",
            segment_idx=0,
            chunk_idx=i,
        )
        for i, content in enumerate(
            ["Alpha opens the section.", "Beta follows it.", "Gamma closes it."]
        )
    ]


def _marked(*positions):
    """A batch reply carrying a context for each named marker position."""
    return AIMessage(content="\n\n".join(f"[[{p}]]\nctx {p}" for p in positions))


@pytest.mark.unit
class TestPerSegmentBatching:
    """Test that one request can carry a whole segment's contexts."""

    async def test_one_request_describes_every_chunk(self):
        """Test the segment is sent once, not once per chunk."""
        client = AsyncMock()
        client.ainvoke.side_effect = [_marked(1, 2, 3)]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        assert client.ainvoke.call_count == 1
        assert [c.context for c in batch.chunks] == ["ctx 1", "ctx 2", "ctx 3"]
        assert batch.failed == 0

    async def test_the_prompt_marks_where_each_chunk_begins(self):
        """Test the chunks are located in the segment instead of sent again."""
        client = AsyncMock()
        client.ainvoke.side_effect = [_marked(1, 2, 3)]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        prompt = client.ainvoke.call_args.args[0]
        assert "[[1]]\nAlpha opens the section." in prompt
        assert "[[2]]\nBeta follows it." in prompt
        assert "[[3]]\nGamma closes it." in prompt
        assert "Emit [[1]] to [[3]]." in prompt
        # One copy of the text, not one per chunk.
        assert prompt.count("Beta follows it.") == 1

    async def test_a_chunk_the_reply_skipped_is_asked_for_alone(self):
        """Test a short reply costs the missing chunk a retry, not the segment."""
        client = AsyncMock()
        client.ainvoke.side_effect = [_marked(1, 3), _marked(1)]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        assert client.ainvoke.call_count == 2
        assert [c.context for c in batch.chunks] == ["ctx 1", "ctx 1", "ctx 3"]
        assert batch.failed == 0

        retry = client.ainvoke.call_args_list[1].args[0]
        assert "[[1]]\nBeta follows it." in retry
        assert "Emit [[1]] to [[1]]." in retry

    async def test_an_unlabelled_reply_to_a_lone_chunk_is_taken_whole(self):
        """Test the retry does not need its marker echoed back to count."""
        client = AsyncMock()
        client.ainvoke.side_effect = [_marked(1, 2), AIMessage(content="ctx for gamma")]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        assert batch.chunks[2].context == "ctx for gamma"
        assert batch.failed == 0

    async def test_a_reply_in_no_known_format_leaves_every_chunk_to_retry(self):
        """Test a model that ignores the protocol costs one request, not the batch."""
        client = AsyncMock()
        client.ainvoke.side_effect = [
            AIMessage(content="Sure! Here are the contexts you asked for."),
            AIMessage(content="ctx a"),
            AIMessage(content="ctx b"),
            AIMessage(content="ctx c"),
        ]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        assert client.ainvoke.call_count == 4
        assert [c.context for c in batch.chunks] == ["ctx a", "ctx b", "ctx c"]
        assert batch.failed == 0

    async def test_a_truncated_reply_re_asks_the_context_it_cut(self):
        """Test the context the ceiling cut is regenerated, not indexed half-written."""
        cut_off = AIMessage(
            content="[[1]]\nctx 1\n\n[[2]]\nctx 2\n\n[[3]]\nGamma closes the sec",
            response_metadata={"finish_reason": "length"},
        )
        client = AsyncMock()
        client.ainvoke.side_effect = [cut_off, _marked(1)]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        assert client.ainvoke.call_count == 2
        assert [c.context for c in batch.chunks] == ["ctx 1", "ctx 2", "ctx 1"]
        assert batch.truncated == 0

    async def test_a_truncated_retry_is_kept_and_counted(self):
        """Test a lone context cut off is reported rather than dropped for good."""
        cut_off = AIMessage(
            content="[[1]]\nGamma closes the sec",
            response_metadata={"finish_reason": "length"},
        )
        client = AsyncMock()
        client.ainvoke.side_effect = [_marked(1, 2), cut_off]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        assert batch.chunks[2].context == "Gamma closes the sec"
        assert batch.truncated == 1
        assert batch.failed == 0

    async def test_a_retry_that_never_answers_is_counted_as_failed(self):
        """Test a chunk left without context still reaches the coverage check."""

        def reply(prompt):
            if "[[1]] to [[1]]" in prompt:
                raise APITimeoutError(request=None)
            return _marked(1, 2)

        client, retries = _answers(reply)
        with client, retries:
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks(), prompt_template=BATCH_PROMPT
            )

        assert batch.chunks[2].context == ""
        assert batch.failed == 1

    async def test_a_lone_chunk_is_not_worth_a_batch(self):
        """Test a one-chunk segment goes straight to the per-chunk protocol."""
        client = AsyncMock()
        client.ainvoke.side_effect = [AIMessage(content="ctx alone")]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment=SEGMENT, chunks=_segment_chunks()[:1], prompt_template=BATCH_PROMPT
            )

        assert client.ainvoke.call_count == 1
        assert batch.chunks[0].context == "ctx alone"
        assert "Emit [[1]]" not in client.ainvoke.call_args.args[0]

    async def test_chunks_missing_from_their_segment_fall_back(self):
        """Test a corpus the markers cannot be placed in still gets its contexts."""
        client = AsyncMock()
        client.ainvoke.side_effect = [AIMessage(content=f"ctx {i}") for i in range(3)]

        with patch("konte.contextualize.generator.get_llm", return_value=client):
            batch = await generate_contexts_batch(
                segment="a segment none of them came from",
                chunks=_segment_chunks(),
                prompt_template=BATCH_PROMPT,
            )

        assert client.ainvoke.call_count == 3
        assert [c.context for c in batch.chunks] == ["ctx 0", "ctx 1", "ctx 2"]
        assert batch.failed == 0


@pytest.mark.unit
class TestPromptDecidesProtocol:
    """Test that a prompt is never dropped in favour of the cheaper protocol."""

    def test_a_per_chunk_prompt_overrides_a_per_segment_project(self, tmp_path):
        """Test a project's own prompt outranks the configured strategy."""
        from konte.contextualize.generator import resolve_prompt

        path = tmp_path / "domain_prompt.txt"
        path.write_text(PROMPT, encoding="utf-8")

        prompt = resolve_prompt(path, "per_segment")

        assert prompt.strategy == "per_chunk"
        assert prompt.template == PROMPT

    def test_the_packaged_prompts_carry_the_placeholders_they_promise(self):
        """Test each strategy's default prompt is fillable by its own protocol."""
        from konte.contextualize.generator import resolve_prompt

        per_chunk = resolve_prompt(None, "per_chunk")
        per_segment = resolve_prompt(None, "per_segment")

        assert per_chunk.strategy == "per_chunk"
        assert {"{segment}", "{chunk}"} <= set(re.findall(r"\{\w+\}", per_chunk.template))
        assert per_segment.strategy == "per_segment"
        assert set(re.findall(r"\{\w+\}", per_segment.template)) == {"{segment}", "{count}"}

    async def test_a_per_chunk_template_still_sends_one_request_per_chunk(self):
        """Test the existing protocol is unchanged by the new default strategy."""
        with _replies(AIMessage(content="ctx a"), AIMessage(content="ctx b")):
            batch = await generate_contexts_batch(
                segment="segment", chunks=_chunks(2), prompt_template=PROMPT, strategy="per_segment"
            )

        assert [c.context for c in batch.chunks] == ["ctx a", "ctx b"]
