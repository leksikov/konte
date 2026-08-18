"""Context generation module using LLM to generate chunk context."""

from __future__ import annotations

import asyncio
import importlib.resources
import re
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from weakref import WeakKeyDictionary

import structlog

from konte.domain.config import ContextStrategy
from konte.domain.models import Chunk, ContextualizedChunk, encode_segment_key
from konte.runtime.llm import get_llm, response_text, was_truncated
from konte.runtime.settings import settings

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_openai import ChatOpenAI

__all__ = [
    "ContextBatch",
    "ContextPrompt",
    "call_limiter",
    "generate_context",
    "generate_contexts_batch",
    "get_llm",
    "load_prompt_template",
    "resolve_prompt",
]

logger = structlog.get_logger()

MAX_RETRIES = 10
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 120.0  # seconds

# One chunk costs one request either way.
BATCH_MIN_CHUNKS = 2

# 400 is what one chunk is allowed on its own; see get_llm. Bucketing keeps
# segments of differing chunk counts sharing a few cached clients.
BATCH_TOKENS_PER_CHUNK = 400
BATCH_TOKENS_OVERHEAD = 256
BATCH_TOKEN_BUCKET = 512


@cache
def _retryable_api_errors() -> tuple[type[Exception], ...]:
    """The API failures worth another attempt.

    Read on first use: only a request that reaches the endpoint needs the
    client stack behind these.
    """
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    return (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError)


_PACKAGED_PROMPTS: dict[ContextStrategy, str] = {
    "per_chunk": "context_prompt.txt",
    "per_segment": "context_prompt_segment.txt",
}

_CHUNK_PLACEHOLDER = "{chunk}"

# Not anchored to the line start: a marker is not reliably alone on its line.
_RESPONSE_MARKER = re.compile(r"\[\[\s*(\d+)\s*\]\]")

# A semaphore binds to the loop that first awaits it, and a process may run several.
_limiters: WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = (
    WeakKeyDictionary()
)


def call_limiter() -> asyncio.Semaphore:
    """Return the shared ceiling on context requests, settings.MAX_CONCURRENT_CALLS.

    Returns:
        The semaphore every context request on the running loop passes through.
    """
    loop = asyncio.get_running_loop()
    limit = settings.concurrency_limit
    cached = _limiters.get(loop)
    if cached is None or cached[0] != limit:
        cached = (limit, asyncio.Semaphore(limit))
        _limiters[loop] = cached
    return cached[1]


@cache
def _packaged_prompt(strategy: ContextStrategy) -> str:
    """Read the prompt this distribution ships for one strategy."""
    # Read via importlib.resources so zip/frozen installs work.
    ref = importlib.resources.files("konte.contextualize") / "prompts" / _PACKAGED_PROMPTS[strategy]
    return ref.read_text(encoding="utf-8")


def load_prompt_template(
    prompt_path: Path | None = None,
    strategy: ContextStrategy = "per_chunk",
) -> str:
    """Load the context generation prompt template.

    Args:
        prompt_path: Path to prompt file. Defaults to settings.PROMPT_PATH,
            then falls back to the packaged default prompt. A configured path
            is used whatever `strategy` asks for; see resolve_prompt.
        strategy: Which packaged default to fall back to. "per_chunk" carries
            {segment} and {chunk}; "per_segment" carries {segment} and {count}.

    Returns:
        Prompt template string with the placeholders that strategy implies.

    Raises:
        FileNotFoundError: If a configured prompt path does not exist.
    """
    path = prompt_path or settings.PROMPT_PATH
    if path is not None:
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Context prompt not found: {path}. The packaged defaults live "
                "at konte/contextualize/prompts/ and domain examples at "
                "examples/prompts/; update PROMPT_PATH or the project's "
                "context_prompt_path in config.json."
            ) from e
    return _packaged_prompt(strategy)


class ContextPrompt(NamedTuple):
    """A prompt template, and the protocol its placeholders commit it to."""

    strategy: ContextStrategy
    template: str


def _protocol_of(template: str) -> ContextStrategy:
    """Which protocol a template is written for.

    Only a per-chunk template can name the chunk it describes, so {chunk} is
    what separates the two.
    """
    return "per_chunk" if _CHUNK_PLACEHOLDER in template else "per_segment"


def resolve_prompt(prompt_path: Path | None, strategy: ContextStrategy) -> ContextPrompt:
    """Pick the prompt a build runs on, and the protocol that follows from it.

    A configured prompt outranks the strategy: dropping a project's own prompt
    to save requests would change what the corpus is indexed as.

    Args:
        prompt_path: Prompt override, or None for settings.PROMPT_PATH and then
            the packaged default.
        strategy: The configured strategy, used when nothing overrides it.

    Returns:
        The template and the strategy actually in force.

    Raises:
        FileNotFoundError: If a configured prompt path does not exist.
    """
    path = prompt_path or settings.PROMPT_PATH
    if path is None:
        return ContextPrompt(strategy, _packaged_prompt(strategy))

    template = load_prompt_template(path)
    protocol = _protocol_of(template)
    if protocol != strategy:
        logger.warning(
            "context_strategy_set_by_prompt",
            prompt_path=str(path),
            configured=strategy,
            used=protocol,
        )
    return ContextPrompt(protocol, template)


def _format_prompt(template: str, segment: str, chunk: str) -> str:
    """Format the prompt template with segment and chunk content.

    Args:
        template: Prompt template with {segment} and {chunk} placeholders.
        segment: Segment text to include.
        chunk: Chunk text to include.

    Returns:
        Formatted prompt string.
    """
    return template.format(segment=segment, chunk=chunk)


class _Outcome(NamedTuple):
    """One chunk's context, flagged when the request came up short.

    A failure cannot be inferred from the empty context it leaves behind: a
    skip_context run produces the same thing deliberately.
    """

    context: str
    failed: bool
    truncated: bool


class ContextBatch(NamedTuple):
    """A segment's contextualized chunks, and how many of them came up short."""

    chunks: list[ContextualizedChunk]
    failed: int
    truncated: int


_FAILED = _Outcome("", True, False)


def _without_context(chunks: list[Chunk]) -> ContextBatch:
    """Wrap chunks with an empty context, as skip_context asks for.

    Nothing was requested, so nothing counts as failed.
    """
    return ContextBatch([ContextualizedChunk(chunk=chunk, context="") for chunk in chunks], 0, 0)


async def _send(
    llm: ChatOpenAI,
    build_prompt: Callable[[], str],
    limiter: asyncio.Semaphore,
    max_attempts: int,
    log_context: dict[str, Any],
) -> BaseMessage | None:
    """Send one prompt, retrying that request and no other.

    The prompt carries a whole segment, so build_prompt runs under the limiter:
    only the requests in flight hold a copy. Returns None once the attempts run
    out or a non-retryable error ends them; log_context names the request in
    this module's log lines.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            async with limiter:
                return await llm.ainvoke(build_prompt())
        except _retryable_api_errors() as e:
            error_type = type(e).__name__
            status_code = getattr(e, "status_code", None)
            if attempt == max_attempts:
                logger.error(
                    "api_retries_exhausted",
                    **log_context,
                    error_type=error_type,
                    status_code=status_code,
                    error=str(e),
                )
                return None
            delay = min(BASE_DELAY * 2 ** (attempt - 1), MAX_DELAY)
            logger.warning(
                "api_error_retrying",
                **log_context,
                attempt=attempt,
                max_retries=max_attempts,
                delay=delay,
                error_type=error_type,
                status_code=status_code,
                error=str(e),
            )
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error(
                "context_generation_failed",
                **log_context,
                error_type=type(e).__name__,
                error=str(e),
            )
            return None

    return None  # Unreachable: the last attempt either returns or gives up above.


async def _request_context(
    llm: ChatOpenAI,
    template: str,
    segment: str,
    chunk: Chunk,
    limiter: asyncio.Semaphore,
    max_attempts: int,
) -> _Outcome:
    """Ask for one chunk's context, retrying that chunk and no other.

    Args:
        llm: Client to send the request through.
        template: Prompt template with {segment} and {chunk} placeholders.
        segment: The parent segment text (~8000 tokens).
        chunk: The chunk to describe.
        limiter: The in-flight ceiling to acquire for each attempt.
        max_attempts: How many times to send the request before giving up.

    Returns:
        The generated context, marked failed once the attempts run out.
    """
    response = await _send(
        llm,
        lambda: _format_prompt(template, segment, chunk.content),
        limiter,
        max_attempts,
        {"chunk_id": chunk.chunk_id},
    )
    if response is None:
        return _FAILED

    context = response_text(response)
    if not context:
        # A success carrying no usable text leaves the chunk as unfindable as
        # an outage would.
        logger.error("context_generation_empty", chunk_id=chunk.chunk_id)
        return _FAILED

    truncated = was_truncated(response)
    if truncated:
        logger.warning("context_truncated", chunk_id=chunk.chunk_id, context_chars=len(context))
    return _Outcome(context, False, truncated)


def _chunk_offsets(segment: str, chunks: Sequence[Chunk]) -> list[int] | None:
    """Where in the segment each chunk begins, or None if one is not there.

    Chunks are verbatim slices of their segment, so each is found by scanning
    on from the one before. One that is missing - an older build's segment, or
    a corpus edited since - leaves the batch protocol nothing to mark.
    """
    offsets: list[int] = []
    search_from = 0
    for chunk in chunks:
        found = segment.find(chunk.content, search_from)
        if found < 0:
            return None
        offsets.append(found)
        search_from = found + 1
    return offsets


def _mark(segment: str, marks: Sequence[tuple[int, int]]) -> str:
    """Return the segment with a [[position]] marker at each offset.

    Marking costs a few tokens where sending the chunks alongside the segment
    would cost a second copy of it. `marks` must be in offset order.
    """
    parts: list[str] = []
    cursor = 0
    for position, offset in marks:
        parts.append(segment[cursor:offset])
        parts.append(f"\n[[{position}]]\n")
        cursor = offset
    parts.append(segment[cursor:])
    return "".join(parts)


def _parse_marked_contexts(text: str, count: int) -> dict[int, str]:
    """Read a batch response as the contexts its markers label.

    Positions outside 1..count are dropped, as is a marker repeated after the
    first time it carried text. A reply to a one-chunk request is taken whole
    when it carries no marker: nothing else could be confused with it.
    """
    markers = list(_RESPONSE_MARKER.finditer(text))
    parsed: dict[int, str] = {}
    for order, marker in enumerate(markers):
        position = int(marker.group(1))
        if not 1 <= position <= count or position in parsed:
            continue
        end = markers[order + 1].start() if order + 1 < len(markers) else len(text)
        body = text[marker.end() : end].strip()
        if body:
            parsed[position] = body

    if not parsed and count == 1:
        body = _RESPONSE_MARKER.sub("", text).strip()
        if body:
            return {1: body}
    return parsed


def _batch_max_tokens(count: int) -> int:
    """Response ceiling for `count` contexts in one reply, rounded to a bucket."""
    needed = BATCH_TOKENS_OVERHEAD + BATCH_TOKENS_PER_CHUNK * count
    return -(-needed // BATCH_TOKEN_BUCKET) * BATCH_TOKEN_BUCKET


async def _request_marked_contexts(
    llm: ChatOpenAI,
    template: str,
    segment: str,
    marks: Sequence[tuple[int, int]],
    limiter: asyncio.Semaphore,
    max_attempts: int,
    log_context: dict[str, Any],
) -> tuple[dict[int, str], bool]:
    """Ask one request to describe every marked chunk of a segment.

    `segment` is passed unmarked and `marks` names the chunks this request
    covers. Returns the contexts the reply labelled and whether it hit the
    token ceiling; the positions it left out are the caller's to ask for again.
    """
    count = len(marks)
    response = await _send(
        llm,
        lambda: template.format(segment=_mark(segment, marks), count=count),
        limiter,
        max_attempts,
        log_context,
    )
    if response is None:
        return {}, False

    text = response_text(response)
    if not text:
        logger.error("context_generation_empty", **log_context)
        return {}, False

    parsed = _parse_marked_contexts(text, count)
    truncated = was_truncated(response)
    if truncated and parsed and count > 1:
        # The last context the reply labelled is the one that was cut. Dropped
        # so it is asked for again rather than indexed half-written.
        parsed.pop(max(parsed))
        logger.warning("segment_batch_truncated", **log_context, kept=len(parsed))
    return parsed, truncated


async def _generate_per_segment(
    segment: str,
    chunks: list[Chunk],
    template: str,
    model: str | None,
    timeout: float,
    limiter: asyncio.Semaphore,
) -> list[_Outcome] | None:
    """Contextualize a segment's chunks in one request, retrying what it missed.

    Returns None when the chunks cannot be marked up in the segment, leaving
    the caller to fall back to the per-chunk protocol.
    """
    offsets = None if not segment else _chunk_offsets(segment, chunks)
    if offsets is None:
        return None

    log_context = {
        "segment_key": encode_segment_key((chunks[0].source, chunks[0].segment_idx)),
        "num_chunks": len(chunks),
    }
    marks = list(enumerate(offsets, start=1))
    contexts, _ = await _request_marked_contexts(
        get_llm(model=model, timeout=timeout, max_tokens=_batch_max_tokens(len(chunks))),
        template,
        segment,
        marks,
        limiter,
        MAX_RETRIES,
        log_context,
    )

    outcomes = [_FAILED] * len(chunks)
    for position, context in contexts.items():
        outcomes[position - 1] = _Outcome(context, False, False)

    missing = [index for index, outcome in enumerate(outcomes) if outcome.failed]
    if not missing:
        return outcomes

    # Asked for on its own under the same prompt, so a reply that came back
    # short costs those chunks a retry rather than the segment its contexts.
    logger.warning("segment_batch_incomplete", **log_context, retried=len(missing))
    alone = get_llm(model=model, timeout=timeout, max_tokens=_batch_max_tokens(1))
    retried = await asyncio.gather(
        *(
            _request_marked_contexts(
                alone,
                template,
                segment,
                [(1, offsets[index])],
                limiter,
                MAX_RETRIES,
                {"chunk_id": chunks[index].chunk_id},
            )
            for index in missing
        )
    )
    for index, (contexts, truncated) in zip(missing, retried, strict=True):
        context = contexts.get(1, "")
        outcomes[index] = _Outcome(context, False, truncated) if context else _FAILED

    return outcomes


async def generate_context(
    segment: str,
    chunk: Chunk,
    model: str | None = None,
    prompt_template: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Generate context for a single chunk using LLM.

    Args:
        segment: The parent segment text (~8000 tokens).
        chunk: The chunk to generate context for.
        model: LLM model to use. Defaults to settings.CONTEXT_MODEL.
        prompt_template: Prompt template. Loads from file if not provided.
        timeout: Request timeout in seconds.

    Returns:
        Generated context string (100-200 tokens), or an empty string if the
        request failed.
    """
    outcome = await _request_context(
        llm=get_llm(model=model, timeout=timeout),
        template=prompt_template or load_prompt_template(),
        segment=segment,
        chunk=chunk,
        limiter=call_limiter(),
        max_attempts=1,
    )
    return outcome.context


async def generate_contexts_batch(
    segment: str,
    chunks: list[Chunk],
    model: str | None = None,
    prompt_template: str | None = None,
    timeout: float = 120.0,
    skip_context: bool = False,
    strategy: ContextStrategy | None = None,
) -> ContextBatch:
    """Generate context for multiple chunks using LLM.

    Under "per_segment" the segment is sent once, marked where each chunk
    begins, and every context is read out of the one reply; only what that
    reply left out is asked for again. Under "per_chunk" each chunk gets its
    own request. Both run concurrently up to the shared ceiling (see
    call_limiter) and retry with exponential backoff.

    Args:
        segment: The parent segment text (~8000 tokens).
        chunks: List of chunks to generate context for.
        model: LLM model to use. Defaults to settings.CONTEXT_MODEL.
        prompt_template: Prompt template. Loads from file if not provided. Its
            placeholders decide the protocol, whatever `strategy` says.
        timeout: Request timeout in seconds.
        skip_context: If True, return chunks with empty context (standard RAG mode).
        strategy: Which protocol to use when no template is passed. Defaults to
            settings.CONTEXT_STRATEGY.

    Returns:
        A ContextBatch whose chunks are in the order they were passed. A chunk
        whose request could not be completed carries an empty context and is
        counted in `failed`.
    """
    if skip_context:
        return _without_context(chunks)

    if not chunks:
        return ContextBatch([], 0, 0)

    prompt = (
        ContextPrompt(_protocol_of(prompt_template), prompt_template)
        if prompt_template is not None
        else resolve_prompt(None, strategy or settings.CONTEXT_STRATEGY)
    )
    limiter = call_limiter()

    outcomes: list[_Outcome] | None = None
    if prompt.strategy == "per_segment" and len(chunks) >= BATCH_MIN_CHUNKS:
        outcomes = await _generate_per_segment(
            segment, chunks, prompt.template, model, timeout, limiter
        )

    if outcomes is None:
        # A per-segment prompt cannot name a single chunk, so a segment that
        # could not be marked up falls back to the packaged per-chunk prompt.
        template = (
            prompt.template if prompt.strategy == "per_chunk" else _packaged_prompt("per_chunk")
        )
        if prompt.strategy == "per_segment":
            logger.warning(
                "segment_not_markable",
                segment_key=encode_segment_key((chunks[0].source, chunks[0].segment_idx)),
                num_chunks=len(chunks),
            )
        llm = get_llm(model=model, timeout=timeout)
        outcomes = list(
            await asyncio.gather(
                *(
                    _request_context(llm, template, segment, chunk, limiter, MAX_RETRIES)
                    for chunk in chunks
                )
            )
        )

    return ContextBatch(
        [
            ContextualizedChunk(chunk=chunk, context=outcome.context)
            for chunk, outcome in zip(chunks, outcomes, strict=True)
        ],
        sum(outcome.failed for outcome in outcomes),
        sum(outcome.truncated for outcome in outcomes),
    )
