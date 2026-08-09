"""Context generation module using LLM to generate chunk context."""

import asyncio
import importlib.resources
from pathlib import Path
from weakref import WeakKeyDictionary

import structlog
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from konte.config import settings
from konte.llm import get_llm, response_text
from konte.models import Chunk, ContextualizedChunk

__all__ = [
    "call_limiter",
    "generate_context",
    "generate_contexts_batch",
    "get_llm",
    "load_prompt_template",
]

logger = structlog.get_logger()

MAX_RETRIES = 10
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 120.0  # seconds

_RETRYABLE_API_ERRORS = (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError)

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


def load_prompt_template(prompt_path: Path | None = None) -> str:
    """Load the context generation prompt template.

    Args:
        prompt_path: Path to prompt file. Defaults to settings.PROMPT_PATH,
            then falls back to the packaged default prompt.

    Returns:
        Prompt template string with {segment} and {chunk} placeholders.

    Raises:
        FileNotFoundError: If a configured prompt path does not exist.
    """
    path = prompt_path or settings.PROMPT_PATH
    if path is not None:
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Context prompt not found: {path}. The packaged default lives "
                "at konte/prompts/context_prompt.txt and domain examples at "
                "examples/prompts/; update PROMPT_PATH or the project's "
                "context_prompt_path in config.json."
            ) from e
    # Packaged default - read via importlib.resources so zip/frozen installs work
    ref = importlib.resources.files("konte") / "prompts" / "context_prompt.txt"
    return ref.read_text(encoding="utf-8")


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


def _without_context(chunks: list[Chunk]) -> list[ContextualizedChunk]:
    """Wrap chunks with an empty context, the degraded form used on any failure."""
    return [ContextualizedChunk(chunk=chunk, context="") for chunk in chunks]


async def _request_context(
    llm: ChatOpenAI,
    template: str,
    segment: str,
    chunk: Chunk,
    limiter: asyncio.Semaphore,
    max_attempts: int,
) -> str:
    """Ask for one chunk's context, retrying that chunk and no other.

    The prompt carries the whole segment, so it is formatted under the limiter:
    only the requests in flight hold a copy.

    Args:
        llm: Client to send the request through.
        template: Prompt template with {segment} and {chunk} placeholders.
        segment: The parent segment text (~8000 tokens).
        chunk: The chunk to describe.
        limiter: The in-flight ceiling to acquire for each attempt.
        max_attempts: How many times to send the request before giving up.

    Returns:
        The generated context, or an empty string once the attempts run out.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            async with limiter:
                response = await llm.ainvoke(_format_prompt(template, segment, chunk.content))
        except _RETRYABLE_API_ERRORS as e:
            error_type = type(e).__name__
            status_code = getattr(e, "status_code", None)
            if attempt == max_attempts:
                logger.error(
                    "api_retries_exhausted",
                    chunk_id=chunk.chunk_id,
                    error_type=error_type,
                    status_code=status_code,
                    error=str(e),
                )
                return ""
            delay = min(BASE_DELAY * 2 ** (attempt - 1), MAX_DELAY)
            logger.warning(
                "api_error_retrying",
                chunk_id=chunk.chunk_id,
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
                chunk_id=chunk.chunk_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            return ""
        else:
            return response_text(response)

    return ""  # Unreachable: the last attempt either returns or gives up above.


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
    return await _request_context(
        llm=get_llm(model=model, timeout=timeout),
        template=prompt_template or load_prompt_template(),
        segment=segment,
        chunk=chunk,
        limiter=call_limiter(),
        max_attempts=1,
    )


async def generate_contexts_batch(
    segment: str,
    chunks: list[Chunk],
    model: str | None = None,
    prompt_template: str | None = None,
    timeout: float = 120.0,
    skip_context: bool = False,
) -> list[ContextualizedChunk]:
    """Generate context for multiple chunks using LLM.

    One request per chunk, run concurrently up to the shared ceiling (see
    call_limiter) and retried individually with exponential backoff.

    Args:
        segment: The parent segment text (~8000 tokens).
        chunks: List of chunks to generate context for.
        model: LLM model to use. Defaults to settings.CONTEXT_MODEL.
        prompt_template: Prompt template. Loads from file if not provided.
        timeout: Request timeout in seconds.
        skip_context: If True, return chunks with empty context (standard RAG mode).

    Returns:
        List of ContextualizedChunk objects, in the order they were passed. A
        chunk whose request could not be completed carries an empty context.
    """
    if skip_context:
        return _without_context(chunks)

    if not chunks:
        return []

    template = prompt_template or load_prompt_template()
    llm = get_llm(model=model, timeout=timeout)
    limiter = call_limiter()

    contexts = await asyncio.gather(
        *(
            _request_context(llm, template, segment, chunk, limiter, MAX_RETRIES)
            for chunk in chunks
        )
    )

    return [
        ContextualizedChunk(chunk=chunk, context=context)
        for chunk, context in zip(chunks, contexts, strict=True)
    ]
