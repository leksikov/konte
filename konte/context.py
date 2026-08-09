"""Context generation module using LLM to generate chunk context."""

import asyncio
import importlib.resources
from pathlib import Path

import structlog
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
    template = prompt_template or load_prompt_template()
    llm = get_llm(model=model, timeout=timeout)
    prompt = _format_prompt(template, segment, chunk.content)

    try:
        response = await llm.ainvoke(prompt)
    except Exception as e:
        logger.warning("context_generation_failed", chunk_id=chunk.chunk_id, error=str(e))
        return ""
    return response_text(response)


async def generate_contexts_batch(
    segment: str,
    chunks: list[Chunk],
    model: str | None = None,
    prompt_template: str | None = None,
    timeout: float = 120.0,
    skip_context: bool = False,
) -> list[ContextualizedChunk]:
    """Generate context for multiple chunks using LLM batch.

    Uses LangChain's abatch() for efficient parallel processing with
    a single LLM instance, enabling OpenAI prompt caching.
    Implements exponential backoff for rate limit errors.

    Args:
        segment: The parent segment text (~8000 tokens).
        chunks: List of chunks to generate context for.
        model: LLM model to use. Defaults to settings.CONTEXT_MODEL.
        prompt_template: Prompt template. Loads from file if not provided.
        timeout: Request timeout in seconds.
        skip_context: If True, return chunks with empty context (standard RAG mode).

    Returns:
        List of ContextualizedChunk objects. Every chunk carries an empty
        context if the batch could not be generated.
    """
    if skip_context:
        return _without_context(chunks)

    if not chunks:
        return []

    template = prompt_template or load_prompt_template()
    llm = get_llm(model=model, timeout=timeout)

    prompts = [_format_prompt(template, segment, chunk.content) for chunk in chunks]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            responses = await llm.abatch(prompts, config={"max_concurrency": len(prompts)})
        except _RETRYABLE_API_ERRORS as e:
            delay = min(BASE_DELAY * 2 ** (attempt - 1), MAX_DELAY)
            error_type = type(e).__name__
            status_code = getattr(e, "status_code", None)
            logger.warning(
                "api_error_retrying",
                attempt=attempt,
                max_retries=MAX_RETRIES,
                delay=delay,
                error_type=error_type,
                status_code=status_code,
                error=str(e),
            )
            if attempt == MAX_RETRIES:
                logger.error(
                    "api_retries_exhausted",
                    error_type=error_type,
                    status_code=status_code,
                    error=str(e),
                )
                return _without_context(chunks)
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error(
                "batch_context_generation_failed",
                error_type=type(e).__name__,
                error=str(e),
            )
            return _without_context(chunks)
        else:
            return [
                ContextualizedChunk(chunk=chunk, context=response_text(response))
                for chunk, response in zip(chunks, responses, strict=True)
            ]

    # Unreachable: the final attempt either returns a batch or the degraded form.
    return _without_context(chunks)
