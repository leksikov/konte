"""Context generation module using LLM to generate chunk context."""

import asyncio
from pathlib import Path

import structlog
from langchain_openai import ChatOpenAI
from openai import RateLimitError

from konte.config import settings
from konte.models import Chunk, ContextualizedChunk

logger = structlog.get_logger()

# Rate limit retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds

# Module-level LLM instance cache
_llm_cache: dict[str, ChatOpenAI] = {}


def get_llm(model: str | None = None, timeout: float = 30.0) -> ChatOpenAI:
    """Get or create a cached ChatOpenAI instance.

    Args:
        model: Model name. Defaults to settings.CONTEXT_MODEL.
        timeout: Request timeout in seconds.

    Returns:
        Cached ChatOpenAI instance.
    """
    model_name = model or settings.CONTEXT_MODEL
    cache_key = f"{model_name}_{timeout}"

    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=timeout,
            max_retries=2,
        )

    return _llm_cache[cache_key]


def load_prompt_template(prompt_path: Path | None = None) -> str:
    """Load the context generation prompt template.

    Args:
        prompt_path: Path to prompt file. Defaults to settings.PROMPT_PATH.

    Returns:
        Prompt template string with {segment} and {chunk} placeholders.
    """
    path = prompt_path or settings.PROMPT_PATH
    return Path(path).read_text(encoding="utf-8")


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


async def generate_context(
    segment: str,
    chunk: Chunk,
    model: str | None = None,
    prompt_template: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Generate context for a single chunk using LLM.

    Args:
        segment: The parent segment text (~8000 tokens).
        chunk: The chunk to generate context for.
        model: LLM model to use. Defaults to settings.CONTEXT_MODEL.
        prompt_template: Prompt template. Loads from file if not provided.
        timeout: Request timeout in seconds.

    Returns:
        Generated context string (100-200 tokens).
    """
    template = prompt_template or load_prompt_template()
    llm = get_llm(model=model, timeout=timeout)
    prompt = _format_prompt(template, segment, chunk.content)

    try:
        response = await llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning("context_generation_failed", chunk_id=chunk.chunk_id, error=str(e))
        return ""


async def generate_contexts_batch(
    segment: str,
    chunks: list[Chunk],
    model: str | None = None,
    prompt_template: str | None = None,
    timeout: float = 30.0,
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
        List of ContextualizedChunk objects.
    """
    if skip_context:
        return [
            ContextualizedChunk(chunk=chunk, context="")
            for chunk in chunks
        ]

    if not chunks:
        return []

    template = prompt_template or load_prompt_template()
    llm = get_llm(model=model, timeout=timeout)

    # Format all prompts - segment stays the same (cacheable prefix)
    prompts = [_format_prompt(template, segment, chunk.content) for chunk in chunks]

    # Retry with exponential backoff for rate limits
    for attempt in range(MAX_RETRIES):
        try:
            # Use abatch for efficient parallel processing
            responses = await llm.abatch(prompts)

            return [
                ContextualizedChunk(
                    chunk=chunk,
                    context=response.content.strip() if response.content else ""
                )
                for chunk, response in zip(chunks, responses)
            ]
        except RateLimitError as e:
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            logger.warning(
                "rate_limit_hit_retrying",
                attempt=attempt + 1,
                max_retries=MAX_RETRIES,
                delay=delay,
                error=str(e)
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
            else:
                logger.error("rate_limit_retries_exhausted", error=str(e))
                return [
                    ContextualizedChunk(chunk=chunk, context="")
                    for chunk in chunks
                ]
        except Exception as e:
            logger.error("batch_context_generation_failed", error=str(e))
            return [
                ContextualizedChunk(chunk=chunk, context="")
                for chunk in chunks
            ]

    # Should not reach here, but return empty context as fallback
    return [
        ContextualizedChunk(chunk=chunk, context="")
        for chunk in chunks
    ]
