"""Context generation module using LLM to generate chunk context."""

import asyncio
from pathlib import Path

import structlog
from langchain_openai import ChatOpenAI

from konte.models import Chunk, ContextualizedChunk
from konte.settings import settings

logger = structlog.get_logger()


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
    model_name = model or settings.CONTEXT_MODEL
    template = prompt_template or load_prompt_template()

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        timeout=timeout,
        max_retries=2,
    )

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
    max_concurrent: int | None = None,
    timeout: float = 30.0,
    skip_context: bool = False,
) -> list[ContextualizedChunk]:
    """Generate context for multiple chunks in parallel.

    Args:
        segment: The parent segment text (~8000 tokens).
        chunks: List of chunks to generate context for.
        model: LLM model to use. Defaults to settings.CONTEXT_MODEL.
        prompt_template: Prompt template. Loads from file if not provided.
        max_concurrent: Max parallel LLM calls. Defaults to settings.MAX_CONCURRENT_CALLS.
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

    concurrency = max_concurrent or settings.MAX_CONCURRENT_CALLS
    template = prompt_template or load_prompt_template()
    semaphore = asyncio.Semaphore(concurrency)

    async def _generate_with_semaphore(chunk: Chunk) -> ContextualizedChunk:
        async with semaphore:
            context = await generate_context(
                segment=segment,
                chunk=chunk,
                model=model,
                prompt_template=template,
                timeout=timeout,
            )
            return ContextualizedChunk(chunk=chunk, context=context)

    tasks = [_generate_with_semaphore(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    contextualized_chunks = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                "batch_context_generation_error",
                chunk_id=chunks[i].chunk_id,
                error=str(result),
            )
            contextualized_chunks.append(
                ContextualizedChunk(chunk=chunks[i], context="")
            )
        else:
            contextualized_chunks.append(result)

    return contextualized_chunks
