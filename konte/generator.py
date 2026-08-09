"""RAG answer generation module using LLM to generate grounded answers from retrieved chunks."""

from typing import Any

import structlog
from pydantic import BaseModel

from konte.llm import active_answer_model, get_answer_llm, response_text
from konte.models import RetrievalResponse

logger = structlog.get_logger()

DEFAULT_ANSWER_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Context (retrieved documents, ordered by relevance score):
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- Prioritize information from higher-scored chunks (score closer to 1.0 = more relevant)
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and direct
- Cite relevant parts of the context when helpful

Answer:"""


class GeneratedAnswer(BaseModel):
    """Model for LLM-generated answer with metadata."""

    answer: str
    sources_used: int
    model: str


def _format_metadata(metadata: dict[str, Any]) -> str:
    """Format metadata dict into readable string.

    Args:
        metadata: Chunk metadata dict.

    Returns:
        Formatted metadata string or empty string if no metadata.
    """
    parts = [f"{key}={value}" for key, value in metadata.items() if value is not None]
    return " | ".join(parts)


def _format_context(retrieval_response: RetrievalResponse, max_chunks: int = 10) -> str:
    """Format retrieved chunks into context string for LLM.

    Includes relevance score, source, metadata, generated context, and original
    chunk content for maximum information retrieval.

    Args:
        retrieval_response: Response from retrieval query.
        max_chunks: Maximum number of chunks to include.

    Returns:
        Formatted context string.
    """
    context_parts = []

    for position, result in enumerate(retrieval_response.results[:max_chunks], 1):
        header = f"[{position}] Score: {result.score:.3f} | Source: {result.source or 'unknown'}"
        metadata_str = _format_metadata(result.metadata)
        if metadata_str:
            header = f"{header} | {metadata_str}"

        if result.context:
            chunk_text = f"Context: {result.context}\n\nContent: {result.content}"
        else:
            chunk_text = result.content

        context_parts.append(f"{header}\n{chunk_text}")

    return "\n\n".join(context_parts)


async def generate_answer(
    question: str,
    retrieval_response: RetrievalResponse,
    prompt_template: str | None = None,
    max_chunks: int = 10,
    timeout: float = 60.0,
) -> GeneratedAnswer:
    """Generate an answer from retrieved chunks using LLM.

    Args:
        question: The user's question.
        retrieval_response: Response from retrieval query containing chunks.
        prompt_template: Custom prompt template with {context} and {question} placeholders.
        max_chunks: Maximum number of chunks to use as context.
        timeout: LLM request timeout in seconds.

    Returns:
        GeneratedAnswer with the LLM response and metadata. A failed request
        yields an answer describing the failure and zero sources used.
    """
    model_name = active_answer_model()

    if not retrieval_response.results:
        return GeneratedAnswer(
            answer="No relevant documents were found to answer this question.",
            sources_used=0,
            model=model_name,
        )

    template = prompt_template or DEFAULT_ANSWER_PROMPT
    prompt = template.format(
        context=_format_context(retrieval_response, max_chunks),
        question=question,
    )
    sources_used = min(len(retrieval_response.results), max_chunks)

    try:
        response = await get_answer_llm(timeout=timeout).ainvoke(prompt)
    except Exception as e:
        logger.error("answer_generation_failed", error_type=type(e).__name__, error=str(e))
        return GeneratedAnswer(
            answer=f"Failed to generate answer: {e}",
            sources_used=0,
            model=model_name,
        )

    answer_text = response_text(response)
    logger.info(
        "answer_generated",
        question_length=len(question),
        context_chunks=sources_used,
        answer_length=len(answer_text),
    )
    return GeneratedAnswer(answer=answer_text, sources_used=sources_used, model=model_name)
