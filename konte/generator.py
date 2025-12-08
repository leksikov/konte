"""RAG answer generation module using LLM to generate grounded answers from retrieved chunks."""

import structlog
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from konte.config import settings
from konte.models import RetrievalResponse

logger = structlog.get_logger()

# Module-level LLM instance cache for answer generation
_answer_llm_cache: dict[str, ChatOpenAI] = {}

# Default answer generation prompt
DEFAULT_ANSWER_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Context (retrieved documents):
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and direct
- Cite relevant parts of the context when helpful

Answer:"""


class GeneratedAnswer(BaseModel):
    """Model for LLM-generated answer with metadata."""

    answer: str
    sources_used: int
    model: str


def get_answer_llm(timeout: float = 60.0) -> ChatOpenAI:
    """Get or create a cached ChatOpenAI instance for answer generation.

    Uses BackendAI by default if configured, otherwise falls back to OpenAI.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        Cached ChatOpenAI instance.
    """
    if settings.use_backendai:
        model_name = settings.BACKENDAI_MODEL_NAME
        base_url = settings.BACKENDAI_ENDPOINT
        api_key = settings.BACKENDAI_API_KEY or "not-needed"
        cache_key = f"answer_backendai_{model_name}_{timeout}"

        if cache_key not in _answer_llm_cache:
            logger.info(
                "using_backendai_for_answer",
                endpoint=base_url,
                model=model_name,
            )
            _answer_llm_cache[cache_key] = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0,
                timeout=timeout,
                max_retries=2,
            )
        return _answer_llm_cache[cache_key]

    # Fallback to OpenAI
    model_name = settings.CONTEXT_MODEL
    cache_key = f"answer_openai_{model_name}_{timeout}"

    if cache_key not in _answer_llm_cache:
        _answer_llm_cache[cache_key] = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=timeout,
            max_retries=2,
        )

    return _answer_llm_cache[cache_key]


def _format_context(retrieval_response: RetrievalResponse, max_chunks: int = 10) -> str:
    """Format retrieved chunks into context string for LLM.

    Args:
        retrieval_response: Response from retrieval query.
        max_chunks: Maximum number of chunks to include.

    Returns:
        Formatted context string.
    """
    chunks_to_use = retrieval_response.results[:max_chunks]

    context_parts = []
    for i, result in enumerate(chunks_to_use, 1):
        source = result.source or "unknown"
        context_parts.append(f"[{i}] Source: {source}\n{result.content}")

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
        GeneratedAnswer with the LLM response and metadata.
    """
    if not retrieval_response.results:
        return GeneratedAnswer(
            answer="No relevant documents were found to answer this question.",
            sources_used=0,
            model=settings.BACKENDAI_MODEL_NAME or settings.CONTEXT_MODEL,
        )

    template = prompt_template or DEFAULT_ANSWER_PROMPT
    context = _format_context(retrieval_response, max_chunks)
    prompt = template.format(context=context, question=question)

    llm = get_answer_llm(timeout=timeout)
    model_name = settings.BACKENDAI_MODEL_NAME if settings.use_backendai else settings.CONTEXT_MODEL

    try:
        response = await llm.ainvoke(prompt)
        answer_text = response.content.strip() if response.content else ""

        logger.info(
            "answer_generated",
            question_length=len(question),
            context_chunks=min(len(retrieval_response.results), max_chunks),
            answer_length=len(answer_text),
        )

        return GeneratedAnswer(
            answer=answer_text,
            sources_used=min(len(retrieval_response.results), max_chunks),
            model=model_name,
        )
    except Exception as e:
        logger.error("answer_generation_failed", error=str(e))
        return GeneratedAnswer(
            answer=f"Failed to generate answer: {e}",
            sources_used=0,
            model=model_name,
        )
