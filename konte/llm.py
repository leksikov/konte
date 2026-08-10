"""Chat model clients for context generation, answering and query processing.

Every outbound chat request goes through the factory in this module, so the
custom-endpoint override, the client retry budget and the instance cache are
each configured in exactly one place.
"""

from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from konte.config import settings

logger = structlog.get_logger()

CLIENT_MAX_RETRIES = 2
_client_cache: dict[str, ChatOpenAI] = {}


def _build_client(
    *,
    model: str,
    timeout: float,
    max_tokens: int | None = None,
    max_retries: int = CLIENT_MAX_RETRIES,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> ChatOpenAI:
    """Construct an OpenAI-compatible chat client.

    Arguments left as None are omitted from the constructor call rather than
    forwarded. ChatOpenAI resolves a missing api_key or base_url from
    OPENAI_API_KEY and OPENAI_API_BASE; handing it an explicit None suppresses
    that fallback on some releases and silently breaks env-configured
    deployments.

    Args:
        model: Model name to request.
        timeout: Request timeout in seconds.
        max_tokens: Ceiling on response tokens, or None for the model default.
        max_retries: Client-side retry budget; each retry costs another full timeout.
        base_url: OpenAI-compatible endpoint, or None for the OpenAI default.
        api_key: Credential for that endpoint, or None to resolve from the env.
        extra_body: Extra JSON merged into the request body by the server.

    Returns:
        A configured, unshared ChatOpenAI instance.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    for name, value in (
        ("max_tokens", max_tokens),
        ("base_url", base_url),
        ("api_key", api_key),
        ("extra_body", extra_body),
    ):
        if value is not None:
            kwargs[name] = value
    return ChatOpenAI(**kwargs)


def _build_custom_client(
    *,
    timeout: float,
    max_tokens: int | None,
    max_retries: int = CLIENT_MAX_RETRIES,
    extra_body: dict[str, Any] | None,
    log_event: str,
) -> ChatOpenAI:
    """Construct a client aimed at the configured custom chat endpoint.

    Args:
        timeout: Request timeout in seconds.
        max_tokens: Ceiling on response tokens, or None for the model default.
        max_retries: Client-side retry budget; see _build_client.
        extra_body: Extra JSON merged into the request body by the server.
        log_event: Event name recorded when the connection is first opened.

    Returns:
        A configured ChatOpenAI instance pointed at the custom endpoint.
    """
    logger.info(
        log_event,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
    )
    return _build_client(
        model=settings.LLM_MODEL or "",
        timeout=timeout,
        max_tokens=max_tokens,
        max_retries=max_retries,
        base_url=settings.LLM_BASE_URL,
        # Self-hosted endpoints often need no credential, but the client demands one.
        api_key=settings.LLM_API_KEY or "not-needed",
        extra_body=extra_body,
    )


def _get_or_build(cache_key: str, build: Callable[[], ChatOpenAI]) -> ChatOpenAI:
    """Return the cached client for cache_key, building it on first request."""
    client = _client_cache.get(cache_key)
    if client is None:
        client = _client_cache[cache_key] = build()
    return client


def get_llm(
    model: str | None = None,
    timeout: float = 120.0,
    max_tokens: int = 400,
    max_retries: int = CLIENT_MAX_RETRIES,
) -> ChatOpenAI:
    """Get or create a cached client for context generation and query processing.

    Routes to the custom endpoint when LLM_BASE_URL and LLM_MODEL are both set
    and the caller did not ask for a different model; otherwise talks to OpenAI.

    Args:
        model: Model name. Defaults to settings.CONTEXT_MODEL or settings.LLM_MODEL.
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens for the LLM response.
        max_retries: Client-side retry budget. Worst-case latency is
            timeout * (max_retries + 1).

    Returns:
        Cached ChatOpenAI instance.
    """
    # Every argument reaching the constructor has to key the cache, or the
    # first caller's timeout and retry budget would be served to the next one.
    if settings.use_custom_llm and model in (None, settings.LLM_MODEL):
        return _get_or_build(
            f"custom_{settings.LLM_MODEL}_{timeout}_{max_tokens}_{max_retries}",
            lambda: _build_custom_client(
                timeout=timeout,
                max_tokens=max_tokens,
                max_retries=max_retries,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                log_event="using_custom_llm",
            ),
        )

    model_name = model or settings.CONTEXT_MODEL
    return _get_or_build(
        f"openai_{model_name}_{timeout}_{max_tokens}_{max_retries}",
        lambda: _build_client(
            model=model_name,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        ),
    )


def get_answer_llm(timeout: float = 60.0) -> ChatOpenAI:
    """Get or create a cached client for answer generation.

    Uses the custom endpoint when configured, otherwise OpenAI. Unlike the
    context client this one sets no token ceiling and leaves server-side
    reasoning enabled, so a reasoning model can deliberate before answering.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        Cached ChatOpenAI instance.
    """
    if settings.use_custom_llm:
        return _get_or_build(
            f"answer_custom_{settings.LLM_MODEL}_{timeout}",
            lambda: _build_custom_client(
                timeout=timeout,
                max_tokens=None,
                extra_body=None,
                log_event="using_custom_llm_for_answer",
            ),
        )

    return _get_or_build(
        f"answer_openai_{settings.CONTEXT_MODEL}_{timeout}",
        lambda: _build_client(model=settings.CONTEXT_MODEL, timeout=timeout),
    )


def _block_text(block: Any) -> str:
    """Return the text one content block carries, empty when it carries none."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        text = block.get("text")
        if isinstance(text, str):
            return text
    return ""


def response_text(response: BaseMessage) -> str:
    """Return a chat response as plain text.

    Content blocks have their text parts joined; a block carrying anything
    else, such as a reasoning trace, is dropped.

    Args:
        response: Message returned by a chat client.

    Returns:
        The stripped text, or an empty string when there is none.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(_block_text(block) for block in content).strip()
    return ""


def was_truncated(response: BaseMessage) -> bool:
    """Whether the model stopped because it ran into its token ceiling.

    Args:
        response: Message returned by a chat client.

    Returns:
        True when the provider reported hitting the limit.
    """
    return response.response_metadata.get("finish_reason") == "length"


def active_answer_model() -> str:
    """Return the model name that get_answer_llm() will talk to.

    Returns:
        The custom-endpoint model when that endpoint is active, else the OpenAI model.
    """
    if settings.use_custom_llm:
        return settings.LLM_MODEL or settings.CONTEXT_MODEL
    return settings.CONTEXT_MODEL
