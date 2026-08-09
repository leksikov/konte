"""Chat model clients for context generation, answering and query processing.

Every outbound chat request goes through the factory in this module, so the
Backend.AI / vLLM override, the client retry budget and the instance cache are
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
        "max_retries": CLIENT_MAX_RETRIES,
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


def _build_backendai_client(
    *,
    timeout: float,
    max_tokens: int | None,
    extra_body: dict[str, Any] | None,
    log_event: str,
) -> ChatOpenAI:
    """Construct a client aimed at the configured Backend.AI / vLLM endpoint.

    Args:
        timeout: Request timeout in seconds.
        max_tokens: Ceiling on response tokens, or None for the model default.
        extra_body: Extra JSON merged into the request body by the server.
        log_event: Event name recorded when the connection is first opened.

    Returns:
        A configured ChatOpenAI instance pointed at the custom endpoint.
    """
    logger.info(
        log_event,
        endpoint=settings.BACKENDAI_ENDPOINT,
        model=settings.BACKENDAI_MODEL_NAME,
    )
    return _build_client(
        model=settings.BACKENDAI_MODEL_NAME or "",
        timeout=timeout,
        max_tokens=max_tokens,
        base_url=settings.BACKENDAI_ENDPOINT,
        # vLLM endpoints usually need no credential, but the client refuses to send none.
        api_key=settings.BACKENDAI_API_KEY or "not-needed",
        extra_body=extra_body,
    )


def _get_or_build(cache_key: str, build: Callable[[], ChatOpenAI]) -> ChatOpenAI:
    """Return the cached client for cache_key, building it on first request."""
    client = _client_cache.get(cache_key)
    if client is None:
        client = _client_cache[cache_key] = build()
    return client


def get_llm(model: str | None = None, timeout: float = 120.0, max_tokens: int = 400) -> ChatOpenAI:
    """Get or create a cached client for context generation and query processing.

    Routes to the Backend.AI / vLLM endpoint when BACKENDAI_ENDPOINT and
    BACKENDAI_MODEL_NAME are both set and the caller did not ask for a different
    model; otherwise talks to OpenAI.

    Args:
        model: Model name. Defaults to settings.CONTEXT_MODEL or BACKENDAI_MODEL_NAME.
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens for the LLM response.

    Returns:
        Cached ChatOpenAI instance.
    """
    if settings.use_backendai and model in (None, settings.BACKENDAI_MODEL_NAME):
        return _get_or_build(
            f"backendai_{settings.BACKENDAI_MODEL_NAME}_{timeout}_{max_tokens}",
            lambda: _build_backendai_client(
                timeout=timeout,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                log_event="using_backendai",
            ),
        )

    model_name = model or settings.CONTEXT_MODEL
    return _get_or_build(
        f"openai_{model_name}_{timeout}_{max_tokens}",
        lambda: _build_client(model=model_name, timeout=timeout, max_tokens=max_tokens),
    )


def get_answer_llm(timeout: float = 60.0) -> ChatOpenAI:
    """Get or create a cached client for answer generation.

    Uses Backend.AI when configured, otherwise OpenAI. Unlike the context client
    this one sets no token ceiling and leaves server-side reasoning enabled, so
    a reasoning model can deliberate before answering.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        Cached ChatOpenAI instance.
    """
    if settings.use_backendai:
        return _get_or_build(
            f"answer_backendai_{settings.BACKENDAI_MODEL_NAME}_{timeout}",
            lambda: _build_backendai_client(
                timeout=timeout,
                max_tokens=None,
                extra_body=None,
                log_event="using_backendai_for_answer",
            ),
        )

    return _get_or_build(
        f"answer_openai_{settings.CONTEXT_MODEL}_{timeout}",
        lambda: _build_client(model=settings.CONTEXT_MODEL, timeout=timeout),
    )


def response_text(response: BaseMessage) -> str:
    """Return a chat response as plain text.

    Multi-part content (a provider answering with content blocks instead of a
    string) carries nothing this library can use, so it degrades to the same
    empty result as a failed request.

    Args:
        response: Message returned by a chat client.

    Returns:
        The stripped text, or an empty string when there is none.
    """
    content = response.content
    return content.strip() if isinstance(content, str) else ""


def active_answer_model() -> str:
    """Return the model name that get_answer_llm() will talk to.

    Returns:
        The Backend.AI model when that endpoint is active, else the OpenAI model.
    """
    if settings.use_backendai:
        return settings.BACKENDAI_MODEL_NAME or settings.CONTEXT_MODEL
    return settings.CONTEXT_MODEL
