"""Query preprocessing for better BM25 retrieval (Korean and English)."""

from collections.abc import Sequence
from functools import lru_cache
from typing import NamedTuple

import structlog
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from konte.domain.models import RetrievalRequest
from konte.runtime.llm import get_llm
from konte.runtime.settings import settings

logger = structlog.get_logger()

_EXTRACTION_MAX_TOKENS = 800

# A retry only multiplies the delay a stalled endpoint imposes on the waiting
# query, and the tokenizer fallback already covers the failure.
_EXTRACTION_MAX_RETRIES = 0

_CACHE_SIZE = 512

# English stopwords to filter in fallback tokenizer
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "can", "may",
    "might", "must", "if", "then", "else", "and", "or", "but", "not", "no",
    "this", "that", "these", "those", "what", "when", "where", "who", "which",
    "why", "how", "for", "from", "to", "of", "in", "on", "at", "by", "with",
    "about", "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "once", "here", "there", "any",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "than", "too", "very", "just", "also",
    "now", "please", "based", "us", "me", "my", "your", "his", "her", "their",
    "our", "you", "i", "we", "she", "him", "them", "am", "been", "being",
})

# Bilingual prompt for keyword extraction. BM25 matches a multi-word keyword
# term by term, so rule 4 asks for the modifier, not for the phrase.
KEYWORD_EXTRACTION_PROMPT = """Extract search keywords from the query.

Rules:
1. Extract only meaningful nouns, verbs, proper nouns, and technical terms
2. Remove English stopwords (a, an, the, is, are, was, were, be, have, has, had, do, does, did, will, would, could, should, can, may, might, must, if, then, else, and, or, but, not, no, this, that, these, those, what, when, where, who, which, why, how, for, from, to, of, in, on, at, by, with, about, into, through, during, before, after, above, below, between, under, again, further, once, here, there, any, all, each, every, both, few, more, most, other, some, such, only, own, same, so, than, too, very, just, also, now, please, based, us, me, my, your, his, her, their, our, you, i, we, she, him, them)
3. Remove Korean particles (은/는/이/가/을/를/에/의/로 등) and stopwords (어느, 어떤, 무엇, 어디)
4. Keep every word of a compound term, modifiers included (e.g., "working capital", not "capital"; "의류 탈수기", not "탈수기")
5. Keep codes, numbers, identifiers as-is (e.g., "FY2022", "HS 8471", "HS 코드")
6. Extract 3-10 keywords

Query: {query}"""


class ExtractedKeywords(BaseModel):
    """Extracted keywords from query."""
    keywords: list[str]


def _fallback_tokenize(query: str) -> list[str]:
    """Fallback tokenizer with stopword filtering."""
    tokens = query.split()
    return [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]


class _CacheSlot:
    """Mutable holder for one query's extracted keywords.

    The cache stores the slot rather than the keywords, so the sync and async
    paths share one set of entries and one eviction order whichever of them
    fills it. A slot left empty is how a failed extraction avoids being
    remembered.
    """

    __slots__ = ("keywords",)

    def __init__(self) -> None:
        self.keywords: tuple[str, ...] | None = None


@lru_cache(maxsize=_CACHE_SIZE)
def _cache_slot(query: str) -> _CacheSlot:
    """Return the slot for query, allocating it on first sight."""
    return _CacheSlot()


def clear_keyword_cache() -> None:
    """Forget every extraction cached so far.

    Entries are keyed by query text alone, so a change of endpoint, model or
    prompt keeps producing the previous keywords until the cache is emptied.
    """
    _cache_slot.cache_clear()


def _remember(query: str, result: ExtractedKeywords, event: str) -> list[str]:
    """Cache a successful extraction and hand back a copy the caller may mutate."""
    keywords = tuple(result.keywords)
    _cache_slot(query).keywords = keywords
    logger.debug(event, query=query, keywords=keywords)
    return list(keywords)


def _extraction_request(query: str) -> tuple[Runnable, str]:
    """Build the structured-output client and prompt for a keyword extraction.

    Unlike context generation, which runs once per chunk at build time, this
    call sits between a caller and their search results, so it gets its own
    short timeout instead of the batch-sized default.

    Args:
        query: Natural language query (Korean or English).

    Returns:
        Tuple of (client bound to the ExtractedKeywords schema, formatted prompt).
    """
    llm = get_llm(
        timeout=settings.KEYWORD_EXTRACTION_TIMEOUT,
        max_tokens=_EXTRACTION_MAX_TOKENS,
        max_retries=_EXTRACTION_MAX_RETRIES,
    )
    return llm.with_structured_output(ExtractedKeywords), KEYWORD_EXTRACTION_PROMPT.format(
        query=query
    )


def extract_search_keywords(query: str) -> list[str]:
    """Extract keywords from query for BM25 search (supports Korean and English).

    Uses LLM with structured output to extract meaningful keywords,
    removing stopwords and particles. A repeated query is answered from cache
    without another round trip. Blocks on the network; async callers want
    extract_search_keywords_async.

    Args:
        query: Natural language query (Korean or English).

    Returns:
        List of clean keywords for BM25 search. Falls back to whitespace
        tokenization with stopword filtering if the LLM call fails.

    Examples:
        Korean: "의류 탈수기는 어느 HS 코드에 분류되나요?"
        Output: ["의류 탈수기", "HS 코드", "분류"]

        English: "Does Paypal have positive working capital based on FY2022 data?"
        Output: ["Paypal", "positive", "working capital", "FY2022", "data"]
    """
    cached = _cache_slot(query).keywords
    if cached is not None:
        return list(cached)

    try:
        structured_llm, prompt = _extraction_request(query)
        result = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning("keyword_extraction_failed", query=query, error=str(e))
        return _fallback_tokenize(query)

    return _remember(query, result, "keywords_extracted")


async def extract_search_keywords_async(query: str) -> list[str]:
    """Async version of extract_search_keywords.

    Uses LLM with structured output to extract meaningful keywords,
    removing stopwords and particles. Shares its cache with the sync variant.

    Args:
        query: Natural language query (Korean or English).

    Returns:
        List of clean keywords for BM25 search. Falls back to whitespace
        tokenization with stopword filtering if the LLM call fails.
    """
    cached = _cache_slot(query).keywords
    if cached is not None:
        return list(cached)

    try:
        structured_llm, prompt = _extraction_request(query)
        result = await structured_llm.ainvoke(prompt)
    except Exception as e:
        logger.warning("keyword_extraction_failed_async", query=query, error=str(e))
        return _fallback_tokenize(query)

    return _remember(query, result, "keywords_extracted_async")


class Queries(NamedTuple):
    """The text each index is asked to rank against.

    `semantic` is the query exactly as the caller wrote it — an embedding model
    reads a natural-language question better than a bag of keywords. `lexical`
    is the same string unless keyword extraction reduced it for BM25.

    Resolving both up front lifts the one network-bound step out of the ranking
    helpers, leaving those pure and the async entry point an await apart.
    """

    semantic: str
    lexical: str


def _lexical_query(query: str, keywords: Sequence[str]) -> str:
    """Assemble the string BM25 will tokenize from an extraction result.

    An extraction that keeps nothing — an empty list from the model, or a
    question made entirely of stopwords reaching the fallback — would search
    for the empty string, scoring every chunk zero. The original query stands in.
    """
    search_query = " ".join(keywords)
    logger.debug(
        "bm25_keyword_extraction",
        original_query=query,
        keywords=keywords,
        search_query=search_query,
    )
    return search_query or query


def _extraction_applies(request: RetrievalRequest, has_lexical: bool) -> bool:
    """True when extraction would change what this retrieval actually reads.

    Semantic mode never reads the lexical query, and a project without a
    lexical index degrades to semantic whatever the mode asked for; neither
    should pay for a keyword call whose result is discarded.
    """
    if request.mode == "semantic" or not has_lexical:
        return False
    override = request.keyword_extraction
    return settings.BM25_KEYWORD_EXTRACTION if override is None else override


def resolve_queries(request: RetrievalRequest, has_lexical: bool) -> Queries:
    """Resolve the per-index query text, extracting keywords when asked to.

    Args:
        request: What this retrieval was asked for.
        has_lexical: Whether a non-empty lexical index is attached.

    Returns:
        The text each index searches. Blocks on the extraction call;
        resolve_queries_async is the variant that does not.
    """
    if not _extraction_applies(request, has_lexical):
        return Queries(request.query, request.query)
    keywords = extract_search_keywords(request.query)
    return Queries(request.query, _lexical_query(request.query, keywords))


async def resolve_queries_async(request: RetrievalRequest, has_lexical: bool) -> Queries:
    """Async twin of resolve_queries; the extraction call is the only difference.

    Args:
        request: What this retrieval was asked for.
        has_lexical: Whether a non-empty lexical index is attached.

    Returns:
        The text each index searches.
    """
    if not _extraction_applies(request, has_lexical):
        return Queries(request.query, request.query)
    keywords = await extract_search_keywords_async(request.query)
    return Queries(request.query, _lexical_query(request.query, keywords))
