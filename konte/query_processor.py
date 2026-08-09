"""Query preprocessing for better BM25 retrieval (Korean and English)."""

import structlog
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from konte.llm import get_llm

logger = structlog.get_logger()

_EXTRACTION_MAX_TOKENS = 800

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

# Bilingual prompt for keyword extraction
KEYWORD_EXTRACTION_PROMPT = """Extract search keywords from the query.

Rules:
1. Extract only meaningful nouns, verbs, proper nouns, and technical terms
2. Remove English stopwords (a, an, the, is, are, was, were, be, have, has, had, do, does, did, will, would, could, should, can, may, might, must, if, then, else, and, or, but, not, no, this, that, these, those, what, when, where, who, which, why, how, for, from, to, of, in, on, at, by, with, about, into, through, during, before, after, above, below, between, under, again, further, once, here, there, any, all, each, every, both, few, more, most, other, some, such, only, own, same, so, than, too, very, just, also, now, please, based, us, me, my, your, his, her, their, our, you, i, we, she, him, them)
3. Remove Korean particles (은/는/이/가/을/를/에/의/로 등) and stopwords (어느, 어떤, 무엇, 어디)
4. Keep compound terms together (e.g., "working capital" as one keyword, "의류 탈수기" as one keyword)
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


def _extraction_request(query: str) -> tuple[Runnable, str]:
    """Build the structured-output client and prompt for a keyword extraction.

    Args:
        query: Natural language query (Korean or English).

    Returns:
        Tuple of (client bound to the ExtractedKeywords schema, formatted prompt).
    """
    llm = get_llm(max_tokens=_EXTRACTION_MAX_TOKENS)
    return llm.with_structured_output(ExtractedKeywords), KEYWORD_EXTRACTION_PROMPT.format(
        query=query
    )


def extract_search_keywords(query: str) -> list[str]:
    """Extract keywords from query for BM25 search (supports Korean and English).

    Uses LLM with structured output to extract meaningful keywords,
    removing stopwords and particles.

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
    try:
        structured_llm, prompt = _extraction_request(query)
        result = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning("keyword_extraction_failed", query=query, error=str(e))
        return _fallback_tokenize(query)

    logger.debug("keywords_extracted", query=query, keywords=result.keywords)
    return result.keywords


async def extract_search_keywords_async(query: str) -> list[str]:
    """Async version of extract_search_keywords.

    Uses LLM with structured output to extract meaningful keywords,
    removing stopwords and particles.

    Args:
        query: Natural language query (Korean or English).

    Returns:
        List of clean keywords for BM25 search. Falls back to whitespace
        tokenization with stopword filtering if the LLM call fails.
    """
    try:
        structured_llm, prompt = _extraction_request(query)
        result = await structured_llm.ainvoke(prompt)
    except Exception as e:
        logger.warning("keyword_extraction_failed_async", query=query, error=str(e))
        return _fallback_tokenize(query)

    logger.debug("keywords_extracted_async", query=query, keywords=result.keywords)
    return result.keywords
