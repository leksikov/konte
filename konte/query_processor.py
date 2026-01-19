"""Query preprocessing for better Korean BM25 retrieval."""

import structlog
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from konte.config import settings

logger = structlog.get_logger()


class ExtractedKeywords(BaseModel):
    """Extracted keywords from query."""
    keywords: list[str]


_llm_instance: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Get cached LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=settings.CONTEXT_MODEL,
            temperature=0.0,
            max_tokens=100,
        )
    return _llm_instance


def extract_search_keywords(query: str) -> list[str]:
    """Extract clean Korean keywords from query for BM25 search.

    Removes particles (조사), endings (어미), and extracts core terms.

    Args:
        query: Natural language Korean query.

    Returns:
        List of clean keywords without particles.

    Example:
        Input: "의류 탈수기는 어느 HS 코드에 분류되나요?"
        Output: ["의류", "탈수기", "HS", "코드", "분류"]
    """
    prompt = f"""다음 한국어 질문에서 검색에 사용할 핵심 키워드를 추출하세요.

규칙:
1. 조사(은/는/이/가/을/를/에/의/로 등)를 제거한 순수 명사/동사 어간만 추출
2. 복합어는 분리하지 말고 그대로 유지 (예: "의류 탈수기" → "의류 탈수기")
3. HS 코드, 숫자는 그대로 유지
4. 불용어(어느, 어떤, 무엇, 어디) 제외
5. 3-7개 키워드 추출

질문: {query}

키워드만 쉼표로 구분하여 출력하세요."""

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)

        # Parse comma-separated keywords
        keywords_text = response.content.strip()
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

        logger.debug(
            "keywords_extracted",
            query=query,
            keywords=keywords,
        )

        return keywords

    except Exception as e:
        logger.warning(
            "keyword_extraction_failed",
            query=query,
            error=str(e),
        )
        # Fallback: simple whitespace split
        return query.split()


async def extract_search_keywords_async(query: str) -> list[str]:
    """Async version of extract_search_keywords."""
    prompt = f"""다음 한국어 질문에서 검색에 사용할 핵심 키워드를 추출하세요.

규칙:
1. 조사(은/는/이/가/을/를/에/의/로 등)를 제거한 순수 명사/동사 어간만 추출
2. 복합어는 분리하지 말고 그대로 유지 (예: "의류 탈수기" → "의류 탈수기")
3. HS 코드, 숫자는 그대로 유지
4. 불용어(어느, 어떤, 무엇, 어디) 제외
5. 3-7개 키워드 추출

질문: {query}

키워드만 쉼표로 구분하여 출력하세요."""

    try:
        llm = _get_llm()
        response = await llm.ainvoke(prompt)

        # Parse comma-separated keywords
        keywords_text = response.content.strip()
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

        logger.debug(
            "keywords_extracted_async",
            query=query,
            keywords=keywords,
        )

        return keywords

    except Exception as e:
        logger.warning(
            "keyword_extraction_failed_async",
            query=query,
            error=str(e),
        )
        # Fallback: simple whitespace split
        return query.split()
