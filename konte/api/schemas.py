"""Request and response schemas for the Konte API."""

from pydantic import BaseModel, Field

from konte.answer import GeneratedAnswer
from konte.domain.models import RetrievalMode, RetrievalResponse

_KEYWORD_EXTRACTION_DESCRIPTION = (
    "Reduce the query to keywords before BM25 search, improving Korean lexical "
    "matching at the cost of one LLM call per distinct query. Omit to follow "
    "the server's BM25_KEYWORD_EXTRACTION setting; send false to skip it."
)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., min_length=1, description="The search query")
    mode: RetrievalMode = Field(
        default="hybrid",
        description="Retrieval mode",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    use_keyword_extraction: bool | None = Field(
        default=None,
        description=_KEYWORD_EXTRACTION_DESCRIPTION,
    )


class AskRequest(BaseModel):
    """Request model for ask endpoint."""

    query: str = Field(..., min_length=1, description="The question to answer")
    mode: RetrievalMode = Field(
        default="hybrid",
        description="Retrieval mode",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of chunks to retrieve",
    )
    max_chunks: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum chunks for answer generation",
    )
    use_keyword_extraction: bool | None = Field(
        default=None,
        description=_KEYWORD_EXTRACTION_DESCRIPTION,
    )


class AskResponse(BaseModel):
    """Response model for ask endpoint."""

    retrieval: RetrievalResponse
    answer: GeneratedAnswer


class ProjectListResponse(BaseModel):
    """Response model for list projects endpoint."""

    projects: list[str]
    total: int


class ProjectExistsResponse(BaseModel):
    """Response model for project exists endpoint."""

    name: str
    exists: bool
