"""Request and response schemas for the Konte API."""

from typing import Literal

from pydantic import BaseModel, Field

from konte.models import RetrievalResponse
from konte.generator import GeneratedAnswer


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., min_length=1, description="The search query")
    mode: Literal["hybrid", "semantic", "lexical"] = Field(
        default="hybrid",
        description="Retrieval mode",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of results to return",
    )


class AskRequest(BaseModel):
    """Request model for ask endpoint."""

    query: str = Field(..., min_length=1, description="The question to answer")
    mode: Literal["hybrid", "semantic", "lexical"] = Field(
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
