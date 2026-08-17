"""Settings module - Single Source of Truth for configuration."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from konte.domain.config import ContextStrategy


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    OPENAI_API_KEY: str | None = None  # Still needed for embeddings when FAISS is on

    # Custom OpenAI-compatible chat endpoint (vLLM, Ollama, LM Studio, ...)
    LLM_BASE_URL: str | None = None
    LLM_MODEL: str | None = None
    LLM_API_KEY: str | None = None

    # Reranker (optional - requires a vLLM server exposing a /score endpoint)
    RERANKER_BASE_URL: str | None = None
    RERANKER_MODEL: str | None = None
    RERANKER_VERIFY_SSL: bool = True

    # Storage
    STORAGE_PATH: Path = Path("~/.konte")

    # What happens to an index whose record is missing or stale.
    INDEX_INTEGRITY: Literal["enforce", "warn", "off"] = "enforce"

    # Manifest of index digests, for committing alongside the indexes it
    # covers: every checkout then verifies one record instead of adopting the
    # files once per machine.
    INDEX_MANIFEST: Path | None = None

    # Key the signatures are made with when no manifest anchors them. Unset
    # keeps it in the storage root; set it where that root is shared.
    INDEX_SIGNING_KEY: str | None = None

    # Models
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CONTEXT_MODEL: str = "gpt-4.1-mini"  # OpenAI model for context/evaluation

    # Segmentation
    SEGMENT_SIZE: int = 8000  # tokens
    SEGMENT_OVERLAP: int = 800  # tokens (10%)

    # Chunking
    CHUNK_SIZE: int = 800  # tokens
    CHUNK_OVERLAP: int = 80  # tokens (10%)

    # Retrieval
    DEFAULT_TOP_K: int = 20
    CONFIDENCE_THRESHOLD: float = 0.7  # "deliver" threshold
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.4  # "query_more" threshold

    # BM25 query preprocessing. Extraction costs one LLM round trip per
    # distinct query; turning it off here keeps retrieval purely in-memory.
    BM25_KEYWORD_EXTRACTION: bool = True
    KEYWORD_EXTRACTION_TIMEOUT: float = 5.0  # seconds before the tokenizer fallback

    # Serving
    PROJECT_CACHE_SIZE: int = 4  # opened projects held in memory, indexes included
    PRELOAD_PROJECTS: str = ""  # comma-separated names opened at API startup, "*" for all

    # Share of a corpus that may be indexed without generated context before
    # build() fails instead. 1.0 accepts any number of them.
    CONTEXT_FAILURE_THRESHOLD: float = 0.02

    # Concurrency
    MAX_CONCURRENT_CALLS: int = 16  # context and embedding requests in flight

    # Prompt path — None means resolve at usage time via importlib.resources or __file__ fallback
    PROMPT_PATH: Path | None = None

    # See ContextStrategy for the trade-off between the two.
    CONTEXT_STRATEGY: ContextStrategy = "per_segment"

    @property
    def use_custom_llm(self) -> bool:
        """Check if the custom chat endpoint should be used instead of OpenAI."""
        return bool(self.LLM_BASE_URL and self.LLM_MODEL)

    @property
    def concurrency_limit(self) -> int:
        """int: In-flight ceiling on outbound model calls, never below one."""
        return max(1, self.MAX_CONCURRENT_CALLS)

    @field_validator("STORAGE_PATH", mode="before")
    @classmethod
    def expand_storage_path(cls, v: str | Path) -> Path:
        """Expand ~ in storage path."""
        return Path(v).expanduser()


settings = Settings()
