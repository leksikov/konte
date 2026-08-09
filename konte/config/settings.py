"""Settings module - Single Source of Truth for configuration."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    RERANKER_MODEL: str = "Qwen3-Reranker-8B"
    RERANKER_VERIFY_SSL: bool = True

    # Storage
    STORAGE_PATH: Path = Path("~/.konte")

    # Models
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CONTEXT_MODEL: str = "gpt-4.1-mini"  # OpenAI model for context/evaluation

    # Segmentation
    SEGMENT_SIZE: int = 8000  # tokens
    SEGMENT_OVERLAP: int = 800  # tokens (10%)

    # Chunking
    CHUNK_SIZE: int = 800  # tokens
    CHUNK_OVERLAP: int = 80  # tokens (10%)

    # Context generation
    CONTEXT_MIN_TOKENS: int = 100
    CONTEXT_MAX_TOKENS: int = 200

    # Retrieval
    DEFAULT_TOP_K: int = 20
    CONFIDENCE_THRESHOLD: float = 0.7  # "deliver" threshold
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.4  # "query_more" threshold

    # Concurrency
    MAX_CONCURRENT_CALLS: int = 1  # Sequential processing to avoid rate limits

    # Prompt path — None means resolve at usage time via importlib.resources or __file__ fallback
    PROMPT_PATH: Path | None = None

    @property
    def use_custom_llm(self) -> bool:
        """Check if the custom chat endpoint should be used instead of OpenAI."""
        return bool(self.LLM_BASE_URL and self.LLM_MODEL)

    @field_validator("STORAGE_PATH", mode="before")
    @classmethod
    def expand_storage_path(cls, v: str | Path) -> Path:
        """Expand ~ in storage path."""
        return Path(v).expanduser()


settings = Settings()
