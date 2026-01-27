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
    OPENAI_API_KEY: str | None = None  # Optional - Backend.AI is default

    # Custom Backend (vLLM with OpenAI schema)
    BACKENDAI_ENDPOINT: str | None = None  # Disabled - using OpenAI
    BACKENDAI_MODEL_NAME: str | None = None
    BACKENDAI_API_KEY: str | None = None  # Optional API key for BackendAI

    # Storage
    STORAGE_PATH: Path = Path("~/.konte")

    # Models
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CONTEXT_MODEL: str = "gpt-4.1-mini"  # OpenAI model for context/evaluation

    @property
    def use_backendai(self) -> bool:
        """Check if BackendAI custom endpoint should be used."""
        return bool(self.BACKENDAI_ENDPOINT and self.BACKENDAI_MODEL_NAME)

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

    # Prompt path (konte/config/settings.py -> konte -> prompts)
    PROMPT_PATH: Path = Path(__file__).parent.parent.parent / "prompts" / "context_prompt.txt"

    @field_validator("STORAGE_PATH", mode="before")
    @classmethod
    def expand_storage_path(cls, v: str | Path) -> Path:
        """Expand ~ in storage path."""
        return Path(v).expanduser()

    @field_validator("PROMPT_PATH", mode="before")
    @classmethod
    def resolve_prompt_path(cls, v: str | Path) -> Path:
        """Resolve prompt path."""
        return Path(v)


settings = Settings()
