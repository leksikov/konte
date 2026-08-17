"""How one project is configured, independent of where it is stored."""

from pathlib import Path, PurePath

from pydantic import BaseModel, ConfigDict, Field, field_validator


def validate_project_name(name: str) -> str:
    """Return the name if it names one directory under the storage root.

    Callers join the name onto the storage path, so one the filesystem reads
    as more than a single component reaches outside the root — where
    delete_project() would remove whatever it found.

    Raises:
        ValueError: If the name is not a single path component.
    """
    if not name or name == ".." or "\0" in name or PurePath(name).name != name:
        raise ValueError(
            f"Invalid project name: {name!r}. A project name is one directory "
            "name under the storage root: no path separators, no '..', not absolute."
        )
    return name


class ProjectConfig(BaseModel):
    """Configuration for a project."""

    name: str
    storage_path: Path

    # Segmentation
    segment_size: int = 8000
    segment_overlap: int = 800

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 80

    # Context
    context_prompt_path: Path | None = None  # Per-project prompt override

    # Models
    embedding_model: str = "text-embedding-3-small"
    context_model: str | None = None  # Defaults to settings.LLM_MODEL or CONTEXT_MODEL

    # Index options
    enable_faiss: bool = True
    enable_bm25: bool = True

    # Fusion weights. Only their ratio matters, not their scale.
    fusion_weight_semantic: float = Field(default=0.5, ge=0.0)
    fusion_weight_lexical: float = Field(default=0.5, ge=0.0)

    model_config = ConfigDict(ser_json_path="str")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """A name read back from config.json is checked like any other."""
        return validate_project_name(value)
