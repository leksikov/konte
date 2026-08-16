"""On-disk layout of a project directory."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from konte.models import (
    Chunk,
    ContextualizedChunk,
    ProjectConfig,
    SegmentKey,
    decode_segment_key,
    encode_segment_key,
)
from konte.storage import read_json, write_json
from konte.stores import BM25Store, FAISSStore
from konte.stores.bm25_store import INDEX_FILENAME as BM25_INDEX_FILENAME
from konte.stores.bm25_store import LEGACY_INDEX_FILENAME as LEGACY_BM25_INDEX_FILENAME
from konte.stores.faiss_store import INDEX_FILENAME as FAISS_INDEX_FILENAME

logger = structlog.get_logger()

CONFIG_FILENAME = "config.json"
RAW_CHUNKS_FILENAME = "raw_chunks.json"
SEGMENTS_FILENAME = "segments.json"
CHUNKS_FILENAME = "chunks.json"

_CONFIG_INDENT = 2  # config.json is the one artifact people open by hand


class ProjectStore:
    """The artifacts a project keeps in one directory.

    Every corpus artifact reads back empty rather than raising when its file is
    absent, so a directory holding only some of them still opens.

    Args:
        directory: Directory the project's artifacts live in.
    """

    __slots__ = ("_directory",)

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        """Path: The directory this store reads and writes."""
        return self._directory

    def exists(self) -> bool:
        """Whether the project directory is there to be read."""
        return self._directory.exists()

    def write(
        self,
        config: ProjectConfig,
        *,
        chunks: list[Chunk],
        segments: dict[SegmentKey, str],
        contextualized_chunks: list[ContextualizedChunk],
        faiss: FAISSStore | None,
        bm25: BM25Store | None,
    ) -> None:
        """Replace every artifact of a project with the state passed in.

        Each file is replaced in one step, but the set of them is not
        transactional: a crash partway through leaves newer artifacts beside
        older ones, which a rebuild resolves. An index passed as None is left
        on disk as it stands.
        """
        directory = self._directory
        directory.mkdir(parents=True, exist_ok=True)

        write_json(
            directory / CONFIG_FILENAME,
            self._portable_config(config),
            indent=_CONFIG_INDENT,
        )
        write_json(directory / RAW_CHUNKS_FILENAME, [c.model_dump() for c in chunks])
        write_json(
            directory / SEGMENTS_FILENAME,
            {encode_segment_key(key): text for key, text in segments.items()},
        )
        write_json(
            directory / CHUNKS_FILENAME,
            [c.to_storage_dict() for c in contextualized_chunks],
        )

        if faiss is not None:
            faiss.save(directory)

        if bm25 is not None:
            bm25.save(directory)

        logger.info("project_saved", path=str(directory))

    def read_config(self, storage_root: Path) -> ProjectConfig:
        """Read the stored config, rebasing its relative paths on `storage_root`.

        Raises:
            FileNotFoundError: If the project's config.json does not exist.
        """
        config_path = self._directory / CONFIG_FILENAME

        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")

        data = json.loads(config_path.read_text(encoding="utf-8"))

        if not Path(data.get("storage_path", "")).is_absolute():
            data["storage_path"] = str(storage_root.resolve())

        prompt_path = data.get("context_prompt_path")
        if prompt_path and not Path(prompt_path).is_absolute():
            data["context_prompt_path"] = str((self._directory / prompt_path).resolve())

        return ProjectConfig(**data)

    def read_chunks(self) -> list[Chunk]:
        """Rebuild the raw chunk list, empty when the artifact is absent."""
        data = read_json(self._directory / RAW_CHUNKS_FILENAME)
        return [] if data is None else [Chunk(**item) for item in data]

    def read_segments(self) -> dict[SegmentKey, str]:
        """Rebuild the segment texts, empty when the artifact is absent."""
        data = read_json(self._directory / SEGMENTS_FILENAME)
        if data is None:
            return {}
        return {decode_segment_key(key): text for key, text in data.items()}

    def read_contextualized_chunks(self) -> list[ContextualizedChunk]:
        """Rebuild the contextualized chunks, empty when the artifact is absent."""
        data = read_json(self._directory / CHUNKS_FILENAME)
        if data is None:
            return []
        return [ContextualizedChunk.from_storage_dict(item) for item in data]

    def read_indexes(
        self,
        config: ProjectConfig,
        contextualized_chunks: Callable[[], list[ContextualizedChunk]],
    ) -> tuple[FAISSStore | None, BM25Store | None]:
        """Attach whichever indexes exist on disk and are enabled in config.

        A pickled index counts as present so the caller can refuse it by name,
        rather than the project opening quietly without one.

        Args:
            config: The project's configuration.
            contextualized_chunks: Called only if the lexical index needs its
                payload, so an unread corpus stays unparsed.

        Returns:
            The (semantic, lexical) pair, each None where absent.
        """
        directory = self._directory
        faiss: FAISSStore | None = None
        bm25: BM25Store | None = None

        if config.enable_faiss and (directory / FAISS_INDEX_FILENAME).exists():
            faiss = FAISSStore(embedding_model=config.embedding_model)
            faiss.load(directory)

        if config.enable_bm25 and (
            (directory / BM25_INDEX_FILENAME).exists()
            or (directory / LEGACY_BM25_INDEX_FILENAME).exists()
        ):
            bm25 = BM25Store()
            bm25.load(directory, contextualized_chunks)

        return faiss, bm25

    def _portable_config(self, config: ProjectConfig) -> dict[str, Any]:
        """Serialize the config with absolute paths made relative where possible.

        An absolute storage_path would pin the project to the machine that wrote
        it, so it is stored relative to the project's parent directory and
        rebased by read_config().
        """
        data = config.model_dump(mode="json")

        storage_path = Path(data["storage_path"])
        if storage_path.is_absolute():
            try:
                data["storage_path"] = str(storage_path.relative_to(self._directory.parent))
            except ValueError:
                data["storage_path"] = "."

        prompt_path = data.get("context_prompt_path")
        if prompt_path and Path(prompt_path).is_absolute():
            try:
                data["context_prompt_path"] = str(Path(prompt_path).relative_to(self._directory))
            except ValueError:
                pass  # Prompt lives outside the project; keep it absolute.

        return data
