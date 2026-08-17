"""Reading documents off disk and cutting them into a project's corpus."""

from collections.abc import Sequence
from pathlib import Path

import structlog

from konte.domain.config import ProjectConfig
from konte.domain.corpus import Corpus
from konte.ingest.chunker import create_chunks
from konte.ingest.loader import load_document
from konte.ingest.source_naming import duplicate_source, pick_source

logger = structlog.get_logger()


def add_documents(
    corpus: Corpus,
    config: ProjectConfig,
    file_paths: Sequence[str | Path],
) -> int:
    """Load, segment and chunk documents into an existing corpus.

    A document is filed under its filename, or under as much of its path as it
    takes to be unique — two `report.md` would otherwise collide.

    Args:
        corpus: Corpus to extend, in place.
        config: Segmentation and chunking sizes to cut with.
        file_paths: Document paths to add.

    Returns:
        Number of chunks created.

    Raises:
        ValueError: If a document's text is already in the corpus; the ones
            ahead of it in `file_paths` are already added.
    """
    chunks = corpus.chunks
    segments = corpus.segments

    added = 0
    taken = {chunk.source for chunk in chunks}

    for raw_path in file_paths:
        file_path = Path(raw_path)
        logger.info("loading_document", path=str(file_path))

        source = pick_source(file_path, taken)
        new_chunks, segments_map = create_chunks(
            text=load_document(file_path),
            source=source,
            segment_size=config.segment_size,
            segment_overlap=config.segment_overlap,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        duplicate = duplicate_source(segments, segments_map, source)
        if duplicate is not None:
            raise ValueError(
                f"{file_path} holds the same text as '{duplicate}', already in "
                f"project '{config.name}'. Indexing it twice would spend "
                "two of every response's results on one passage."
            )

        taken.add(source)
        segments.update(segments_map)
        chunks.extend(new_chunks)
        added += len(new_chunks)

        logger.info(
            "document_chunked",
            path=str(file_path),
            source=source,
            num_chunks=len(new_chunks),
        )

    logger.info("documents_added", total_chunks=len(chunks))
    return added
