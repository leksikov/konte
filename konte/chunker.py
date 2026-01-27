"""Chunker module for segmenting and chunking documents."""

import structlog
import tiktoken

from konte.config import settings
from konte.models import Chunk

logger = structlog.get_logger()

# Use o200k_base encoding (used by gpt-4.1 and newer models - ~45% more efficient for Korean)
_ENCODING = tiktoken.encoding_for_model("gpt-4.1")


def count_tokens(text: str) -> int:
    """Count the number of tokens in text.

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    return len(_ENCODING.encode(text))


def _find_word_boundary(text: str, position: int, direction: str = "backward") -> int:
    """Find the nearest word boundary from a position.

    Args:
        text: The text to search in.
        position: Starting position.
        direction: "backward" or "forward".

    Returns:
        Position of word boundary.
    """
    if position >= len(text):
        return len(text)
    if position <= 0:
        return 0

    if direction == "backward":
        while position > 0 and text[position - 1] not in " \n\t":
            position -= 1
    else:
        while position < len(text) and text[position] not in " \n\t":
            position += 1

    return position


def _split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into chunks by token count with overlap.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to overlap.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    total_tokens = count_tokens(text)
    if total_tokens <= max_tokens:
        return [text]

    chunks = []
    tokens = _ENCODING.encode(text)
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))

        # Decode this chunk
        chunk_tokens = tokens[start:end]
        chunk_text = _ENCODING.decode(chunk_tokens)

        # Adjust to word boundary if not at the end
        if end < len(tokens):
            # Find word boundary within the chunk
            boundary = _find_word_boundary(chunk_text, len(chunk_text), "backward")

            # Only adjust if we found a reasonable boundary (at least 10 chars in)
            if boundary > 10:
                chunk_text = chunk_text[:boundary].strip()

        # Skip chunks that are too small (less than overlap size)
        stripped = chunk_text.strip()
        if len(_ENCODING.encode(stripped)) >= overlap_tokens:
            chunks.append(stripped)

        # Move start position with overlap
        step = max_tokens - overlap_tokens
        if step <= 0:
            step = max_tokens // 2

        start += step

        # Break if we've processed all content
        if start >= len(tokens):
            break

    return chunks


def segment_document(
    text: str,
    segment_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split document into segments of approximately segment_size tokens.

    Args:
        text: Document text to segment.
        segment_size: Target size in tokens for each segment. Defaults to settings.SEGMENT_SIZE.
        overlap: Number of tokens to overlap between segments. Defaults to settings.SEGMENT_OVERLAP.

    Returns:
        List of text segments.
    """
    size = segment_size if segment_size is not None else settings.SEGMENT_SIZE
    ovlp = overlap if overlap is not None else settings.SEGMENT_OVERLAP
    return _split_by_tokens(text, size, ovlp)


def chunk_segment(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split segment into chunks of approximately chunk_size tokens.

    Args:
        text: Segment text to chunk.
        chunk_size: Target size in tokens for each chunk. Defaults to settings.CHUNK_SIZE.
        overlap: Number of tokens to overlap between chunks. Defaults to settings.CHUNK_OVERLAP.

    Returns:
        List of text chunks.
    """
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    ovlp = overlap if overlap is not None else settings.CHUNK_OVERLAP
    return _split_by_tokens(text, size, ovlp)


def create_chunks(
    text: str,
    source: str,
    segment_size: int | None = None,
    segment_overlap: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[list[Chunk], dict[int, str]]:
    """Create Chunk objects from document text.

    First segments the document, then chunks each segment.

    Args:
        text: Document text.
        source: Source filename for metadata.
        segment_size: Target size in tokens for segments. Defaults to settings.SEGMENT_SIZE.
        segment_overlap: Overlap between segments in tokens. Defaults to settings.SEGMENT_OVERLAP.
        chunk_size: Target size in tokens for chunks. Defaults to settings.CHUNK_SIZE.
        chunk_overlap: Overlap between chunks in tokens. Defaults to settings.CHUNK_OVERLAP.

    Returns:
        Tuple of (List of Chunk objects, Dict mapping segment_idx to segment text).
    """
    total_tokens = count_tokens(text)
    logger.debug("segmentation_started", source=source, total_tokens=total_tokens)

    chunks = []
    segments_map: dict[int, str] = {}
    segments = segment_document(text, segment_size, segment_overlap)

    for seg_idx, segment in enumerate(segments):
        segment_tokens = count_tokens(segment)
        logger.debug(
            "segment_created",
            source=source,
            segment_index=seg_idx,
            token_count=segment_tokens,
        )
        segments_map[seg_idx] = segment
        segment_chunks = chunk_segment(segment, chunk_size, chunk_overlap)
        logger.debug(
            "chunking_segment",
            source=source,
            segment_index=seg_idx,
            num_chunks=len(segment_chunks),
        )

        for chunk_idx, chunk_text in enumerate(segment_chunks):
            chunk_id = f"{source}_s{seg_idx}_c{chunk_idx}"
            chunk = Chunk(
                chunk_id=chunk_id,
                content=chunk_text,
                source=source,
                segment_idx=seg_idx,
                chunk_idx=chunk_idx,
            )
            chunks.append(chunk)

    logger.debug(
        "chunks_created",
        source=source,
        total_segments=len(segments),
        total_chunks=len(chunks),
    )
    return chunks, segments_map
