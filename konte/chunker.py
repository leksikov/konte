"""Chunker module for segmenting and chunking documents."""

import re
from pathlib import Path

import structlog
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from konte.config import settings
from konte.models import Chunk

logger = structlog.get_logger()


def extract_metadata_from_source(source: str) -> dict[str, str]:
    """Extract metadata (company, year) from source filename.

    Parses filenames like:
    - ADOBE_2022_10K.md -> company: ADOBE, year: 2022
    - PEPSICO_2023Q1_EARNINGS.md -> company: PEPSICO, year: 2023
    - 3M_2018_10K.md -> company: 3M, year: 2018

    Args:
        source: Source filename or path.

    Returns:
        Dict with 'company' and 'year' keys (empty dict if not parsed).
    """
    filename = Path(source).stem
    match = re.match(r"^(.+?)_(\d{4})", filename, re.IGNORECASE)
    if match:
        return {
            "company": match.group(1).upper(),
            "year": match.group(2),
        }
    return {}

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


# Sentence-aware separators: paragraphs > lines > sentences > words
_SENTENCE_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "。",
    " ",
]


def _make_splitter(max_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """Create a sentence-boundary-aware splitter with token counting."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        separators=_SENTENCE_SEPARATORS,
        keep_separator="end",
    )


def _split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into chunks by token count at sentence boundaries.

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

    splitter = _make_splitter(max_tokens, overlap_tokens)
    return splitter.split_text(text)


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
) -> tuple[list[Chunk], dict[tuple[str, int], str]]:
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
        Tuple of (List of Chunk objects, Dict mapping (source, segment_idx) to segment text).
    """
    total_tokens = count_tokens(text)
    logger.debug("segmentation_started", source=source, total_tokens=total_tokens)

    chunks = []
    segments_map: dict[tuple[str, int], str] = {}
    segments = segment_document(text, segment_size, segment_overlap)

    for seg_idx, segment in enumerate(segments):
        segment_tokens = count_tokens(segment)
        logger.debug(
            "segment_created",
            source=source,
            segment_index=seg_idx,
            token_count=segment_tokens,
        )
        key = (source, seg_idx)
        segments_map[key] = segment
        segment_chunks = chunk_segment(segment, chunk_size, chunk_overlap)
        logger.debug(
            "chunking_segment",
            source=source,
            segment_index=seg_idx,
            num_chunks=len(segment_chunks),
        )

        # Extract metadata from source filename
        metadata = extract_metadata_from_source(source)

        for chunk_idx, chunk_text in enumerate(segment_chunks):
            chunk_id = f"{source}_s{seg_idx}_c{chunk_idx}"
            chunk = Chunk(
                chunk_id=chunk_id,
                content=chunk_text,
                source=source,
                segment_idx=seg_idx,
                chunk_idx=chunk_idx,
                metadata=metadata,
            )
            chunks.append(chunk)

    logger.debug(
        "chunks_created",
        source=source,
        total_segments=len(segments),
        total_chunks=len(chunks),
    )
    return chunks, segments_map
