"""Chunker module for segmenting and chunking documents."""

import re
from functools import cache
from pathlib import Path

import structlog
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from konte.config import settings
from konte.models import Chunk, SegmentKey

logger = structlog.get_logger()

# o200k_base (gpt-4.1 and newer) is ~45% more token-efficient for Korean than cl100k_base.
TOKEN_ENCODING_NAME = "o200k_base"

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

_SOURCE_METADATA_PATTERN = re.compile(r"^(.+?)_(\d{4})", re.IGNORECASE)


@cache
def _get_encoding() -> tiktoken.Encoding:
    """Return the shared tiktoken encoding, loading it on first use.

    tiktoken downloads and caches the BPE table the first time an encoding is
    resolved. Doing that at import time would make `import konte` reach out to
    the network, so the cost is deferred to the first token count instead.
    """
    return tiktoken.get_encoding(TOKEN_ENCODING_NAME)


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
    match = _SOURCE_METADATA_PATTERN.match(Path(source).stem)
    if match is None:
        return {}
    return {
        "company": match.group(1).upper(),
        "year": match.group(2),
    }


def count_tokens(text: str) -> int:
    """Count the number of tokens in text.

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def _make_splitter(max_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """Create a sentence-boundary-aware splitter with token counting."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING_NAME,
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
) -> tuple[list[Chunk], dict[SegmentKey, str]]:
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
    logger.debug("segmentation_started", source=source, total_tokens=count_tokens(text))

    metadata = extract_metadata_from_source(source)

    chunks: list[Chunk] = []
    segments_map: dict[SegmentKey, str] = {}

    for seg_idx, segment in enumerate(segment_document(text, segment_size, segment_overlap)):
        logger.debug(
            "segment_created",
            source=source,
            segment_index=seg_idx,
            token_count=count_tokens(segment),
        )
        segments_map[(source, seg_idx)] = segment

        segment_chunks = chunk_segment(segment, chunk_size, chunk_overlap)
        logger.debug(
            "chunking_segment",
            source=source,
            segment_index=seg_idx,
            num_chunks=len(segment_chunks),
        )

        chunks.extend(
            Chunk(
                chunk_id=f"{source}_s{seg_idx}_c{chunk_idx}",
                content=chunk_text,
                source=source,
                segment_idx=seg_idx,
                chunk_idx=chunk_idx,
                metadata=metadata,
            )
            for chunk_idx, chunk_text in enumerate(segment_chunks)
        )

    logger.debug(
        "chunks_created",
        source=source,
        total_segments=len(segments_map),
        total_chunks=len(chunks),
    )
    return chunks, segments_map
