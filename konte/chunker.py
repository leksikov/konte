"""Chunker module for segmenting and chunking documents."""

import re
from functools import cache, lru_cache
from pathlib import Path
from typing import NamedTuple

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


class _Split(NamedTuple):
    """The parts a text was split into, plus the token count measured to split it."""

    parts: list[str]
    total_tokens: int


class _SplitSpec(NamedTuple):
    """Token budget for one level of splitting."""

    max_tokens: int
    overlap_tokens: int

    @classmethod
    def for_segments(cls, size: int | None, overlap: int | None) -> "_SplitSpec":
        return cls(
            settings.SEGMENT_SIZE if size is None else size,
            settings.SEGMENT_OVERLAP if overlap is None else overlap,
        )

    @classmethod
    def for_chunks(cls, size: int | None, overlap: int | None) -> "_SplitSpec":
        return cls(
            settings.CHUNK_SIZE if size is None else size,
            settings.CHUNK_OVERLAP if overlap is None else overlap,
        )

    def split(self, text: str) -> _Split:
        """Split text by token count at sentence boundaries.

        Args:
            text: Text to split.

        Returns:
            The resulting parts and the token count of `text`.
        """
        if not text.strip():
            return _Split([], 0)

        total_tokens = count_tokens(text)
        if total_tokens <= self.max_tokens:
            return _Split([text], total_tokens)

        return _Split(_make_splitter(self).split_text(text), total_tokens)


@lru_cache(maxsize=16)
def _make_splitter(spec: _SplitSpec) -> RecursiveCharacterTextSplitter:
    """Return the shared sentence-boundary-aware splitter for a token budget."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING_NAME,
        chunk_size=spec.max_tokens,
        chunk_overlap=spec.overlap_tokens,
        separators=_SENTENCE_SEPARATORS,
        keep_separator="end",
    )


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
    return _SplitSpec.for_segments(segment_size, overlap).split(text).parts


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
    return _SplitSpec.for_chunks(chunk_size, overlap).split(text).parts


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
    logger.debug("segmentation_started", source=source)

    chunk_spec = _SplitSpec.for_chunks(chunk_size, chunk_overlap)
    segmented = _SplitSpec.for_segments(segment_size, segment_overlap).split(text)
    metadata = extract_metadata_from_source(source)

    chunks: list[Chunk] = []
    segments_map: dict[SegmentKey, str] = {}

    for seg_idx, segment in enumerate(segmented.parts):
        chunked = chunk_spec.split(segment)
        logger.debug(
            "segment_created",
            source=source,
            segment_index=seg_idx,
            token_count=chunked.total_tokens,
        )
        segments_map[(source, seg_idx)] = segment

        logger.debug(
            "chunking_segment",
            source=source,
            segment_index=seg_idx,
            num_chunks=len(chunked.parts),
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
            for chunk_idx, chunk_text in enumerate(chunked.parts)
        )

    logger.debug(
        "chunks_created",
        source=source,
        total_tokens=segmented.total_tokens,
        total_segments=len(segments_map),
        total_chunks=len(chunks),
    )
    return chunks, segments_map
