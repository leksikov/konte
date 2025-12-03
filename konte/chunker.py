"""Chunker module for segmenting and chunking documents."""

import tiktoken

from konte.models import Chunk

# Use cl100k_base encoding (used by text-embedding-3-small and GPT-4)
_ENCODING = tiktoken.get_encoding("cl100k_base")


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
            # Find the character position we're at
            prefix_text = _ENCODING.decode(tokens[:end])
            boundary = _find_word_boundary(prefix_text, len(prefix_text), "backward")

            # Only adjust if we found a reasonable boundary
            if boundary > len(_ENCODING.decode(tokens[:start])) + 10:
                chunk_text = prefix_text[:boundary].strip()
                # Recalculate end based on actual chunk
                end = start + len(_ENCODING.encode(chunk_text))

        chunks.append(chunk_text.strip())

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
    segment_size: int = 8000,
    overlap: int = 800,
) -> list[str]:
    """Split document into segments of approximately segment_size tokens.

    Args:
        text: Document text to segment.
        segment_size: Target size in tokens for each segment.
        overlap: Number of tokens to overlap between segments.

    Returns:
        List of text segments.
    """
    return _split_by_tokens(text, segment_size, overlap)


def chunk_segment(
    text: str,
    chunk_size: int = 800,
    overlap: int = 80,
) -> list[str]:
    """Split segment into chunks of approximately chunk_size tokens.

    Args:
        text: Segment text to chunk.
        chunk_size: Target size in tokens for each chunk.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks.
    """
    return _split_by_tokens(text, chunk_size, overlap)


def create_chunks(
    text: str,
    source: str,
    segment_size: int = 8000,
    segment_overlap: int = 800,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """Create Chunk objects from document text.

    First segments the document, then chunks each segment.

    Args:
        text: Document text.
        source: Source filename for metadata.
        segment_size: Target size in tokens for segments.
        segment_overlap: Overlap between segments in tokens.
        chunk_size: Target size in tokens for chunks.
        chunk_overlap: Overlap between chunks in tokens.

    Returns:
        List of Chunk objects.
    """
    chunks = []
    segments = segment_document(text, segment_size, segment_overlap)

    for seg_idx, segment in enumerate(segments):
        segment_chunks = chunk_segment(segment, chunk_size, chunk_overlap)

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

    return chunks
