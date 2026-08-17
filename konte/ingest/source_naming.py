"""Filing a document under a source name no other document in the project holds."""

from collections import defaultdict
from pathlib import Path

import structlog

from konte.domain.models import SegmentKey

logger = structlog.get_logger()


def pick_source(path: Path, taken: set[str]) -> str:
    """File a document under the shortest tail of its path no other one holds.

    Sharing a source name collides chunk ids and overwrites the segment map.
    The bare filename comes first: metadata filters are written against it.

    Args:
        path: Document being added.
        taken: Source names the project already uses.

    Returns:
        The name to file this document under.
    """
    resolved = path.resolve()
    parts = resolved.parts

    for depth in range(1, len(parts)):
        name = "/".join(parts[-depth:])
        if name not in taken:
            if depth > 1:
                logger.warning("document_source_disambiguated", path=str(resolved), source=name)
            return name

    name = str(resolved)
    suffix = 1
    while name in taken:
        suffix += 1
        name = f"{resolved}#{suffix}"
    return name


def duplicate_source(
    stored: dict[SegmentKey, str],
    incoming: dict[SegmentKey, str],
    source: str,
) -> str | None:
    """The document already in the project holding exactly these segments.

    Segments cover the whole text, so matching all of another's is that
    document again — under a second name, just handed to it by pick_source.

    Args:
        stored: Segment texts the project already holds.
        incoming: Segment texts the document being added produced.
        source: Name the incoming document was filed under.

    Returns:
        The name of the document already holding this text, or None.
    """
    if not incoming:
        return None

    segment_counts: defaultdict[str, int] = defaultdict(int)
    for other, _ in stored:
        segment_counts[other] += 1

    size = len(incoming)
    for other, count in segment_counts.items():
        if count == size and all(
            stored[(other, index)] == incoming[(source, index)] for index in range(size)
        ):
            return other
    return None
