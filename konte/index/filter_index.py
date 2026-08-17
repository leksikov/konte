"""The inverted index both stores filter through, and the rule it filters by.

Filtering otherwise walks every chunk in Python on every query; the walk happens
once here instead, on the first filtered query. One implementation for both
stores, so a filter cannot select one set of chunks in the vector index and a
different set in the lexical one.
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from konte.domain.models import MetadataFilter

# One indexed document: the id to return it under, and the fields to post it on.
Fields = Iterable[tuple[str, Any]]
Entries = Iterable[tuple[int, Fields]]

_NO_IDS = np.empty(0, dtype=np.intp)


def matches_filter_value(actual: Any, expected: Any) -> bool:
    """Check one metadata field against one filter value.

    A list on the filter side means "match any"; any other value is compared for
    equality. Both stores route through this so a filter cannot mean one thing
    in the vector index and another in the lexical one.

    Args:
        actual: Value read from the chunk.
        expected: Value supplied in the filter.

    Returns:
        True if the field satisfies the filter.
    """
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


class FilterIndex:
    """Inverted index over a corpus: field value -> the ids carrying it.

    Args:
        entries: One (id, fields) pair per indexed document. A field whose
            values cannot be hashed stops being posted, and every filter
            naming it falls back to a scan.
        unposted: Fields to leave out whatever they hold, such as payload a
            filter never names and hashing would walk in full.
    """

    __slots__ = ("_absent", "_all_ids", "_postings", "_size", "_unposted")

    def __init__(self, entries: Entries, unposted: Iterable[str] = ()) -> None:
        postings: dict[str, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
        skipped = set(unposted)
        ids: list[int] = []

        for identifier, fields in entries:
            ids.append(identifier)
            for key, value in fields:
                if key in skipped:
                    continue
                try:
                    postings[key][value].append(identifier)
                except TypeError:
                    skipped.add(key)

        self._size = max(ids, default=-1) + 1
        self._all_ids = np.array(ids, dtype=np.intp)
        # Appended in the order the entries arrived, so posting lists ascend
        # like a scan's, and every merge below preserves that.
        self._postings = {
            key: {value: np.array(found, dtype=np.intp) for value, found in values.items()}
            for key, values in postings.items()
        }
        self._unposted = skipped
        self._absent: dict[str, np.ndarray] = {}

    def select(
        self,
        metadata_filter: MetadataFilter | None,
        source_filter: str | None,
    ) -> np.ndarray | None:
        """Return the ids satisfying both filters, or None to fall back to a scan.

        Args:
            metadata_filter: Filter results by metadata (equality match, AND logic).
            source_filter: Substring match on chunk source field.

        Returns:
            The matching ids, ascending; an empty array where the filters
            matched nothing; None where a filtered field is unposted and only
            a scan can answer.
        """
        fields = set(metadata_filter or ())
        if source_filter:
            fields.add("source")
        if fields & self._unposted:
            return None

        matched = [self._ids_for(key, value) for key, value in (metadata_filter or {}).items()]
        if source_filter:
            matched.append(self._ids_matching_source(source_filter))
        if not matched:
            return self._all_ids
        if len(matched) == 1:
            return matched[0]

        # A mask per field beats intersecting id lists, which re-sorts.
        mask = np.ones(self._size, dtype=bool)
        for ids in matched:
            keep = np.zeros(self._size, dtype=bool)
            keep[ids] = True
            mask &= keep
        return np.flatnonzero(mask)

    def _ids_for(self, field: str, expected: Any) -> np.ndarray:
        """Ids whose field satisfies one filter value; a list means match-any."""
        postings = self._postings.get(field, {})
        wanted = expected if isinstance(expected, list) else (expected,)

        found: list[np.ndarray] = []
        for value in wanted:
            try:
                posted = postings.get(value)
            except TypeError:  # an unhashable filter value equals no posted one
                continue
            if posted is not None:
                found.append(posted)
            if value is None:
                found.append(self._absent_from(field))
        return self._union(found)

    def _ids_matching_source(self, needle: str) -> np.ndarray:
        """Ids whose source contains the needle."""
        return self._union(
            [
                ids
                for source, ids in self._postings.get("source", {}).items()
                if isinstance(source, str) and needle in source
            ]
        )

    def _union(self, found: list[np.ndarray]) -> np.ndarray:
        """Merge posting lists of one field, which cannot overlap, back into order."""
        if not found:
            return _NO_IDS
        if len(found) == 1:
            return found[0]

        mask = np.zeros(self._size, dtype=bool)
        for ids in found:
            mask[ids] = True
        return np.flatnonzero(mask)

    def _absent_from(self, field: str) -> np.ndarray:
        """Ids of documents carrying no such field — a reader sees them as None."""
        absent = self._absent.get(field)
        if absent is None:
            mask = np.zeros(self._size, dtype=bool)
            mask[self._all_ids] = True
            for posted in self._postings.get(field, {}).values():
                mask[posted] = False
            self._absent[field] = absent = np.flatnonzero(mask)
        return absent
