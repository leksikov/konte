"""Crash-safe file primitives for the artifacts a project keeps on disk."""

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

_DEFAULT_FILE_MODE = 0o644


def _target_mode(path: Path) -> int:
    """Return the destination's permissions, or the default when it is absent."""
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return _DEFAULT_FILE_MODE


def _sync_directory(directory: Path) -> None:
    """Flush the rename itself; a no-op where directories cannot be synced."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@contextmanager
def atomic_writer(path: Path, *, encoding: str | None = None) -> Iterator[IO[Any]]:
    """Open a handle whose contents replace `path` once the block exits cleanly.

    An exception inside the block removes the temporary file and leaves the
    original intact.

    Args:
        path: Destination path. Its parent directory is created if missing.
        encoding: Text encoding, or None to open the handle in binary mode.

    Yields:
        The writable handle, closed on exit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Beside the target, so the replace stays within one filesystem.
    descriptor, raw_tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(raw_tmp_path)

    try:
        if encoding is None:
            handle = os.fdopen(descriptor, "wb")
        else:
            handle = os.fdopen(descriptor, "w", encoding=encoding, newline="\n")

        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())

        os.chmod(tmp_path, _target_mode(path))
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    _sync_directory(path.parent)


def write_json(path: Path, data: Any, *, indent: int | None = None) -> None:
    """Write data to path as UTF-8 JSON, atomically.

    Args:
        path: Destination path.
        data: Any JSON-serializable value.
        indent: Indentation, for files a person opens by hand. None keeps the
            compact form.
    """
    with atomic_writer(path, encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON from path, or return None when the file is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> tuple[list[Any], int]:
    """Read the intact records of a JSONL file, stopping at the first torn one.

    Args:
        path: JSONL file to read.

    Returns:
        Tuple of (records, byte offset just past the last intact record).
    """
    records: list[Any] = []
    offset = 0

    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            try:
                records.append(json.loads(line))
            except (UnicodeDecodeError, ValueError):
                break
            offset += len(line)

    return records, offset
