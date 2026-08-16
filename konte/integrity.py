"""Integrity records over the index files a project reads when it opens.

INDEX_MANIFEST picks the anchor. Unset, files are signed with an HMAC key kept
in the storage root, so a record means "written by this installation" and every
other machine has to adopt what it did not build. Set, files are pinned by
SHA-256 in a manifest meant to be committed alongside them, so every checkout
verifies the same record with no secret to distribute.

INDEX_INTEGRITY turns the check down to a warning, or off.
"""

import hmac
import json
import os
import secrets
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import structlog

from konte.config import settings
from konte.storage import atomic_writer, read_json, write_json

logger = structlog.get_logger()

SIGNATURE_SUFFIX = ".sig"
MANIFEST_VERSION = 1

_HMAC_ALGORITHM = "hmac-sha256"
_DIGEST_ALGORITHM = "sha256"
_KEY_FILENAME = ".signing-key"
_KEY_SIZE = 32
_CHUNK_SIZE = 1 << 20


class IntegrityError(Exception):
    """An index file is unrecorded, or no longer matches what was recorded."""


def _manifest_path() -> Path | None:
    """Return the manifest that anchors trust, or None when the key file does."""
    configured = settings.INDEX_MANIFEST
    return Path(configured).expanduser() if configured else None


def _create_key(path: Path) -> bytes:
    """Write a fresh key readable only by its owner, or read the one that won."""
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return path.read_bytes()

    key = secrets.token_bytes(_KEY_SIZE)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(key)

    logger.info("signing_key_created", path=str(path))
    return key


def _key_for(directory: Path) -> bytes:
    """Return the configured key, or the storage root's own, created on first use."""
    if settings.INDEX_SIGNING_KEY:
        return settings.INDEX_SIGNING_KEY.encode("utf-8")

    path = directory.parent / _KEY_FILENAME
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return _create_key(path)


def _mac(key: bytes, path: Path) -> str:
    """Authenticate a file under the key.

    The name is part of the message, so a signature cannot be transplanted
    onto a different file.
    """
    mac = hmac.new(key, digestmod=sha256)
    mac.update(path.name.encode("utf-8") + b"\0")

    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_SIZE):
            mac.update(block)

    return mac.hexdigest()


def _hash(path: Path) -> str:
    """Digest a file's contents, so `shasum -a 256` reproduces a pinned entry."""
    digest = sha256()

    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_SIZE):
            digest.update(block)

    return digest.hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    """Read a JSON object, treating an absent or unusable file as an empty one."""
    try:
        data = read_json(path)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _pinned(manifest: Path, project: str) -> dict[str, Any]:
    """Return the digests pinned for one project, or an empty mapping."""
    projects = _read_json_or_empty(manifest).get("projects")
    if not isinstance(projects, dict):
        return {}
    entry = projects.get(project)
    return entry if isinstance(entry, dict) else {}


def _pin(manifest: Path, project: str, digests: dict[str, str]) -> None:
    """Record one project's digests, leaving the entries around it alone."""
    projects = _read_json_or_empty(manifest).get("projects")
    if not isinstance(projects, dict):
        projects = {}
    projects[project] = dict(sorted(digests.items()))

    payload = {
        "version": MANIFEST_VERSION,
        "algorithm": _DIGEST_ALGORITHM,
        "projects": dict(sorted(projects.items())),
    }
    # Sorted, indented and newline-terminated: this file is read in review diffs.
    with atomic_writer(manifest, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _manifest_failure(manifest: Path, directory: Path, filenames: Iterable[str]) -> str | None:
    """Check files against the digests pinned in the manifest."""
    pinned = _pinned(manifest, directory.name)

    for name in filenames:
        path = directory / name
        expected = pinned.get(name)

        if not isinstance(expected, str):
            return (
                f"{path} is not pinned in {manifest}. An index is only read once the "
                f"digest it should have is recorded somewhere the index itself cannot "
                f"reach. Rebuild the project, or pin what is already there with "
                f"`konte trust {directory.name}` and commit the manifest."
            )

        if not hmac.compare_digest(expected, _hash(path)):
            return (
                f"{path} does not match the digest pinned in {manifest}. It was replaced "
                f"or edited after it was pinned; nothing was loaded. Rebuild the project, "
                f"or re-pin it with `konte trust {directory.name}` if the change was yours."
            )

    return None


def _signature_failure(directory: Path, filenames: Iterable[str]) -> str | None:
    """Check files against the HMAC signatures written beside them."""
    key = _key_for(directory)

    for name in filenames:
        path = directory / name
        signature = _read_json_or_empty(path.with_name(f"{name}{SIGNATURE_SUFFIX}"))

        if not signature:
            return (
                f"{path} is not signed by this installation, so where it came from "
                f"cannot be told. Rebuild the project, sign what is already there with "
                f"`konte trust {directory.name}` if you trust its source, or set "
                f"INDEX_MANIFEST to verify against digests shared between machines."
            )

        expected = signature.get("digest") if signature.get("alg") == _HMAC_ALGORITHM else None
        if not isinstance(expected, str) or not hmac.compare_digest(expected, _mac(key, path)):
            return (
                f"{path} does not match its signature. It was written by a different "
                f"installation, or changed after it was signed; nothing was loaded. "
                f"Rebuild the project, or sign it with `konte trust {directory.name}` "
                f"if you trust where it came from."
            )

    return None


def sign(directory: Path, filenames: Iterable[str]) -> None:
    """Record index files, so a later load can tell they are the ones written.

    Args:
        directory: Project directory holding the files.
        filenames: Names of the files to record, relative to the directory.
    """
    if settings.INDEX_INTEGRITY == "off":
        return

    directory = Path(directory)
    names = list(filenames)
    manifest = _manifest_path()

    if manifest is not None:
        _pin(manifest, directory.name, {name: _hash(directory / name) for name in names})
        return

    key = _key_for(directory)
    for name in names:
        write_json(
            directory / f"{name}{SIGNATURE_SUFFIX}",
            {"alg": _HMAC_ALGORITHM, "digest": _mac(key, directory / name)},
        )


def verify(directory: Path, filenames: Iterable[str]) -> None:
    """Check index files against their record before anything reads them.

    Args:
        directory: Project directory holding the files.
        filenames: Names of the files to check, relative to the directory.

    Raises:
        IntegrityError: If a file is unrecorded or no longer matches, and
            INDEX_INTEGRITY is "enforce".
    """
    if settings.INDEX_INTEGRITY == "off":
        return

    directory = Path(directory)
    manifest = _manifest_path()
    failure = (
        _manifest_failure(manifest, directory, filenames)
        if manifest is not None
        else _signature_failure(directory, filenames)
    )

    if failure is None:
        return

    if settings.INDEX_INTEGRITY == "warn":
        logger.warning("index_integrity_failed", directory=str(directory), reason=failure)
        return

    raise IntegrityError(failure)
