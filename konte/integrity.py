"""Signatures over the index files a project deserializes when it opens.

Loading an index unpickles it, and unpickling runs whatever the file asks for.
So every index file is signed when it is written and checked before it is read:
an index that arrived from somewhere else — a shared volume, a download, another
machine — carries no signature this installation can verify, and is refused
instead of executed.

The key lives outside the directory it signs, in the storage root, because a key
shipped alongside the files it vouches for vouches for nothing. Deployments
whose storage root is itself shared can set INDEX_SIGNING_KEY and keep the key
off the volume entirely.
"""

import hmac
import os
import secrets
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

import structlog

from konte.config import settings
from konte.storage import read_json, write_json

logger = structlog.get_logger()

SIGNATURE_SUFFIX = ".sig"

_ALGORITHM = "hmac-sha256"
_KEY_FILENAME = ".signing-key"
_KEY_SIZE = 32
_CHUNK_SIZE = 1 << 20


class IntegrityError(Exception):
    """An index file is unsigned, or no longer matches the signature it carries."""


def _create_key(path: Path) -> bytes:
    """Write a fresh key readable only by its owner, or read the one that won.

    Args:
        path: Key file to create.

    Returns:
        The key now on disk, this call's or a concurrent one's.
    """
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
    """Return the key that signs the given index directory.

    Args:
        directory: Project directory holding the index files.

    Returns:
        The configured key, or the storage root's local key, created on first use.
    """
    if settings.INDEX_SIGNING_KEY:
        return settings.INDEX_SIGNING_KEY.encode("utf-8")

    path = directory.parent / _KEY_FILENAME
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return _create_key(path)


def _digest(key: bytes, path: Path) -> str:
    """Authenticate a file's name and contents under the key.

    The name is part of the message so a signature cannot be transplanted onto
    a different file.
    """
    mac = hmac.new(key, digestmod=sha256)
    mac.update(path.name.encode("utf-8") + b"\0")

    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_SIZE):
            mac.update(block)

    return mac.hexdigest()


def sign(directory: Path, filenames: Iterable[str]) -> None:
    """Sign index files, so a later load can tell they are this installation's.

    Args:
        directory: Project directory holding the files.
        filenames: Names of the files to sign, relative to the directory.
    """
    key = _key_for(directory)

    for name in filenames:
        digest = _digest(key, directory / name)
        write_json(
            directory / f"{name}{SIGNATURE_SUFFIX}",
            {"alg": _ALGORITHM, "digest": digest},
        )


def verify(directory: Path, filenames: Iterable[str]) -> None:
    """Check index files against their signatures before anything reads them.

    Args:
        directory: Project directory holding the files.
        filenames: Names of the files to check, relative to the directory.

    Raises:
        IntegrityError: If a file is unsigned, signed by another key, or was
            changed after it was signed.
    """
    key = _key_for(directory)

    for name in filenames:
        path = directory / name
        signature = read_json(path.with_name(f"{name}{SIGNATURE_SUFFIX}"))

        if signature is None:
            raise IntegrityError(
                f"{path} is not signed. Loading an index deserializes it, which runs "
                f"whatever it contains, so an unsigned one is never read. Rebuild the "
                f"project, or sign what is already there with "
                f"`konte trust {directory.name}` if you trust where it came from."
            )

        expected = signature.get("digest") if signature.get("alg") == _ALGORITHM else None
        if not isinstance(expected, str) or not hmac.compare_digest(
            expected, _digest(key, path)
        ):
            raise IntegrityError(
                f"{path} does not match its signature. It was written by a different "
                f"installation, or changed after it was signed; nothing was loaded. "
                f"Rebuild the project, or sign it with `konte trust {directory.name}` "
                f"if you trust where it came from."
            )
