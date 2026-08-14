"""Build a project identically on either revision.

The public ``Project`` surface is unchanged across the range under test, so one
helper drives both sides. Everything is passed by keyword: the newer revision
inserted a parameter into several signatures, which breaks positional callers
but leaves keyword callers alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.harness import assert_scratch_storage, isolate_revision, subprocess_env


def build_project(
    name: str,
    storage: Path,
    document: Path,
    *,
    enable_faiss: bool = False,
    enable_bm25: bool = True,
    skip_context: bool = True,
    metadata: dict | None = None,
):
    """Create, populate, build and save a project. Returns the built project.

    ``skip_context`` defaults to true because most cases measure storage layout
    and query paths, where generated context is a per-chunk string whose
    presence does not change what is being compared - and generating it for real
    would put an endpoint's latency inside a CPU measurement.
    """
    assert_scratch_storage(storage)
    konte = importlib.import_module("konte")

    project = konte.Project.create(
        name,
        storage_path=storage,
        enable_faiss=enable_faiss,
        enable_bm25=enable_bm25,
    )
    project.add_documents([document])
    if metadata:
        project.set_metadata(metadata)
    asyncio.run(
        project.build(
            skip_context=skip_context,
            enable_faiss=enable_faiss,
            enable_bm25=enable_bm25,
        )
    )
    project.save()
    return project


def project_dir(storage: Path, name: str) -> Path:
    return Path(storage) / name


def konte_revision_sha() -> str:
    """Commit the konte being imported right now was checked out at."""
    konte = importlib.import_module("konte")
    root = Path(konte.__file__).resolve().parent.parent
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _signature(document: Path, options: dict) -> dict:
    """What a cached project on disk has to match to be reusable."""
    return {
        "revision": konte_revision_sha(),
        "document": hashlib.blake2b(document.read_bytes(), digest_size=16).hexdigest(),
        "options": {key: options[key] for key in sorted(options)},
    }


def _signature_path(storage: Path, name: str) -> Path:
    """Beside the project, never inside it - the size cases measure that directory."""
    return Path(storage) / f".{name}.build.json"


def _build_in_subprocess(name: str, storage: Path, document: Path, options: dict) -> None:
    """Build in a fresh process, so its peak memory is not this process's peak.

    ``ru_maxrss`` is a high-water mark that never comes down, so a case that
    builds and then measures reads back the build's peak forever after.
    """
    payload = json.dumps(
        {"name": name, "storage": str(storage), "document": str(document), "options": options}
    )
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.build", payload],
        cwd=Path.cwd(),
        env=subprocess_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"isolated build of {name!r} failed:\n{proc.stderr[-4000:]}")


def ensure_project(
    name: str,
    storage: Path,
    document: Path,
    *,
    isolate: bool = False,
    **kwargs,
) -> Path:
    """Build a project unless one matching this exact input is already on disk.

    Each revision needs its own build: the newer one writes a layout the older
    one cannot read back, so the two must never share a directory.

    Reuse is keyed on a signature - the konte commit, a digest of the source
    document, and the build options - rather than on the directory merely
    existing. A sweep run after the code, the corpus generator or an option
    changed would otherwise report artifacts an earlier sweep produced, which
    is exactly the measurement `storage_size` makes.
    """
    directory = project_dir(storage, name)
    marker = _signature_path(storage, name)
    wanted = _signature(document, kwargs)

    if (directory / "config.json").exists() and marker.exists():
        try:
            if json.loads(marker.read_text()) == wanted:
                return directory
        except json.JSONDecodeError:
            pass

    if directory.exists():
        import shutil

        assert_scratch_storage(storage)
        shutil.rmtree(directory)

    if isolate:
        _build_in_subprocess(name, storage, document, kwargs)
    else:
        build_project(name, storage, document, **kwargs)

    marker.write_text(json.dumps(wanted, indent=2))
    return directory


def open_project(name: str, storage: Path):
    konte = importlib.import_module("konte")
    return konte.Project.open(name, storage_path=storage)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `_build_in_subprocess`: build one project, then exit."""
    isolate_revision()
    request = json.loads((argv or sys.argv[1:])[0])
    build_project(
        request["name"],
        Path(request["storage"]),
        Path(request["document"]),
        **request["options"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
