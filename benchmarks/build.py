"""Build a project identically on either revision.

The public ``Project`` surface is unchanged across the range under test, so one
helper drives both sides. Everything is passed by keyword: the newer revision
inserted a parameter into several signatures, which breaks positional callers
but leaves keyword callers alone.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from benchmarks.harness import assert_scratch_storage


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


def ensure_project(
    name: str,
    storage: Path,
    document: Path,
    **kwargs,
) -> Path:
    """Build a project only if it is not already on disk, and return its directory.

    Each revision needs its own build: the newer one writes a layout the older
    one cannot read back, so the two must never share a directory.
    """
    directory = project_dir(storage, name)
    if (directory / "config.json").exists():
        return directory
    build_project(name, storage, document, **kwargs)
    return directory


def open_project(name: str, storage: Path):
    konte = importlib.import_module("konte")
    return konte.Project.open(name, storage_path=storage)
