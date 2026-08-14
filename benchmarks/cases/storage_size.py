"""What a built project costs on disk.

Claims under test:

- the lexical index stored a tokenized copy of the corpus alongside the corpus
  itself, which it never needed to score with
- the corpus was written a second time under another name for the lexical store
- project JSON was pretty-printed and escaped non-ASCII, so a Korean corpus was
  stored as six-byte escapes

Each revision builds its own project from the same document. They cannot share
one: the newer layout is not readable by the older revision, and pointing the
newer revision at an older directory would measure the compatibility path
rather than the format it actually writes.

Lexical-only by default, so this case needs no embeddings and no endpoint. The
FAISS half of the layout is measured by the live build case, which has an index
to look at.
"""

from __future__ import annotations

from benchmarks.build import ensure_project
from benchmarks.corpus import document_excerpt, projects_dir, synthetic_document
from benchmarks.harness import Context, dir_breakdown, dir_bytes

DEFAULT_CHUNKS = 2000


def run(ctx: Context) -> dict:
    size = int(ctx.options.get("chunks", DEFAULT_CHUNKS))
    source = ctx.options.get("source", "synthetic")

    if source == "excerpt":
        document = document_excerpt(size)
    else:
        document = synthetic_document(size)

    # Per-revision storage: the two layouts are not interchangeable.
    storage = projects_dir(f"storage-{source}-{size}-{ctx.revision}")
    name = f"storage_{size}"
    directory = ensure_project(name, storage, document, enable_faiss=False, skip_context=True)

    text_bytes = document.stat().st_size
    total = dir_bytes(directory)

    return {
        "corpus": {
            "document": document.name,
            "requested_chunks": size,
            "source_text_bytes": text_bytes,
        },
        "project_dir_bytes": total,
        "project_dir_multiple_of_text": total / text_bytes,
        "files": dir_breakdown(directory),
    }
