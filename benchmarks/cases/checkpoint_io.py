"""What build checkpointing writes to disk.

Claim under test: the checkpoint was a single file rewritten in full after every
finished segment, and it carried every chunk contextualized so far - so each
chunk was written again once per remaining segment, and the bytes written grew
with the square of the segment count. It should now be an append-only log with
one record per segment.

Bytes *written* is the measurement, not the file's final size: the whole point
is that the old format rewrote what it already had. The checkpoint is also
deleted on a successful build, so its final size is zero on both revisions and
would say nothing at all.

Context generation goes to the local stub with no injected latency. This case is
about disk, and a real endpoint's round trips would make it a very slow way to
measure writes.
"""

from __future__ import annotations

import time

from benchmarks.build import build_project
from benchmarks.corpus import projects_dir, synthetic_document
from benchmarks.harness import Context, counting_writes, point_llm_at
from benchmarks.stub_endpoint import MODEL_NAME, stub_endpoint

#: konte segments at 8000 tokens with 800 of overlap, so a segment advances the
#: document by 7200 tokens - ten chunks' worth at the synthetic corpus's sizing.
CHUNKS_PER_SEGMENT = 10
DEFAULT_SEGMENTS = 100


def run(ctx: Context) -> dict:
    segments = int(ctx.options.get("segments", DEFAULT_SEGMENTS))
    document = synthetic_document(segments * CHUNKS_PER_SEGMENT)
    storage = projects_dir(f"checkpoint-{segments}-{ctx.revision}")
    name = f"checkpoint_{segments}"

    directory = storage / name
    if directory.exists():
        import shutil

        shutil.rmtree(directory)

    with stub_endpoint() as (base_url, state), point_llm_at(base_url, MODEL_NAME):
        with counting_writes("checkpoint") as written:
            start = time.perf_counter()
            project = build_project(
                name,
                storage,
                document,
                enable_faiss=False,
                skip_context=False,
            )
            build_seconds = time.perf_counter() - start

    chunks = len(getattr(project, "_contextualized_chunks", []) or [])
    return {
        "requested_segments": segments,
        "chunks_contextualized": chunks,
        "context_requests": state.requests,
        "checkpoint_bytes_written": written[0],
        "checkpoint_bytes_per_segment": written[0] / max(segments, 1),
        "build_seconds": build_seconds,
    }
