"""What it costs to turn a document into chunks.

Two claims, both about work that was being done twice:

- every string was tokenized once to fill a debug log and again to decide where
  to split it, so the tokenizer ran twice over the whole document and twice over
  every segment
- a text splitter was constructed fresh for the document and for each segment,
  each construction resolving its own encoder, instead of being reused

Counts are the primary evidence here. Tokenizer calls and splitter constructions
are exact integers, where wall time on a CPU-bound loop is noisy. Both are
counted by patching the third-party libraries rather than konte, so the same
probe works on either revision.
"""

from __future__ import annotations

import importlib
import time

from benchmarks.corpus import synthetic_document
from benchmarks.harness import Context, counting, summarize

DEFAULT_CHUNKS = 500
TIMED_TRIALS = 3


def run(ctx: Context) -> dict:
    import tiktoken
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunker = importlib.import_module("konte.chunker")
    size = int(ctx.options.get("chunks", DEFAULT_CHUNKS))
    document = synthetic_document(size)
    text = document.read_text()

    # First call in a fresh process: the splitter cache, if this revision has
    # one, is still empty, so construction counts are the real per-build cost.
    with counting(tiktoken.Encoding, "encode") as encode_calls:
        with counting(RecursiveCharacterTextSplitter, "from_tiktoken_encoder") as splitters:
            start = time.perf_counter()
            chunks, segments = chunker.create_chunks(text, source=document.name)
            cold_seconds = time.perf_counter() - start

    # Repeat calls show the steady state, where a memoized splitter pays off.
    warm = []
    for _ in range(TIMED_TRIALS):
        start = time.perf_counter()
        chunker.create_chunks(text, source=document.name)
        warm.append(time.perf_counter() - start)

    return {
        "corpus": {
            "document": document.name,
            "characters": len(text),
            "chunks_produced": len(chunks),
            "segments_produced": len(segments),
        },
        "first_build": {
            "tokenizer_encode_calls": encode_calls[0],
            "splitter_constructions": splitters[0],
            "seconds": cold_seconds,
        },
        "repeat_build_ms": summarize(warm),
    }
