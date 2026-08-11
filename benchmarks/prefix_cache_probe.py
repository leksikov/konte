"""Does an endpoint reward the way konte schedules its context requests?

konte builds every context prompt as ``[segment ~8000 tokens][chunk ~800]``, so
all chunks of a segment share a long prefix. The documented intent is that a
server with prefix caching computes that prefix once per segment.

Whether that pays off depends on *when* the requests arrive, not just on what
they contain: a prefix cache is populated by a prefill that has finished, so
requests fired simultaneously can all miss it. Three conditions separate that
out:

``sequential_shared``
    One prefix, requests one after another. The best case, and the proof that
    the endpoint caches prefixes at all.

``concurrent_shared``
    One prefix, all requests at once. This is what konte does within a segment.

``concurrent_distinct``
    A different prefix per request, all at once. This is what interleaving
    segments looks like to the server.

If ``concurrent_shared`` is no faster than ``concurrent_distinct``, sharing a
prefix is buying nothing at that arrival pattern, and how segments are
scheduled cannot be costing anything either.

Endpoints are passed in; nothing here is specific to a provider or a model.

    uv run python -m benchmarks.prefix_cache_probe --label hosted \\
        --base-url https://... --model ... --api-key-env OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

from benchmarks.harness import RESULTS_DIR

#: Roughly one segment's worth of text, so the prefix is the size konte
#: actually sends and is comfortably over any provider's caching threshold.
FILLER = "품목분류 해설서 본문 단락으로 구성된 참고 자료입니다. "
PREFIX_REPEATS = 700
REQUESTS = 6
MAX_TOKENS = 8


def _prefix(marker: str) -> str:
    """A long prefix nothing has seen before, so a run starts genuinely cold."""
    return f"[{marker}]\n" + FILLER * PREFIX_REPEATS


async def _call(client, model: str, prefix: str, suffix: str) -> tuple[float, int | None]:
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prefix + suffix}],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    elapsed = time.perf_counter() - start
    details = getattr(response.usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    return elapsed, cached


async def probe(label: str, base_url: str | None, model: str, api_key: str) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    stamp = f"{label}-{int(time.time())}"
    result: dict = {"label": label, "model": model, "base_url": base_url}

    shared = _prefix(f"{stamp}-seq")
    sequential = [await _call(client, model, shared, f"\n청크 {i}") for i in range(REQUESTS)]

    shared = _prefix(f"{stamp}-conc")
    start = time.perf_counter()
    concurrent_shared = await asyncio.gather(
        *(_call(client, model, shared, f"\n청크 {i}") for i in range(REQUESTS))
    )
    wall_shared = time.perf_counter() - start

    start = time.perf_counter()
    concurrent_distinct = await asyncio.gather(
        *(_call(client, model, _prefix(f"{stamp}-d{i}"), f"\n청크 {i}") for i in range(REQUESTS))
    )
    wall_distinct = time.perf_counter() - start

    def summarize(samples, wall=None):
        times = [t for t, _ in samples]
        cached = [c for _, c in samples if c is not None]
        return {
            "first_s": times[0],
            "rest_median_s": statistics.median(times[1:]) if len(times) > 1 else None,
            "median_s": statistics.median(times),
            "wall_s": wall if wall is not None else sum(times),
            "cached_tokens": cached or None,
        }

    result["sequential_shared"] = summarize(sequential)
    result["concurrent_shared"] = summarize(concurrent_shared, wall_shared)
    result["concurrent_distinct"] = summarize(concurrent_distinct, wall_distinct)

    cold = result["sequential_shared"]["first_s"]
    warm = result["sequential_shared"]["rest_median_s"]
    result["caches_prefixes_when_sequential"] = bool(warm and cold > warm * 1.5)
    result["sharing_helps_when_concurrent"] = (
        result["concurrent_shared"]["wall_s"] < result["concurrent_distinct"]["wall_s"] * 0.85
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="name for this endpoint in the report")
    parser.add_argument("--base-url", default=None, help="omit for the default provider")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "prefix_cache.json")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env) or "not-needed"
    result = asyncio.run(probe(args.label, args.base_url, args.model, api_key))

    stored = {}
    if args.out.exists():
        stored = json.loads(args.out.read_text())
    stored[args.label] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stored, indent=2, default=str))

    seq, cs, cd = (
        result["sequential_shared"],
        result["concurrent_shared"],
        result["concurrent_distinct"],
    )
    print(f"[{args.label}] {args.model}")
    print(f"  sequential shared   : cold {seq['first_s']:.2f}s -> warm {seq['rest_median_s']:.2f}s")
    print(f"  concurrent shared   : wall {cs['wall_s']:.2f}s  median {cs['median_s']:.2f}s")
    print(f"  concurrent distinct : wall {cd['wall_s']:.2f}s  median {cd['median_s']:.2f}s")
    print(f"  caches prefixes when sequential : {result['caches_prefixes_when_sequential']}")
    print(f"  sharing helps when concurrent   : {result['sharing_helps_when_concurrent']}")
    for name, block in (("sequential", seq), ("concurrent shared", cs)):
        if block["cached_tokens"]:
            print(f"  reported cached_tokens ({name}): {block['cached_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
