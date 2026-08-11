"""What ``import konte`` costs.

The claim under test: the tokenizer is resolved on first use rather than at
import. Resolving it means loading a BPE table - a file read when the tiktoken
cache is warm, an outbound download when it is cold - and the older revision
paid that on every ``import konte``, including ``konte --help``.

Two signals, because wall time alone is noisy at this scale:

- whether any encoding is registered in tiktoken immediately after importing
  konte, which is a yes/no fact rather than a measurement
- import wall time, warm cache and cold cache, over several trials
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmarks.harness import Context, subprocess_env, summarize

TRIALS = 7

#: Import konte, then ask tiktoken which encodings it has had to build. A
#: revision that resolves the tokenizer at import leaves one registered here;
#: one that defers it leaves none. Revision-agnostic - it probes tiktoken, not
#: konte's internals, so it works unchanged on both sides.
PROBE = (
    "import json, konte, tiktoken.registry as r; "
    "print(json.dumps(sorted(getattr(r, 'ENCODINGS', {}))))"
)


def _run(code: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def _time_import(cwd: Path, env: dict[str, str], trials: int) -> list[float]:
    durations = []
    for _ in range(trials):
        start = time.perf_counter()
        proc = _run("import konte", cwd, env)
        durations.append(time.perf_counter() - start)
        if proc.returncode != 0:
            raise RuntimeError(f"import konte failed:\n{proc.stderr[-2000:]}")
    return durations


def run(ctx: Context) -> dict:
    env = subprocess_env()
    warm = _time_import(ctx.root, env, TRIALS)

    probe = _run(PROBE, ctx.root, env)
    encodings = probe.stdout.strip() if probe.returncode == 0 else probe.stderr[-500:]

    # A cache directory that starts empty forces the BPE table to be fetched,
    # which is what a fresh container or CI runner actually experiences.
    with tempfile.TemporaryDirectory(prefix="tiktoken-cold-") as cold_dir:
        cold_env = {**env, "TIKTOKEN_CACHE_DIR": cold_dir}
        cold = _time_import(ctx.root, cold_env, 2)
        cold_cache_files = len(list(Path(cold_dir).iterdir()))

    return {
        "warm_import_ms": summarize(warm),
        "cold_cache_import_ms": summarize(cold),
        # Files the import wrote into an empty tiktoken cache. Non-zero means
        # the import itself pulled the BPE table down.
        "cold_cache_files_written": cold_cache_files,
        "encodings_registered_after_import": encodings,
    }
