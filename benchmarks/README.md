# Benchmarks

Before/after performance comparison between two revisions of konte. Each case
runs against both revisions and the report states, claim by claim, what was
measured and whether it held up.

## Running

```bash
# offline cases: no endpoint, no billing, a few minutes
uv run python -m benchmarks.compare

# add the cases that build against the configured endpoint
uv run python -m benchmarks.compare --all

# one case, at a chosen scale
uv run python -m benchmarks.compare --case query_bm25 --option chunks=100000

# render benchmarks/RESULTS.md from whatever has been stored
uv run python -m benchmarks.report
```

A long sweep should not be interrupted by the machine sleeping:

```bash
nohup caffeinate -dis .venv/bin/python -m benchmarks.compare --all \
  > benchmarks/results/sweep.log 2>&1 &
```

## How the two revisions run side by side

The baseline lives in a git worktree, created automatically on first run:

```bash
git worktree add .benchmark-baseline <ref>    # KONTE_BENCH_BASELINE_REF
```

Both revisions share the repository's virtualenv. That is sound because the
dependency set is identical across the range under comparison, so konte's own
source is the only variable.

Revision selection is by working directory. `sys.path[0]` is the process cwd for
both `python -m` and `python -c`, so running from the worktree makes `import
konte` resolve there, while the repository root stays on `PYTHONPATH` so
`benchmarks` remains importable. `PYTHONPATH` alone is not enough - the cwd
entry precedes it.

The editable-install finder is removed from `sys.meta_path` before konte is
imported. Without that, a submodule the worktree lacks resolves out of the
repository instead of failing, so the baseline process would silently import
modules the baseline never had. `benchmarks/run.py` asserts konte loaded from
the expected tree and fails the run otherwise.

## Safety

Cases never write to real project storage. `Project.save()` on the newer
revision deletes the legacy lexical corpus file and writes a lexical index the
older revision cannot read, so one careless save would destroy the baseline
mid-comparison. Real projects are copied into scratch storage first, and
`assert_scratch_storage` raises on any attempt to write inside `~/.konte`.

Scratch defaults to `$TMPDIR/konte-bench`; override with `KONTE_BENCH_SCRATCH`.
The scale tiers run to tens of gigabytes.

## Corpora

| Tier | What | Used for |
|---|---|---|
| real | copies of already-built projects | open time, query latency, filter selectivity on real prose |
| excerpt | a token-sized slice of a real source document | live builds, and storage sizes on non-ASCII text |
| synthetic | generated from a fixed seed, materialized once | the scale tiers |

Synthetic corpora are written to disk and reused, so both revisions read
byte-identical input. Corpus construction never goes through konte - it must not
change when the code under test does.

## What is measured against a stub, and why

Most cases use the real thing. Three effects cannot be provoked from a real
endpoint, and all three are counts rather than throughput:

- what a stalled endpoint costs one query
- whether a repeated query goes back on the wire at all
- how many requests a single rate-limit response costs

Those use `stub_endpoint.py`, a local OpenAI-compatible server that counts
requests, injects latency, and returns 429s on cue.

The vector-filter case answers embedding requests locally. The effect it
measures is tens of microseconds to tens of milliseconds; a real embedding round
trip is hundreds of milliseconds with a long tail and would bury it. Both
revisions search with the same vector, so the comparison is unaffected.

`build_concurrency --option endpoint=live` builds against the configured
endpoint end to end.

## Method

- Revisions are interleaved rather than run one after the other, so a machine
  that heats up biases both sides equally.
- Median and IQR, not mean and standard deviation: latency is long-tailed.
- Cold and warm are reported separately wherever something is built lazily -
  the splitter cache, the lexical corpus, the filter postings, the chat client.
- Settings whose defaults moved across the range are pinned explicitly, so a
  default change is not mistaken for a code change.
- Keyword extraction is pinned to local tokenization in the query cases. The
  older revision cannot turn it off, and an LLM round trip inside the timed
  region would swamp what is being measured.

`retrieval_parity` is the gate on all of it: it records the ranking each
revision produces for a fixed query set and the report compares them. A faster
ranking that returns different results is not the same operation.

## Adding a case

Drop a module in `benchmarks/cases/` exposing `run(ctx) -> dict`, and add it to
`OFFLINE_CASES` or `LIVE_CASES` in `compare.py`. To surface it in the report,
add a `Claim` in `report.py` saying what was claimed and how to read the number
out of the measurements.
