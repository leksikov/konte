"""Run benchmark cases against both revisions and store the pair.

Cases are run interleaved rather than all-of-one-revision-then-the-other, so a
machine that heats up, or an endpoint that slows down, biases both sides
equally instead of whichever ran second.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmarks.harness import RESULTS_DIR, REVISIONS, ensure_baseline, revision_sha, run_case

#: Cases needing no network. Fast, deterministic, and the ones worth running
#: first when changing the harness.
OFFLINE_CASES = (
    "import_cost",
    "chunking",
    "storage_size",
    "open_cache",
    "query_bm25",
    "query_faiss_filter",
    "keyword_extraction",
    "retrieval_parity",
    # Local stub endpoint, no embeddings.
    "checkpoint_io",
    # Local stub endpoint and stubbed embeddings unless endpoint=live is asked
    # for, which the live sweep does explicitly.
    "build_concurrency",
)

#: Cases that talk to the configured endpoint, or that build an index and so
#: need embeddings. Slower, and billed.
#:
#: `checkpoint_io` and `build_concurrency` are deliberately not here. Both
#: default to the local stub endpoint, and `build_concurrency` stubs embeddings
#: too, so both are reproducible and unbilled as run by default; only
#: `build_concurrency endpoint=live` needs a configured service, and that is
#: requested explicitly.
LIVE_CASES = (
    # Replays a golden question set; needs embeddings and the endpoint.
    "accuracy",
    # Needs the endpoint for generated answers and real embeddings for vector
    # retrieval, so the outputs compared are the ones a caller would get.
    "output_parity",
)

ALL_CASES = OFFLINE_CASES + LIVE_CASES


def run_pair(
    case: str,
    *,
    trials: int,
    timeout: float,
    options: dict | None = None,
    overrides: dict | None = None,
) -> dict:
    """Run one case on both revisions, interleaving trials."""
    runs: dict[str, list[dict]] = {revision: [] for revision in REVISIONS}
    for trial in range(trials):
        for revision in REVISIONS:
            started = time.time()
            print(f"  [{case}] trial {trial + 1}/{trials} {revision} ...", flush=True)
            result = run_case(case, revision, timeout=timeout, options=options, overrides=overrides)
            result["wall_seconds"] = time.time() - started
            runs[revision].append(result)
            status = result.get("status")
            if status != "ok":
                print(f"    -> {status} {result.get('reason') or ''}".rstrip(), flush=True)
    return {
        "case": case,
        "trials": trials,
        # Stamped here rather than derived when the report renders: a report
        # rendered later, or after the baseline ref moved, would otherwise
        # label these measurements with revisions that never produced them.
        "revisions": {revision: revision_sha(revision) for revision in REVISIONS},
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="case to run, repeatable; defaults to the offline set",
    )
    parser.add_argument("--all", action="store_true", help="run offline and live cases")
    parser.add_argument("--live", action="store_true", help="run only the live cases")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="case option applied to every case, repeatable",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "environment override for every case subprocess, repeatable. Use to point a "
            "run at a different endpoint; set both revisions' setting names, since they "
            "were renamed across the range and each revision reads only its own"
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help="suffix for the result filename, so a variant run does not overwrite the default",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    if args.cases:
        cases = tuple(args.cases)
    elif args.all:
        cases = ALL_CASES
    elif args.live:
        cases = LIVE_CASES
    else:
        cases = OFFLINE_CASES

    ensure_baseline()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for case in cases:
        print(f"[{case}]", flush=True)
        payload = run_pair(
            case,
            trials=args.trials,
            timeout=args.timeout,
            options=dict(item.partition("=")[::2] for item in args.option),
            overrides=dict(item.partition("=")[::2] for item in args.env),
        )
        stem = f"{case}_{args.label}" if args.label else case
        (args.out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))
        for revision, runs in payload["runs"].items():
            statuses = {run.get("status") for run in runs}
            if statuses - {"ok", "skipped"}:
                failures += 1
                print(f"  {case}/{revision}: FAILED", flush=True)
            elif "skipped" in statuses:
                # Not a failure: the case said what it needed and did not find it.
                print(f"  {case}/{revision}: skipped", flush=True)
    print(f"done; {failures} failed revision-runs", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
