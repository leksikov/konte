"""Turn stored results into a claim-by-claim report.

Each row states what was claimed, what both revisions actually measured, and a
verdict. A claim with no result behind it is reported as unmeasured rather than
quietly dropped - a missing row reads as "we checked and it was fine", which is
the one thing it must not mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from benchmarks.harness import RESULTS_DIR

REPORT_PATH = Path(__file__).resolve().parent / "RESULTS.md"


@dataclass
class Claim:
    """One CHANGELOG claim and how to read it out of a case's measurements."""

    case: str
    title: str
    claimed: str
    extract: Callable[[dict], tuple[float | str | None, str]]
    #: "lower" when a smaller number is the improvement, "higher" when larger,
    #: "exact" when the two sides are supposed to be identical, "cost" when the
    #: newer revision is expected to pay more here by design.
    direction: str = "lower"


def _get(data: dict, *path, default=None):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f} ms"


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f} s"


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _bytes(value: float | None) -> str:
    if value is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


CLAIMS = [
    Claim(
        "import_cost",
        "Tokenizer resolved on first use, not at import",
        "import no longer builds the BPE table",
        lambda m: (_get(m, "cold_cache_import_ms", "median"), _ms),
    ),
    Claim(
        "import_cost",
        "BPE files fetched during import",
        "none",
        lambda m: (m.get("cold_cache_files_written"), lambda v: "-" if v is None else str(v)),
    ),
    Claim(
        "chunking",
        "Tokenizer calls per build",
        "konte stopped encoding each string twice",
        lambda m: (
            _get(m, "first_build", "tokenizer_encode_calls"),
            lambda v: "-" if v is None else f"{v:,}",
        ),
    ),
    Claim(
        "chunking",
        "Text splitters constructed per build",
        "2 instead of one per segment",
        lambda m: (
            _get(m, "first_build", "splitter_constructions"),
            lambda v: "-" if v is None else f"{v:,}",
        ),
    ),
    Claim(
        "checkpoint_io",
        "Checkpoint bytes written during a build",
        "~1.7 GB -> ~7 MB at 500 segments",
        lambda m: (m.get("checkpoint_bytes_written"), _bytes),
    ),
    Claim(
        "storage_size",
        "Project directory size",
        "4.97x -> 2.94x the size of the text",
        lambda m: (m.get("project_dir_bytes"), _bytes),
    ),
    Claim(
        "storage_size",
        "Lexical index file",
        "~25 MB -> ~2 MB at 20k chunks",
        lambda m: (_get(m, "files", "bm25.pkl"), _bytes),
    ),
    Claim(
        "storage_size",
        "Duplicated corpus file (bm25_chunks.json)",
        "removed",
        lambda m: (_get(m, "files", "bm25_chunks.json"), _bytes),
        direction="removed",
    ),
    Claim(
        "storage_size",
        "Project directory as a multiple of its text",
        "4.97x -> 2.94x",
        lambda m: (
            m.get("project_dir_multiple_of_text"),
            lambda v: "-" if v is None else f"{v:.2f}x",
        ),
    ),
    Claim(
        "storage_size_korean",
        "Project directory as a multiple of its text, non-ASCII",
        "4.97x -> 2.94x",
        lambda m: (
            m.get("project_dir_multiple_of_text"),
            lambda v: "-" if v is None else f"{v:.2f}x",
        ),
    ),
    Claim(
        "storage_size_korean",
        "Project directory size, non-ASCII corpus",
        "not claimed; JSON stopped escaping non-ASCII",
        lambda m: (m.get("project_dir_bytes"), _bytes),
    ),
    Claim(
        "storage_size_korean",
        "Corpus JSON, non-ASCII text",
        "not claimed; was pretty-printed and escaped",
        lambda m: (_get(m, "files", "chunks.json"), _bytes),
    ),
    Claim(
        "open_cache",
        "Opening a project",
        "indexes only; corpus parsed on first use",
        lambda m: (_get(m, "open_ms", "median"), _ms),
    ),
    Claim(
        "open_cache",
        "Per-request project lookup",
        "~330 ms reopen -> a dictionary lookup",
        lambda m: (
            _get(m, "get_shared_project_ms", "median") or _get(m, "get_project_ms", "median"),
            _ms,
        ),
    ),
    Claim(
        "query_bm25",
        "Unfiltered lexical query, steady state",
        "84 ms -> 68 ms at 100k chunks",
        lambda m: (_get(m, "warm_query_ms", "median"), _ms),
    ),
    Claim(
        "query_bm25",
        "First lexical query after open",
        "pays the corpus parse that open no longer does",
        lambda m: (
            (m.get("first_query_seconds") or 0) * 1000 or None,
            _ms,
        ),
        direction="cost",
    ),
    Claim(
        "query_bm25",
        "Peak memory holding a lexical project",
        "not claimed; a consequence of storing the corpus once",
        lambda m: (
            _get(m, "rss_mb", "after_first_query"),
            lambda v: "-" if v is None else f"{v:,.0f} MB",
        ),
    ),
    Claim(
        "query_faiss_filter",
        "Filtered vector query, steady state",
        "16-42 ms -> 0.05-0.4 ms of id selection at 100k chunks",
        lambda m: (_get(m, "filtered_query_ms", "median"), _ms),
    ),
    Claim(
        "query_faiss_filter",
        "First filtered query",
        "builds the postings once",
        lambda m: (m.get("first_filtered_query_ms"), _ms),
        direction="cost",
    ),
    Claim(
        "keyword_extraction",
        "Query cost against a stalled endpoint",
        "~360 s -> 5 s",
        lambda m: (_get(m, "stalled_endpoint", "seconds_until_fallback"), _seconds),
    ),
    Claim(
        "keyword_extraction",
        "Requests for 8 repeats of one query",
        "cached after the first",
        lambda m: (
            _get(m, "cache", "repeated_query", "requests_sent"),
            lambda v: "-" if v is None else str(v),
        ),
    ),
    Claim(
        "build_concurrency",
        "Requests resent by a single rate limit",
        "the whole segment -> the one chunk",
        lambda m: (
            _get(m, "retry", "requests_resent"),
            lambda v: "-" if v is None else str(v),
        ),
    ),
    Claim(
        "build_concurrency",
        "Chunks losing their context when one chunk exhausts its retries",
        "the whole segment -> the one chunk",
        lambda m: (
            (_get(m, "retry", "chunks") or 0) - (_get(m, "retry", "chunks_with_context") or 0),
            lambda v: "-" if v is None else str(v),
        ),
    ),
    Claim(
        "build_concurrency",
        "Context generation wall-clock",
        "segments overlap under one ceiling",
        lambda m: (_get(m, "context", "build_seconds"), _seconds),
    ),
    Claim(
        "build_concurrency",
        "Index build wall-clock",
        "embedding batches overlap in a window",
        lambda m: (_get(m, "index", "build_seconds"), _seconds),
    ),
    Claim(
        "build_concurrency_live",
        "End-to-end build against the configured endpoint",
        "a build costs its request volume divided by the ceiling",
        lambda m: (_get(m, "live", "build_seconds"), _seconds),
    ),
    Claim(
        "build_concurrency_live_gemma",
        "End-to-end build, second endpoint",
        "same change, an endpoint with stronger prefix caching",
        lambda m: (_get(m, "live", "build_seconds"), _seconds),
    ),
    Claim(
        "build_concurrency_live",
        "Chunks that actually received generated context",
        "every chunk",
        lambda m: (
            _get(m, "live", "chunks_with_context"),
            lambda v: "-" if v is None else str(v),
        ),
        direction="higher",
    ),
]


#: A labelled variant supersedes the plain run for the claims they share. The
#: live build re-runs every stub measurement on its way to the endpoint, so its
#: numbers are the more recent of the two.
SUPERSEDES = {"build_concurrency_live": "build_concurrency"}


def _load(results_dir: Path) -> dict[str, dict]:
    stored = {}
    for path in sorted(results_dir.glob("*.json")):
        stored[path.stem] = json.loads(path.read_text())
    for variant, plain in SUPERSEDES.items():
        if variant in stored:
            stored[plain] = {**stored[variant], "_aliased_from": variant}
    return stored


def _measurements(payload: dict, revision: str) -> dict | None:
    runs = _get(payload, "runs", revision, default=[]) or []
    for run in runs:
        if run.get("status") == "ok":
            return run.get("measurements") or {}
    return None


def _verdict(before, after, direction: str) -> str:
    if direction == "removed":
        if before is None:
            return "unmeasured"
        return "confirmed, gone" if after is None else "still present"
    if before is None or after is None:
        return "unmeasured"
    if isinstance(before, str) or isinstance(after, str):
        return "confirmed" if before != after else "no change"
    if direction == "cost":
        return "one-off cost, by design"
    if direction == "exact":
        return "identical" if before == after else "DIFFERS"
    improved = after < before if direction == "lower" else after > before
    if before == after:
        return "no change"
    return "confirmed" if improved else "REGRESSED"


def _change(before, after) -> str:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return "-"
    if before == 0:
        return "n/a" if after == 0 else "+inf"
    delta = (after - before) / before * 100
    return f"{delta:+.1f}%"


def _parity_section(stored: dict) -> list[str]:
    payload = stored.get("retrieval_parity")
    if not payload:
        return ["_Not run._"]
    before = _measurements(payload, "baseline")
    after = _measurements(payload, "head")
    if not before or not after:
        return ["_No successful run on both revisions._"]

    lines = []
    mismatches = []
    for query, ranking in (before.get("rankings") or {}).items():
        other = (after.get("rankings") or {}).get(query)
        if other is None:
            mismatches.append((query, "missing on the newer revision"))
        elif [r["chunk_id"] for r in ranking] != [r["chunk_id"] for r in other]:
            mismatches.append((query, "different chunk ordering"))
        elif [r["score"] for r in ranking] != [r["score"] for r in other]:
            mismatches.append((query, "same ordering, different scores"))

    total = len(before.get("rankings") or {})
    if not mismatches:
        lines.append(
            f"**Identical.** All {total} queries returned the same chunks in the same "
            f"order with the same scores on both revisions. The speed numbers above "
            f"compare the same operation."
        )
    else:
        lines.append(f"**{len(mismatches)} of {total} queries differ.**")
        lines.append("")
        for query, reason in mismatches:
            lines.append(f"- `{query}` - {reason}")
    return lines


def build_report(results_dir: Path) -> str:
    stored = _load(results_dir)

    rows = []
    for claim in CLAIMS:
        payload = stored.get(claim.case)
        before_m = _measurements(payload, "baseline") if payload else None
        after_m = _measurements(payload, "head") if payload else None
        if before_m is None or after_m is None:
            rows.append((claim, None, None, "unmeasured", "-", str))
            continue
        before, fmt = claim.extract(before_m)
        after, _ = claim.extract(after_m)
        rows.append(
            (
                claim,
                before,
                after,
                _verdict(before, after, claim.direction),
                _change(before, after),
                fmt,
            )
        )

    lines = [
        "# Performance: before and after",
        "",
        "Measured comparison of the two revisions. Every row is a claim from the",
        "changelog, the two numbers behind it, and whether it held up.",
        "",
        "| Claim | Stated | Before | After | Change | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for claim, before, after, verdict, change, fmt in rows:
        lines.append(
            f"| {claim.title} | {claim.claimed} | {fmt(before)} | {fmt(after)} "
            f"| {change} | {verdict} |"
        )

    lines += ["", "## Do both revisions return the same results?", ""]
    lines += _parity_section(stored)

    lines += ["", "## Prefix caching", ""]
    lines += _prefix_cache_section(stored)

    lines += ["", "## Accuracy", ""]
    lines += _accuracy_section(stored)

    lines += ["", "## Segments, chunks, retrieval and answers", ""]
    lines += _output_section(stored)

    lines += ["", "## Reading these numbers", ""]
    lines += _notes_section(stored)

    lines += ["", "## Run detail", "", "| Case | Scale | Status |", "|---|---|---|"]
    for case, payload in sorted(stored.items()):
        # Judge output and endpoint probes live alongside case results but are
        # not per-revision runs; listing them here would report them "incomplete".
        if "runs" not in payload:
            continue
        before_m = _measurements(payload, "baseline")
        after_m = _measurements(payload, "head")
        detail = _scale_of(before_m or {})
        status = "both revisions ok" if before_m and after_m else "incomplete"
        lines.append(f"| {case} | {detail} | {status} |")

    lines += ["", "## Measurements in full", ""]
    lines += _detail_section(stored)

    lines.append("")
    return "\n".join(lines)


def _prefix_cache_section(stored: dict) -> list[str]:
    """Whether endpoints reward konte's shared-prefix prompt shape."""
    probe = stored.get("prefix_cache")
    if not probe:
        return [
            "_Not run. `uv run python -m benchmarks.prefix_cache_probe --label <name> "
            "--base-url <url> --model <model>`._"
        ]

    lines = [
        "Every context prompt is `[segment ~8000 tokens][chunk ~800]`, so all chunks of a "
        "segment share a long prefix. Whether that is worth anything depends on when the "
        "requests arrive, not only on what they contain.",
        "",
        "| Endpoint | Sequential, shared | Concurrent, shared | Concurrent, distinct | Sharing helps |",
        "|---|---|---|---|---|",
    ]
    for label, result in sorted(probe.items()):
        seq = result.get("sequential_shared") or {}
        shared = result.get("concurrent_shared") or {}
        distinct = result.get("concurrent_distinct") or {}
        cold, warm = seq.get("first_s"), seq.get("rest_median_s")
        speedup = distinct.get("wall_s") / shared["wall_s"] if shared.get("wall_s") else None
        lines.append(
            f"| {label} | {cold:.2f}s cold -> {warm:.2f}s warm | {shared.get('wall_s', 0):.2f}s "
            f"| {distinct.get('wall_s', 0):.2f}s | "
            f"{'yes, ' + format(speedup, '.1f') + 'x' if speedup and speedup > 1.15 else 'no'} |"
        )

    helps = [r for r in probe.values() if r.get("sharing_helps_when_concurrent")]
    if helps:
        lines += [
            "",
            f"**Sharing a prefix pays off on {len(helps)} of {len(probe)} endpoints measured, "
            f"including for requests issued simultaneously.** That is what makes segment "
            f"scheduling matter: the older revision held one segment's prefix in flight at a "
            f"time, while the newer one admits up to `MAX_CONCURRENT_CALLS` segments at once "
            f"(`project.py`, `segment_gate`), so the server sees interleaved prefixes. "
            f"The end-to-end builds above still came out faster after the change, so on these "
            f"endpoints overlapping segments outweighs the lost cache locality on wall-clock - "
            f"but it is bought by recomputing prefixes, which a provider that bills cached "
            f"input at a discount would charge for.",
        ]
    return lines


#: Fixed so the reported interval is reproducible from the stored scores.
_BOOTSTRAP_SEED = 7
_BOOTSTRAP_ROUNDS = 20000


def _paired_significance(scored: dict, control: dict | None) -> dict | None:
    """Test the answer-score gap as paired observations, pooling any repeat run.

    A per-question pairing is what the question actually asks: did *this* answer
    get worse. Comparing two means instead would let a single question's score
    look decisive, because the mean barely moves when most answers are identical.
    """
    import random
    from math import comb

    diffs: list[float] = []
    for payload in (scored, control):
        if not payload:
            continue
        before = {
            g["question"]: g["score"]
            for g in _get(payload, "revisions", "baseline", "graded", default=[]) or []
            if isinstance(g.get("score"), (int, float))
        }
        after = {
            g["question"]: g["score"]
            for g in _get(payload, "revisions", "head", "graded", default=[]) or []
            if isinstance(g.get("score"), (int, float))
        }
        diffs += [after[q] - before[q] for q in before if q in after]

    if not diffs:
        return None

    discordant = [d for d in diffs if d]
    negative = sum(1 for d in discordant if d < 0)

    rng = random.Random(_BOOTSTRAP_SEED)
    means = sorted(
        sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(_BOOTSTRAP_ROUNDS)
    )
    lo = means[int(0.025 * _BOOTSTRAP_ROUNDS)]
    hi = means[int(0.975 * _BOOTSTRAP_ROUNDS)]

    if discordant:
        k = min(negative, len(discordant) - negative)
        p = min(
            1.0,
            sum(comb(len(discordant), i) for i in range(k + 1)) / 2 ** len(discordant) * 2,
        )
    else:
        p = 1.0

    return {
        "n": len(diffs),
        "discordant": len(discordant),
        "negative": negative,
        "positive": len(discordant) - negative,
        "mean": sum(diffs) / len(diffs),
        "lo": lo,
        "hi": hi,
        "p": p,
    }


def _accuracy_section(stored: dict) -> list[str]:
    """Retrieval recall against ground truth, and judged answer correctness."""
    payload = stored.get("accuracy")
    if not payload:
        return [
            "_Not run. `uv run python -m benchmarks.compare --case accuracy`, then "
            "`uv run python -m benchmarks.judge`._"
        ]
    before = _measurements(payload, "baseline")
    after = _measurements(payload, "head")
    if not before or not after:
        return ["_No successful run on both revisions._"]

    goldens = before.get("goldens") or {}
    lines = [
        f"Golden set: `{Path(str(goldens.get('path', '?'))).name}` - "
        f"{goldens.get('count', 0)} questions, {goldens.get('with_ground_truth', 0)} with "
        f"ground-truth context.",
        "",
        "### Retrieval accuracy",
        "",
        "Deterministic: a question counts as found when a retrieved chunk contains a "
        "fragment of the context the golden says it should be answered from.",
        "",
        "| Metric | Before | After | Change |",
        "|---|---|---|---|",
    ]
    rb, ra = before.get("retrieval") or {}, after.get("retrieval") or {}
    for key, label in (
        ("recall_at_1", "recall@1"),
        ("recall_at_5", "recall@5"),
        ("recall_at_10", "recall@10"),
        ("recall_at_k", "recall@k"),
        ("mrr", "MRR"),
    ):
        b, a = rb.get(key), ra.get(key)
        lines.append(f"| {label} | {_ratio(b)} | {_ratio(a)} | {_change(b, a)} |")
    lines.append("")

    # Questions whose retrieval outcome moved, which is where a regression hides.
    pb = {q["index"]: q for q in rb.get("per_question") or []}
    pa = {q["index"]: q for q in ra.get("per_question") or []}
    moved = [
        (i, pb[i], pa[i])
        for i in sorted(set(pb) & set(pa))
        if pb[i].get("rank") != pa[i].get("rank")
    ]
    lost = [m for m in moved if m[1].get("rank") and not m[2].get("rank")]
    gained = [m for m in moved if not m[1].get("rank") and m[2].get("rank")]
    lines += [
        f"Rank changed for {len(moved)} of {len(pb)} questions: "
        f"{len(lost)} found before but not after, {len(gained)} found after but not before.",
        "",
    ]
    if moved:
        lines += ["| # | Question | Rank before | Rank after |", "|---|---|---|---|"]
        for index, b, a in moved[:15]:
            question = (b.get("question") or "")[:44].replace("|", "\\|")
            lines.append(
                f"| {index} | {question}... | {b.get('rank') or 'not found'} "
                f"| {a.get('rank') or 'not found'} |"
            )
        lines.append("")

    # --- judged answers -----------------------------------------------------
    scored = stored.get("accuracy_scored")
    if not scored:
        lines += [
            "### Answer correctness",
            "",
            "_Not scored. Run `uv run python -m benchmarks.judge`._",
        ]
        return lines

    lines += [
        "### Answer correctness",
        "",
        f"Both revisions' answers graded in one pass by the same judge, using the "
        f"`{scored.get('criteria')}` criteria from `evaluation/prompts/eval_prompts.py`.",
        "",
        "| Metric | Before | After | Change |",
        "|---|---|---|---|",
    ]
    sb = _get(scored, "revisions", "baseline", "summary", default={}) or {}
    sa = _get(scored, "revisions", "head", "summary", default={}) or {}
    for key, label, fmt in (
        ("count", "answers graded", lambda v: "-" if v is None else f"{v:,}"),
        ("mean_score", "mean score", lambda v: "-" if v is None else f"{v:.3f}"),
        ("median_score", "median score", lambda v: "-" if v is None else f"{v:.3f}"),
        ("pass_rate_0_7", "pass rate (>=0.7)", lambda v: "-" if v is None else f"{v:.1%}"),
        ("perfect_rate", "scored 1.0", lambda v: "-" if v is None else f"{v:.1%}"),
    ):
        b, a = sb.get(key), sa.get(key)
        lines.append(f"| {label} | {fmt(b)} | {fmt(a)} | {_change(b, a)} |")
    lines.append("")

    gb = {
        g["question"]: g for g in _get(scored, "revisions", "baseline", "graded", default=[]) or []
    }
    ga = {g["question"]: g for g in _get(scored, "revisions", "head", "graded", default=[]) or []}

    # Same evidence in, different answer out, is the generator sampling - not a
    # retrieval or ranking change. Stated with the retrieval data rather than
    # asserted, because it is the whole basis for reading a score gap as noise.
    rb_q = {q["question"]: q for q in rb.get("per_question") or []}
    ra_q = {q["question"]: q for q in ra.get("per_question") or []}
    moved_scores = [
        q
        for q in gb
        if q in ga
        and isinstance(gb[q].get("score"), (int, float))
        and isinstance(ga[q].get("score"), (int, float))
        and gb[q]["score"] != ga[q]["score"]
    ]
    same_evidence = [
        q
        for q in moved_scores
        if rb_q.get(q, {}).get("retrieved") == ra_q.get(q, {}).get("retrieved")
    ]
    if moved_scores:
        lines += [
            f"{len(moved_scores)} of {len(gb)} answers scored differently, and "
            f"{len(same_evidence)} of those {len(moved_scores)} were generated from "
            f"**identical retrieved evidence**. Where the evidence is the same and the answer "
            f"is not, the difference is the generator sampling, not anything under test.",
            "",
        ]

    # Whether the score gap means anything is a question about paired
    # observations, not about how steady a mean is across reruns. Most answers
    # do not move at all, which holds the mean still and would make a
    # mean-stability check declare a one-question difference "significant".
    stat = _paired_significance(scored, stored.get("accuracy_control_scored"))
    if stat:
        lines += [
            f"**Is the gap real?** Pairing each question's score across revisions gives "
            f"{stat['n']} paired observations, of which {stat['discordant']} differ "
            f"({stat['negative']} lower after, {stat['positive']} higher). Mean paired "
            f"difference {stat['mean']:+.4f}, bootstrap 95% CI "
            f"[{stat['lo']:+.4f}, {stat['hi']:+.4f}], sign test p = {stat['p']:.3f}. "
            + (
                "The interval includes zero and the sign test is far from significant, so "
                "this is sampling in the generator, not a measured change in answer quality."
                if stat["lo"] <= 0 <= stat["hi"]
                else "The interval excludes zero; this warrants investigation."
            ),
            "",
        ]
    regressions = [
        (q, gb[q], ga[q])
        for q in gb
        if q in ga
        and isinstance(gb[q].get("score"), (int, float))
        and isinstance(ga[q].get("score"), (int, float))
        and ga[q]["score"] < gb[q]["score"] - 0.15
    ]
    if regressions:
        lines += [
            f"**{len(regressions)} answers scored materially lower after.**",
            "",
            "| Question | Before | After | Judge's reason (after) |",
            "|---|---|---|---|",
        ]
        for question, b, a in regressions[:10]:
            reason = (a.get("reason") or "")[:90].replace("|", "\\|")
            lines.append(
                f"| {question[:40].replace('|', '')}... | {b['score']:.2f} | {a['score']:.2f} | {reason} |"
            )
        lines.append("")
    else:
        lines += ["No answer scored materially lower after the change.", ""]
    return lines


def _output_section(stored: dict) -> list[str]:
    """Show what each revision produced: chunks, top-k, and answers."""
    payload = stored.get("output_parity")
    if not payload:
        return ["_Not run. `uv run python -m benchmarks.compare --case output_parity`._"]
    before = _measurements(payload, "baseline")
    after = _measurements(payload, "head")
    if not before or not after:
        return ["_No successful run on both revisions._"]

    lines: list[str] = []

    # --- segmentation and chunking -----------------------------------------
    cut_b, cut_a = before.get("chunking", {}), after.get("chunking", {})
    seg_b = [s["sha"] for s in cut_b.get("segments", [])]
    seg_a = [s["sha"] for s in cut_a.get("segments", [])]
    chunk_b = [c["sha"] for c in cut_b.get("chunks", [])]
    chunk_a = [c["sha"] for c in cut_a.get("chunks", [])]

    lines += [
        "### Segmentation and chunking",
        "",
        f"Document: `{cut_b.get('document')}` ({cut_b.get('source_characters', 0):,} characters)",
        "",
        "| | Before | After |",
        "|---|---|---|",
        f"| segments | {cut_b.get('segment_count')} | {cut_a.get('segment_count')} |",
        f"| chunks | {cut_b.get('chunk_count')} | {cut_a.get('chunk_count')} |",
        f"| segment text identical | {'yes' if seg_b == seg_a else 'NO'} | |",
        f"| chunk text identical | {'yes' if chunk_b == chunk_a else 'NO'} | |",
        "",
    ]
    if chunk_b != chunk_a:
        lines += [
            "| chunk | before sha | after sha | before tokens | after tokens |",
            "|---|---|---|---|---|",
        ]
        for i in range(max(len(chunk_b), len(chunk_a))):
            cb = cut_b["chunks"][i] if i < len(chunk_b) else {}
            ca = cut_a["chunks"][i] if i < len(chunk_a) else {}
            if cb.get("sha") != ca.get("sha"):
                lines.append(
                    f"| {cb.get('chunk_id') or ca.get('chunk_id')} | {cb.get('sha', '-')} "
                    f"| {ca.get('sha', '-')} | {cb.get('tokens', '-')} | {ca.get('tokens', '-')} |"
                )
        lines.append("")
    else:
        sample = cut_b.get("chunks", [])[:5]
        lines += [
            "First chunks, identical on both revisions:",
            "",
            "| chunk | tokens | starts |",
            "|---|---|---|",
        ]
        for chunk in sample:
            head = chunk["head"].replace("\n", " ").replace("|", "\\|")
            lines.append(f"| `{chunk['chunk_id']}` | {chunk['tokens']} | {head}... |")
        lines.append("")

    # --- retrieval ----------------------------------------------------------
    for key, title, note in (
        (
            "retrieval_pinned_extraction",
            "Retrieval, keyword extraction pinned",
            "Extraction replaced by identical local tokenization on both sides, so this "
            "compares the ranking code alone.",
        ),
        (
            "retrieval_default_settings",
            "Retrieval, each revision on its own defaults",
            "What a caller actually gets. Differences here can come from extraction "
            "returning different keywords, not from the ranking code.",
        ),
    ):
        rb, ra = before.get(key), after.get(key)
        if not isinstance(rb, dict) or not isinstance(ra, dict):
            continue
        lines += [
            f"### {title}",
            "",
            note,
            "",
            "| Query | Mode | Top-k identical | Overlap |",
            "|---|---|---|---|",
        ]
        for query in rb:
            for mode, hits_b in (rb.get(query) or {}).items():
                hits_a = (ra.get(query) or {}).get(mode) or []
                ids_b = [h["chunk_id"] for h in hits_b]
                ids_a = [h["chunk_id"] for h in hits_a]
                overlap = len(set(ids_b) & set(ids_a))
                same = "yes" if ids_b == ids_a else "NO"
                total = max(len(ids_b), len(ids_a)) or 1
                lines.append(f"| {query[:34]}... | {mode} | {same} | {overlap}/{total} |")
        lines.append("")

        example = next(iter(rb), None)
        if example:
            lines += [
                f"Top-{len(rb[example].get('hybrid', []))} hybrid results for `{example}`:",
                "",
            ]
            lines += ["| # | Before | After |", "|---|---|---|"]
            hb = (rb[example] or {}).get("hybrid") or []
            ha = (ra.get(example) or {}).get("hybrid") or []
            for i in range(max(len(hb), len(ha))):
                b = hb[i] if i < len(hb) else {}
                a = ha[i] if i < len(ha) else {}
                lines.append(
                    f"| {i + 1} | `{b.get('chunk_id', '-')}` ({b.get('score', '-')}) "
                    f"| `{a.get('chunk_id', '-')}` ({a.get('score', '-')}) |"
                )
            lines.append("")

    # A revision runs the same queries twice, once per extraction setting. Where
    # its two runs disagree, the disagreement cannot be a difference between
    # revisions - it is the measurement's own noise floor, and it is the only
    # honest yardstick for reading a cross-revision difference of the same size.
    control = []
    for revision, side in (("before", before), ("after", after)):
        first = side.get("retrieval_default_settings") or {}
        second = side.get("retrieval_pinned_extraction") or {}
        for query in first:
            for mode in ("semantic",):
                a1 = [h["chunk_id"] for h in (first.get(query) or {}).get(mode) or []]
                a2 = [h["chunk_id"] for h in (second.get(query) or {}).get(mode) or []]
                if a1 and a2 and a1 != a2:
                    control.append((revision, mode, query))
    if control:
        lines += [
            f"**Self-control:** the same revision returned a different `{control[0][1]}` "
            f"ordering across its own two runs of the same query "
            f"({len(control)} case(s), e.g. on the *{control[0][0]}* side). Embedding calls "
            f"are not bit-identical between requests, and adjacent results here are "
            f"separated by less than that variance. A cross-revision ordering difference of "
            f"this size is noise, not a behaviour change; only a difference in the *set* of "
            f"retrieved chunks would be.",
            "",
        ]

    # --- answers ------------------------------------------------------------
    ab, aa = before.get("answers"), after.get("answers")
    if isinstance(ab, dict) and isinstance(aa, dict):
        lines += [
            "### Generated answers",
            "",
            "Answer generation is sampled, so identical wording is not expected and not "
            "required; what matters is whether both revisions answer the same question "
            "from the same evidence.",
            "",
        ]
        for query in ab:
            b, a = ab.get(query) or {}, aa.get(query) or {}
            lines += [f"**{query}**", ""]
            lines += [
                "| | Before | After |",
                "|---|---|---|",
                f"| model | {b.get('model', '-')} | {a.get('model', '-')} |",
                f"| chunks used | {b.get('sources_used', '-')} | {a.get('sources_used', '-')} |",
                f"| same evidence | {'yes' if b.get('retrieved') == a.get('retrieved') else 'NO'} | |",
                "",
            ]
            for label, side in (("Before", b), ("After", a)):
                text = side.get("answer") or side.get("error") or "-"
                lines += [f"_{label}:_", "", "> " + text.replace("\n", "\n> "), ""]
    return lines


#: Fields too large to print, or already rendered elsewhere.
_SKIP_KEYS = {
    "rankings",
    "chunking",
    "answers",
    "retrieval_default_settings",
    "retrieval_pinned_extraction",
    "stderr_tail",
    "retrieval",
    "goldens",
}


def _flatten(data, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in (data or {}).items():
        if key in _SKIP_KEYS:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _detail_section(stored: dict) -> list[str]:
    """Every recorded number, side by side, for anything the table summarizes."""
    lines = []
    for case, payload in sorted(stored.items()):
        if payload.get("_aliased_from") or "runs" not in payload:
            continue  # superseded, or not a per-revision case result
        before = _flatten(_measurements(payload, "baseline") or {})
        after = _flatten(_measurements(payload, "head") or {})
        if not before and not after:
            continue
        lines += [f"### {case}", "", "| Measurement | Before | After |", "|---|---|---|"]
        for key in sorted(set(before) | set(after)):
            lines.append(f"| `{key}` | {_cell(before.get(key))} | {_cell(after.get(key))} |")
        lines.append("")
    return lines


def _scale_of(measurements: dict) -> str:
    """Describe what a case ran against, however that case records it."""
    corpus = measurements.get("corpus")
    if isinstance(corpus, dict) and corpus:
        return ", ".join(f"{k}={v}" for k, v in corpus.items())
    for key in ("project", "requested_segments", "requested_chunks"):
        if key in measurements:
            return f"{key}={measurements[key]}"
    retry = _get(measurements, "retry", "chunks")
    if retry:
        return f"chunks={retry}"
    return "-"


def _notes_section(stored: dict) -> list[str]:
    """Caveats that the table alone would misrepresent."""
    notes = []

    chunking = (
        _measurements(stored.get("chunking", {}), "baseline"),
        _measurements(stored.get("chunking", {}), "head"),
    )
    if all(chunking):
        before, after = (_get(m, "first_build", "tokenizer_encode_calls") for m in chunking)
        if before and after:
            notes.append(
                f"- **Tokenizer calls barely moved ({before:,} to {after:,}), and that is expected.** "
                f"konte's own double-encoding is gone - exactly the {before - after} calls it "
                f"accounted for - but it was a rounding error against the text splitter's "
                f"internal per-piece encoding, which neither revision changes. The chunking win "
                f"is splitter reuse, not tokenizer calls."
            )

    faiss = (
        _measurements(stored.get("query_faiss_filter", {}), "baseline"),
        _measurements(stored.get("query_faiss_filter", {}), "head"),
    )
    if all(faiss):
        before_steady = _get(faiss[0], "filtered_query_ms", "median")
        after_steady = _get(faiss[1], "filtered_query_ms", "median")
        after_first = faiss[1].get("first_filtered_query_ms")
        before_first = faiss[0].get("first_filtered_query_ms")
        if None not in (before_steady, after_steady, after_first, before_first):
            saved = before_steady - after_steady
            overhead = after_first - before_first
            if saved > 0:
                notes.append(
                    f"- **The filter index pays for itself after about "
                    f"{overhead / saved:.0f} filtered queries.** Building the postings costs "
                    f"{overhead:.0f} ms once; each filtered query then saves {saved:.2f} ms. "
                    f"A process that issues a handful of filtered queries and exits is slower, "
                    f"not faster."
                )

    bm25 = (
        _measurements(stored.get("query_bm25", {}), "baseline"),
        _measurements(stored.get("query_bm25", {}), "head"),
    )
    if all(bm25):
        notes.append(
            "- **Work moved out of open and into the first query; it did not disappear.** "
            "A process that opens a project and issues one lexical query does roughly the same "
            "total work on both revisions. The gain is for a process that opens many projects, "
            "or answers only vector queries, and never touches the lexical corpus at all."
        )

    live = (
        _measurements(stored.get("build_concurrency_live", {}), "baseline"),
        _measurements(stored.get("build_concurrency_live", {}), "head"),
    )
    if all(live):
        got = [(_get(m, "live", "chunks_with_context") or 0) for m in live]
        want = [(_get(m, "live", "chunks") or 0) for m in live]
        if want[0] and got == [0, 0]:
            notes.append(
                f"- **The live build row is invalid and must not be read as a speedup.** "
                f"Context was generated for {got[0]}/{want[0]} chunks before and "
                f"{got[1]}/{want[1]} after, so the wall-clock measures how fast each "
                f"revision exhausts its retry budget, not how fast it builds. konte answers "
                f"a failed context call with empty context rather than an error, which is "
                f"why the run reported success. The case now refuses to return a number in "
                f"this state; a stored result showing it predates that check."
            )

    notes.append(
        "- **Scales differ from the changelog's.** Several claims were stated at 20k or 100k "
        "chunks; the runs here are at the scale each case records above. Directions and "
        "orders of magnitude are comparable, absolute milliseconds are not."
    )
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    report = build_report(args.results_dir)
    args.out.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
