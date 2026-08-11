"""Did retrieval and answer accuracy hold up?

Speed that costs correctness is not an improvement. This case replays a golden
question set against both revisions and records two things:

``retrieval``
    Whether the chunks a question is supposed to be answered from actually come
    back, and where they rank. The goldens carry their own ground-truth
    ``retrieval_context``, so this needs no judge and no sampling - it is a
    deterministic recall measurement.

``answers``
    The generated answer for each question, stored but not scored here. Scoring
    happens once, afterwards, in ``benchmarks/judge.py``, so that both
    revisions are marked by the same judge in the same pass. A judge that ran
    inside each revision's own process would be a second variable.

The golden sets live outside this repository. Point ``KONTE_BENCH_GOLDENS`` at a
file, or pass ``goldens=<path>``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from benchmarks.build import open_project
from benchmarks.corpus import available_real_projects, real_project
from benchmarks.harness import Context

DEFAULT_PROJECT = "wco_korean_feb2026"
DEFAULT_GOLDENS = Path(
    os.environ.get(
        "KONTE_BENCH_GOLDENS",
        Path.home()
        / "Documents/konte-research-archive/evaluation/data/synthetic"
        / "deepeval_goldens_korean_no_hypothetical.json",
    )
)
TOP_K = 20
#: Characters of a ground-truth context used as its fingerprint. Long enough to
#: be unique in the corpus, short enough to survive the context prefix that the
#: golden carries but a raw chunk does not.
PROBE_CHARS = 60


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _probes(golden: dict) -> list[str]:
    """Distinctive fragments of each ground-truth context for this question."""
    probes = []
    for context in golden.get("retrieval_context") or []:
        normalized = _normalize(context)
        # Skip the generated context prefix where present: it describes the
        # chunk rather than quoting it, and only the chunk text is indexed
        # verbatim on both revisions.
        tail = normalized[-PROBE_CHARS * 3 :]
        if len(tail) >= PROBE_CHARS:
            probes.append(tail[:PROBE_CHARS])
    return probes


def _rank_of_hit(results, probes: list[str]) -> int | None:
    """1-based rank of the first result matching any ground-truth fragment."""
    for rank, result in enumerate(results, start=1):
        haystack = _normalize(result.content)
        if any(probe in haystack for probe in probes):
            return rank
    return None


def _load_goldens(ctx: Context) -> tuple[list[dict], Path]:
    path = Path(ctx.options.get("goldens", DEFAULT_GOLDENS)).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"golden set not found at {path}; set KONTE_BENCH_GOLDENS or pass goldens=<path>"
        )
    data = json.loads(path.read_text())
    goldens = data if isinstance(data, list) else data.get("goldens") or data.get("test_cases")
    limit = int(ctx.options.get("limit", 0))
    return (goldens[:limit] if limit else goldens), path


def run(ctx: Context) -> dict:
    name = ctx.options.get("project", DEFAULT_PROJECT)
    if name not in available_real_projects():
        return {"status": "skipped", "reason": f"no built project {name!r}"}

    goldens, path = _load_goldens(ctx)
    mode = ctx.options.get("mode", "hybrid")
    top_k = int(ctx.options.get("top_k", TOP_K))
    answer_limit = int(ctx.options.get("answers", 20))

    storage, project_name = real_project(name, ctx.revision, reindex_lexical=True)
    project = open_project(project_name, storage)

    per_question = []
    for index, golden in enumerate(goldens):
        question = golden.get("input") or ""
        probes = _probes(golden)
        response = project.query(question, mode=mode, top_k=top_k)
        rank = _rank_of_hit(response.results, probes) if probes else None
        per_question.append(
            {
                "index": index,
                "question": question,
                "has_ground_truth": bool(probes),
                "rank": rank,
                "top_score": response.results[0].score if response.results else None,
                "retrieved": [result.chunk_id for result in response.results[:5]],
            }
        )

    scored = [q for q in per_question if q["has_ground_truth"]]
    found = [q for q in scored if q["rank"]]

    def recall_at(k: int) -> float | None:
        if not scored:
            return None
        return sum(1 for q in scored if q["rank"] and q["rank"] <= k) / len(scored)

    answers = []
    for golden in goldens[:answer_limit]:
        question = golden.get("input") or ""
        try:
            _, generated = asyncio.run(project.query_with_answer(question, mode=mode, max_chunks=5))
            answers.append(
                {
                    "question": question,
                    "expected_output": golden.get("expected_output"),
                    "actual_output": generated.answer,
                    "model": generated.model,
                    "sources_used": generated.sources_used,
                }
            )
        except Exception as error:
            answers.append(
                {
                    "question": question,
                    "expected_output": golden.get("expected_output"),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return {
        "goldens": {"path": str(path), "count": len(goldens), "with_ground_truth": len(scored)},
        "retrieval": {
            "mode": mode,
            "top_k": top_k,
            "recall_at_1": recall_at(1),
            "recall_at_5": recall_at(5),
            "recall_at_10": recall_at(10),
            "recall_at_k": recall_at(top_k),
            "mrr": (sum(1 / q["rank"] for q in found) / len(scored)) if scored else None,
            "median_rank_when_found": (
                sorted(q["rank"] for q in found)[len(found) // 2] if found else None
            ),
            "per_question": per_question,
        },
        "answers": answers,
    }
