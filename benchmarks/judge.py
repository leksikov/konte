"""Score both revisions' answers with one judge, in one pass.

Answer correctness is graded by an LLM, which makes the judge itself a variable.
Running it inside each revision's process would mean two judges, two client
configurations, and two sampling draws - so a scoring difference could not be
attributed to the answers. Instead the cases only *record* answers, and this
module grades both sets afterwards, in a single process, with one client.

Criteria come from ``evaluation/prompts/eval_prompts.py`` so the bar is the same
one the project already evaluates against.

    uv run python -m benchmarks.judge                    # score accuracy.json
    uv run python -m benchmarks.judge --criteria hs_code
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.harness import RESULTS_DIR

JUDGE_TIMEOUT = 120.0
JUDGE_CONCURRENCY = 8


class Verdict(BaseModel):
    """One graded answer."""

    score: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0 per the scoring rubric")
    reason: str = Field(description="One sentence naming the deciding fact")


PROMPT = """{criteria}

Grade this answer.

QUESTION:
{question}

EXPECTED OUTPUT:
{expected}

ACTUAL OUTPUT:
{actual}

Return a score between 0.0 and 1.0 and a one-sentence reason."""


def _criteria(name: str) -> str:
    from evaluation.prompts import eval_prompts

    if name == "hs_code":
        return eval_prompts.HS_CODE_CRITERIA
    return eval_prompts.ANSWER_CORRECTNESS_CRITERIA


async def _grade_all(items: list[dict], criteria: str) -> list[dict]:
    from konte.llm import get_llm

    judge = get_llm(timeout=JUDGE_TIMEOUT, max_tokens=400).with_structured_output(Verdict)
    limiter = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def grade(item: dict) -> dict:
        if item.get("error") or not item.get("actual_output"):
            return {**item, "score": 0.0, "reason": item.get("error", "no answer produced")}
        prompt = PROMPT.format(
            criteria=criteria,
            question=item.get("question", ""),
            expected=item.get("expected_output", ""),
            actual=item.get("actual_output", ""),
        )
        async with limiter:
            try:
                verdict = await judge.ainvoke(prompt)
                return {**item, "score": float(verdict.score), "reason": verdict.reason}
            except Exception as error:
                return {**item, "score": None, "reason": f"judge failed: {error}"}

    return await asyncio.gather(*(grade(item) for item in items))


def _summarize(graded: list[dict]) -> dict:
    scores = [g["score"] for g in graded if isinstance(g.get("score"), (int, float))]
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "pass_rate_0_7": sum(1 for s in scores if s >= 0.7) / len(scores),
        "perfect_rate": sum(1 for s in scores if s >= 0.999) / len(scores),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "accuracy.json")
    parser.add_argument("--criteria", default="answer", choices=("answer", "hs_code"))
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "accuracy_scored.json")
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text())
    criteria = _criteria(args.criteria)

    scored: dict[str, dict] = {}
    for revision, runs in payload.get("runs", {}).items():
        measurements = next(
            (run.get("measurements") for run in runs if run.get("status") == "ok"), None
        )
        answers = (measurements or {}).get("answers") or []
        if not answers:
            scored[revision] = {"summary": {"count": 0}, "graded": []}
            continue
        print(f"grading {len(answers)} answers from {revision} ...", flush=True)
        graded = asyncio.run(_grade_all(answers, criteria))
        scored[revision] = {"summary": _summarize(graded), "graded": graded}

    out = {"criteria": args.criteria, "source": str(args.results), "revisions": scored}
    args.out.write_text(json.dumps(out, indent=2, default=str))

    for revision, data in scored.items():
        summary = data["summary"]
        if summary.get("count"):
            print(
                f"{revision:9s} n={summary['count']:3d}  mean={summary['mean_score']:.3f}  "
                f"pass@0.7={summary['pass_rate_0_7']:.1%}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
