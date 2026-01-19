"""Run full DeepEval correctness evaluation on LLM reranking results.

Supports two evaluation types:
- hs_code: For HS code classification questions (synthetic_goldens_100.json)
- answer: For diverse RAG questions (deepeval_goldens_korean_100.json)
"""

import json
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tqdm import tqdm

from evaluation.custom_llm import BackendAIModel
from evaluation.prompts.eval_prompts import (
    ANSWER_CORRECTNESS_CRITERIA,
    ANSWER_CORRECTNESS_STEPS,
    HS_CODE_CRITERIA,
    HS_CODE_STEPS,
)


def evaluate_all(method: str = "binary", version: str = "", eval_type: str = "answer"):
    """Run DeepEval on all cases.

    Args:
        method: Reranking method (e.g., "binary").
        version: Version suffix for results file.
        eval_type: "hs_code" for HS classification, "answer" for general RAG (default).
    """
    suffix = f"_{version}" if version else ""
    results_path = Path(f"evaluation/experiments/results/llm_rerank_{method}{suffix}.json")
    with open(results_path) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r.get("status") == "success"]
    print(f"Evaluating {len(results)} cases from {method}{suffix} method")
    print(f"Evaluation type: {eval_type}")

    custom_model = BackendAIModel()
    print(f"Model: {custom_model.model_name}")
    print(f"Endpoint: {custom_model.base_url}")

    # Select prompt based on evaluation type
    if eval_type == "hs_code":
        criteria = HS_CODE_CRITERIA
        steps = HS_CODE_STEPS
        metric_name = "HSCodeCorrectness"
    else:
        criteria = ANSWER_CORRECTNESS_CRITERIA
        steps = ANSWER_CORRECTNESS_STEPS
        metric_name = "AnswerCorrectness"

    print(f"Metric: {metric_name}")

    metric = GEval(
        name=metric_name,
        criteria=criteria,
        evaluation_steps=steps,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=custom_model,
        threshold=0.5,
    )

    scores = []
    detailed_results = []

    for r in tqdm(results, desc="Evaluating"):
        tc = LLMTestCase(
            input=r["input"],
            actual_output=r["actual_output"],
            expected_output=r["expected_output"],
        )

        try:
            metric.measure(tc)
            score = metric.score if metric.score is not None else 0.0
            reason = metric.reason or "unknown"
        except Exception as e:
            score = 0.0
            reason = f"Error: {e}"

        scores.append(score)
        detailed_results.append({
            "input": r["input"][:100],
            "score": score,
            "reason": reason[:200],
            "passed": score >= 0.5,
        })

    # Calculate stats
    avg_score = sum(scores) / len(scores)
    pass_rate = sum(1 for s in scores if s >= 0.5) / len(scores)
    passed = sum(1 for s in scores if s >= 0.5)

    result = {
        "method": f"{method}{suffix}",
        "eval_type": eval_type,
        "metric": metric_name,
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "passed": passed,
        "failed": len(scores) - passed,
        "total": len(scores),
        "detailed_results": detailed_results,
    }

    # Save results
    output_path = Path(f"evaluation/experiments/results/{method}{suffix}_deepeval_correctness.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results for {method}{suffix} ({eval_type}):")
    print(f"{'='*60}")
    print(f"  Metric: {metric_name}")
    print(f"  Avg Score: {avg_score:.3f}")
    print(f"  Pass Rate: {pass_rate:.1%}")
    print(f"  Passed: {passed}/{len(scores)}")
    print(f"  Failed: {len(scores) - passed}")
    print(f"\nSaved to: {output_path}")

    # Show failures
    failures = [r for r in detailed_results if not r["passed"]]
    print(f"\n--- FAILURES ({len(failures)}) ---")
    for f in failures[:10]:
        print(f"  Q: {f['input'][:60]}...")
        print(f"  Score: {f['score']:.2f}")
        print(f"  Reason: {f['reason'][:100]}...")
        print()


if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "binary"
    version = sys.argv[2] if len(sys.argv) > 2 else ""
    eval_type = sys.argv[3] if len(sys.argv) > 3 else "answer"
    evaluate_all(method, version, eval_type)
