"""Run full DeepEval correctness evaluation on LLM reranking results."""

import json
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tqdm import tqdm

from evaluation.custom_llm import BackendAIModel


def evaluate_all(method: str = "binary", version: str = ""):
    """Run DeepEval on all cases."""
    suffix = f"_{version}" if version else ""
    results_path = Path(f"evaluation/experiments/results/llm_rerank_{method}{suffix}.json")
    with open(results_path) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r.get("status") == "success"]
    print(f"Evaluating {len(results)} cases from {method}{suffix} method")

    custom_model = BackendAIModel()
    print(f"Model: {custom_model.model_name}")
    print(f"Endpoint: {custom_model.base_url}")

    metric = GEval(
        name="FactualCorrectness",
        criteria="""Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.

Focus on HS code accuracy and semantic equivalence:
- The key information is the HS CODE (e.g., 2523.21, 제8540호, 8540.20)
- Ignore format differences: "제2523.21호" = "2523.21" = "제2523호의 21" (all equivalent)
- Ignore language mixing (Korean/English)
- Ignore length differences or extra explanation

Scoring:
- Score 1.0 if the SAME HS CODE is mentioned (regardless of format)
- Score 0.7-0.9 if mostly correct with minor code variations
- Score 0.4-0.6 if partially correct (related but not exact code)
- Score 0.0-0.3 if wrong HS code or contradictory information

IMPORTANT: If actual output provides a DIFFERENT but MORE CORRECT HS code based on the question context, score 0.7+ (the expected output may be wrong).""",
        evaluation_steps=[
            "Extract the HS code(s) from both expected and actual outputs",
            "Normalize format differences (제2523호 = 2523 = 제2523.00호)",
            "Compare if they refer to the same classification",
            "Score based on code match, ignoring format/language differences",
        ],
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
        "metric": "deepeval_correctness",
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
    print(f"Results for {method} method:")
    print(f"{'='*60}")
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
    evaluate_all(method, version)
