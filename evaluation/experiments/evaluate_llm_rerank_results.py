"""Evaluate LLM reranking results using DeepEval correctness metric."""

import argparse
import json
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tqdm import tqdm

from evaluation.custom_llm import BackendAIModel


def evaluate_correctness(results_path: Path, output_path: Path) -> dict:
    """Run DeepEval correctness metric on LLM reranking results."""
    # Load results
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    method = data["method"]

    print(f"Evaluating {len(results)} results from {method} method")

    # Setup metric
    custom_model = BackendAIModel()
    metric = GEval(
        name="FactualCorrectness",
        criteria="""Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.

Focus on:
1. Are the main HS codes or classification codes mentioned in expected output ALSO present in actual output?
2. Are the key facts, categories, or classifications from expected output covered in actual output?
3. Is the core answer semantically equivalent, even if worded differently or more detailed?

IGNORE these differences:
- Length differences (actual may be longer with more detail - this is OK)
- Format differences (bullet points vs paragraphs, etc.)
- Additional context, explanations, or citations in actual output
- Language mixing (Korean text alongside English)
- Different ordering of information

Score 1.0 if: Actual contains all key facts from expected (even if it has more)
Score 0.7-0.9 if: Actual contains most key facts with minor omissions
Score 0.4-0.6 if: Actual contains some key facts but misses important ones
Score 0.0-0.3 if: Actual is missing most key facts or contains wrong information""",
        evaluation_steps=[
            "Extract the key HS codes, classification codes, or category names from the expected output",
            "Check if these same codes/categories appear in the actual output",
            "Extract the key factual claims from the expected output",
            "Verify these facts are present in the actual output (can be worded differently)",
            "Ignore length, format, and stylistic differences",
            "Score based on factual coverage, not format matching",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=custom_model,
        threshold=0.5,
    )

    scores = []
    errors = 0

    for r in tqdm(results, desc="Evaluating"):
        if r.get("status") != "success":
            errors += 1
            continue

        tc = LLMTestCase(
            input=r["input"],
            actual_output=r["actual_output"],
            expected_output=r["expected_output"],
            retrieval_context=r.get("retrieval_context", []),
        )

        try:
            metric.measure(tc)
            if metric.score is not None:
                scores.append(metric.score)
        except Exception as e:
            print(f"Error: {e}")
            errors += 1

    # Calculate stats
    if scores:
        avg_score = sum(scores) / len(scores)
        pass_rate = sum(1 for s in scores if s >= 0.5) / len(scores)
        passed = sum(1 for s in scores if s >= 0.5)
    else:
        avg_score = 0
        pass_rate = 0
        passed = 0

    result = {
        "method": method,
        "metric": "correctness",
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "passed": passed,
        "failed": len(scores) - passed,
        "total": len(scores),
        "errors": errors,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults for {method}:")
    print(f"  Avg Score: {avg_score:.3f}")
    print(f"  Pass Rate: {pass_rate:.1%}")
    print(f"  Passed: {passed}/{len(scores)}")
    print(f"  Errors: {errors}")
    print(f"Saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluate_correctness(args.results, args.output)


if __name__ == "__main__":
    main()
