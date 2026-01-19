"""Run DeepEval metrics on existing enriched_test_cases.json.

Skips RAG response generation - uses pre-generated data.
"""

import json
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evaluation.custom_llm import BackendAIModel


def load_enriched_cases(json_path: Path) -> list[dict]:
    """Load enriched test cases from JSON file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def create_deepeval_test_cases(enriched_cases: list[dict]) -> list[LLMTestCase]:
    """Convert enriched cases to DeepEval LLMTestCase format."""
    return [
        LLMTestCase(
            input=tc["input"],
            actual_output=tc["actual_output"],
            expected_output=tc.get("expected_output"),
            retrieval_context=tc["retrieval_context"],
        )
        for tc in enriched_cases
    ]


def run_evaluation(
    test_cases: list[LLMTestCase],
    custom_model: BackendAIModel,
    threshold: float = 0.5,
) -> list[dict]:
    """Run all RAG evaluation metrics."""
    metrics = [
        ("Answer Relevancy", AnswerRelevancyMetric(
            threshold=threshold,
            model=custom_model,
            include_reason=True,
        )),
        ("Faithfulness", FaithfulnessMetric(
            threshold=threshold,
            model=custom_model,
            include_reason=False,
            truths_extraction_limit=10,
        )),
        ("Contextual Recall", ContextualRecallMetric(
            threshold=threshold,
            model=custom_model,
            include_reason=True,
        )),
        ("Contextual Precision", ContextualPrecisionMetric(
            threshold=threshold,
            model=custom_model,
            include_reason=True,
        )),
        ("Contextual Relevancy", ContextualRelevancyMetric(
            threshold=threshold,
            model=custom_model,
            include_reason=True,
        )),
        ("Correctness", GEval(
            name="Correctness",
            criteria="Determine if the actual output is factually correct and semantically similar to the expected output. The actual output should convey the same information and meaning as the expected output.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=custom_model,
            threshold=threshold,
        )),
    ]

    all_results = []

    for metric_name, metric in metrics:
        print(f"\nEvaluating {metric_name}...")
        try:
            results = evaluate(
                test_cases=test_cases,
                metrics=[metric],
            )
            scores = [tc.metrics_data[0].score if tc.metrics_data else None for tc in results.test_results]
            valid_scores = [s for s in scores if s is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            all_results.append({
                "metric": metric_name,
                "avg_score": avg_score,
                "pass_rate": sum(1 for s in valid_scores if s >= threshold) / len(valid_scores) if valid_scores else 0,
                "scores": scores,
            })
            print(f"  Average {metric_name}: {avg_score:.3f}")
            print(f"  Pass Rate: {all_results[-1]['pass_rate']:.1%}")
        except Exception as e:
            print(f"  Error in {metric_name}: {e}")
            all_results.append({
                "metric": metric_name,
                "error": str(e),
            })

    return all_results


def main():
    """Run metrics on existing enriched test cases."""
    enriched_path = Path("evaluation/data/enriched_test_cases.json")

    if not enriched_path.exists():
        print(f"Error: {enriched_path} not found. Run full evaluation first.")
        return

    print(f"Loading enriched test cases from {enriched_path}")
    enriched_cases = load_enriched_cases(enriched_path)
    print(f"Loaded {len(enriched_cases)} test cases")

    # Convert to DeepEval format
    deepeval_cases = create_deepeval_test_cases(enriched_cases)

    # Initialize custom model
    print("\nInitializing BackendAI model for evaluation...")
    custom_model = BackendAIModel()

    # Run evaluation
    print("\nRunning RAG evaluation with 6 metrics...")
    print("=" * 60)
    results = run_evaluation(
        test_cases=deepeval_cases,
        custom_model=custom_model,
        threshold=0.5,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if "error" not in r:
            print(f"{r['metric']}: {r['avg_score']:.3f} (Pass Rate: {r['pass_rate']:.1%})")
        else:
            print(f"{r['metric']}: ERROR - {r['error']}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")


if __name__ == "__main__":
    main()
