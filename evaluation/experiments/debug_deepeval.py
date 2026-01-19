"""Debug DeepEval correctness metric on LLM reranking results."""

import json
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evaluation.custom_llm import BackendAIModel


def test_single_case():
    """Test DeepEval on a single case to debug."""
    # Load one result
    results_path = Path("evaluation/experiments/results/llm_rerank_binary.json")
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]

    # Find a successful case
    for r in results[:5]:
        if r.get("status") == "success":
            break

    print("=" * 80)
    print("TEST CASE")
    print("=" * 80)
    print(f"Input: {r['input'][:100]}...")
    print(f"\nExpected: {r['expected_output'][:200]}...")
    print(f"\nActual: {r['actual_output'][:200]}...")

    # Setup metric
    print("\n" + "=" * 80)
    print("RUNNING DEEPEVAL")
    print("=" * 80)

    custom_model = BackendAIModel()
    print(f"Model: {custom_model.model_name}")
    print(f"Endpoint: {custom_model.base_url}")

    metric = GEval(
        name="FactualCorrectness",
        criteria="""Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.

Focus on semantic equivalence - does the actual answer convey the same information?
Score 1.0 if answers are semantically equivalent.
Score 0.0 if answers contradict or are completely different.""",
        evaluation_steps=[
            "Compare the key facts in expected vs actual output",
            "Check if the core answer is the same",
            "Score based on factual agreement",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=custom_model,
        threshold=0.5,
    )

    tc = LLMTestCase(
        input=r["input"],
        actual_output=r["actual_output"],
        expected_output=r["expected_output"],
    )

    try:
        metric.measure(tc)
        print(f"\nScore: {metric.score}")
        print(f"Reason: {metric.reason}")
        print(f"Success: {metric.is_successful()}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def test_batch():
    """Test on first 10 cases."""
    results_path = Path("evaluation/experiments/results/llm_rerank_binary.json")
    with open(results_path) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r.get("status") == "success"][:10]

    custom_model = BackendAIModel()
    metric = GEval(
        name="FactualCorrectness",
        criteria="""Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.
Score 1.0 if semantically equivalent. Score 0.0 if contradictory.""",
        evaluation_steps=[
            "Compare key facts",
            "Score based on agreement",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=custom_model,
        threshold=0.5,
    )

    print("Testing 10 cases...")
    scores = []

    for i, r in enumerate(results):
        tc = LLMTestCase(
            input=r["input"],
            actual_output=r["actual_output"],
            expected_output=r["expected_output"],
        )

        try:
            metric.measure(tc)
            score = metric.score if metric.score is not None else 0.0
            scores.append(score)
            print(f"  Case {i+1}: score={score:.2f}, reason={metric.reason[:50] if metric.reason else 'N/A'}...")
        except Exception as e:
            print(f"  Case {i+1}: ERROR - {e}")
            scores.append(0.0)

    print(f"\nAvg score: {sum(scores)/len(scores):.3f}")
    print(f"Pass rate: {sum(1 for s in scores if s >= 0.5)/len(scores):.1%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        test_batch()
    else:
        test_single_case()
