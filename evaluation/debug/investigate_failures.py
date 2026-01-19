"""Investigate why 20% of test cases fail FactualCorrectness.

Possible causes:
1. No relevant chunks retrieved (retrieval failure)
2. LLM answer generation poor quality
3. LLM-Judge evaluation issue
"""

import json
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evaluation.custom_llm import BackendAIModel


def load_answers(mode: str = "hybrid") -> list[dict]:
    """Load answers from checkpoint."""
    path = Path(f"evaluation/results/checkpoints/answers_{mode}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_failures():
    """Analyze failed test cases."""
    answers = load_answers("hybrid")
    print(f"Total test cases: {len(answers)}")

    # Create metric
    model = BackendAIModel()
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
        model=model,
        threshold=0.5,
        verbose_mode=True,  # Get explanation
    )

    # Evaluate and collect failures
    failures = []
    successes = []

    print("\nEvaluating test cases...")
    for i, ans in enumerate(answers):
        if ans.get("status") != "success":
            continue

        tc = LLMTestCase(
            input=ans["input"],
            actual_output=ans["actual_output"],
            expected_output=ans["expected_output"],
            retrieval_context=ans["retrieval_context"],
        )

        try:
            metric.measure(tc)
            result = {
                "index": i,
                "input": ans["input"],
                "expected_output": ans["expected_output"],
                "actual_output": ans["actual_output"],
                "retrieval_context": ans["retrieval_context"][:3],  # Top 3 chunks
                "top_score": ans["top_score"],
                "score": metric.score,
                "reason": metric.reason,
            }

            if metric.score < 0.5:
                failures.append(result)
            else:
                successes.append(result)

        except Exception as e:
            print(f"Error at {i}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(answers)} | Failures: {len(failures)}")

    print(f"\nTotal evaluated: {len(failures) + len(successes)}")
    print(f"Failures (score < 0.5): {len(failures)} ({len(failures)/(len(failures)+len(successes))*100:.1f}%)")
    print(f"Successes: {len(successes)}")

    # Analyze failure patterns
    print("\n" + "="*80)
    print("FAILURE ANALYSIS")
    print("="*80)

    # 1. Check retrieval scores
    failure_retrieval_scores = [f["top_score"] for f in failures]
    success_retrieval_scores = [s["top_score"] for s in successes]

    print(f"\n1. RETRIEVAL QUALITY:")
    print(f"   Failures avg retrieval score: {sum(failure_retrieval_scores)/len(failure_retrieval_scores):.3f}" if failure_retrieval_scores else "   No failures")
    print(f"   Successes avg retrieval score: {sum(success_retrieval_scores)/len(success_retrieval_scores):.3f}" if success_retrieval_scores else "   No successes")

    # 2. Check answer lengths
    failure_answer_lens = [len(f["actual_output"]) for f in failures]
    success_answer_lens = [len(s["actual_output"]) for s in successes]

    print(f"\n2. ANSWER LENGTH:")
    print(f"   Failures avg answer length: {sum(failure_answer_lens)/len(failure_answer_lens):.0f} chars" if failure_answer_lens else "   No failures")
    print(f"   Successes avg answer length: {sum(success_answer_lens)/len(success_answer_lens):.0f} chars" if success_answer_lens else "   No successes")

    # 3. Show sample failures
    print(f"\n3. SAMPLE FAILURES (first 5):")
    for i, f in enumerate(failures[:5]):
        print(f"\n--- Failure {i+1} (score: {f['score']:.2f}) ---")
        print(f"Question: {f['input'][:100]}...")
        print(f"Expected: {f['expected_output'][:150]}...")
        print(f"Actual: {f['actual_output'][:150]}...")
        print(f"Retrieval Score: {f['top_score']:.3f}")
        print(f"Judge Reason: {f['reason'][:200]}..." if f['reason'] else "No reason")

    # Save detailed results
    output = {
        "summary": {
            "total": len(failures) + len(successes),
            "failures": len(failures),
            "successes": len(successes),
            "failure_rate": len(failures) / (len(failures) + len(successes)),
            "avg_failure_retrieval": sum(failure_retrieval_scores) / len(failure_retrieval_scores) if failure_retrieval_scores else 0,
            "avg_success_retrieval": sum(success_retrieval_scores) / len(success_retrieval_scores) if success_retrieval_scores else 0,
        },
        "failures": failures,
        "sample_successes": successes[:5],
    }

    output_path = Path("evaluation/debug/failure_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed analysis to {output_path}")


if __name__ == "__main__":
    analyze_failures()
