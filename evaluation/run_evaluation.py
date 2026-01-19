"""RAG Evaluation using DeepEval metrics over wco_hs_explanatory_notes knowledge base.

Metrics evaluated:
- Answer Relevancy: How well the answer addresses the question
- Faithfulness: Factual consistency with provided context
- Contextual Recall: Whether context contains necessary information
- Contextual Precision: Whether retrieved context is focused and relevant
- Contextual Relevancy: Overall context-to-query alignment
- Correctness: Whether actual output matches expected output (accuracy)
"""

import asyncio
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
from konte import Project


def load_test_cases(json_path: Path) -> list[dict]:
    """Load test cases from JSON file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


async def generate_rag_responses(
    project: Project,
    test_cases: list[dict],
    top_k: int = 5,
    max_chunks: int = 5,
) -> list[dict]:
    """Generate RAG responses for each test case.

    Returns test cases enriched with:
    - actual_output: The RAG-generated answer
    - retrieval_context: List of retrieved chunk contents
    """
    enriched = []

    for tc in test_cases:
        query = tc["input"]

        # Get retrieval + answer from Konte
        response, answer = await project.query_with_answer(
            query=query,
            mode="hybrid",
            top_k=top_k,
            max_chunks=max_chunks,
        )

        # Extract retrieval context as list of strings
        retrieval_context = [result.content for result in response.results[:max_chunks]]

        enriched.append(
            {
                "input": tc["input"],
                "expected_output": tc.get("expected_output", ""),
                "actual_output": answer.answer,
                "retrieval_context": retrieval_context,
            }
        )

    return enriched


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
    """Run all RAG evaluation metrics.

    Runs metrics one at a time to handle long prompts better.
    Returns list of results per metric.
    """
    # Initialize all metrics with custom model
    metrics = [
        ("Answer Relevancy", AnswerRelevancyMetric(
            threshold=threshold,
            model=custom_model,
            include_reason=True,
        )),
        ("Faithfulness", FaithfulnessMetric(
            threshold=threshold,
            model=custom_model,
            include_reason=False,  # Skip reason generation to save tokens
            truths_extraction_limit=10,  # Limit truths to prevent token explosion
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

    # Run each metric separately to avoid token limit issues
    for metric_name, metric in metrics:
        print(f"\nEvaluating {metric_name}...")
        try:
            results = evaluate(
                test_cases=test_cases,
                metrics=[metric],
            )
            all_results.append({
                "metric": metric_name,
                "results": results,
                "scores": [tc.metrics_data[0].score if tc.metrics_data else None for tc in results.test_results],
            })
            avg_score = sum(s for s in all_results[-1]["scores"] if s is not None) / len([s for s in all_results[-1]["scores"] if s is not None])
            print(f"  Average {metric_name}: {avg_score:.3f}")
        except Exception as e:
            print(f"  Error in {metric_name}: {e}")
            all_results.append({
                "metric": metric_name,
                "error": str(e),
            })

    return all_results


async def main():
    """Main evaluation entry point."""
    # Check for synthetic dataset or use manual test cases
    synthetic_path = Path("evaluation/data/synthetic_goldens.json")

    if synthetic_path.exists():
        print(f"Loading synthetic test cases from {synthetic_path}")
        test_data = load_test_cases(synthetic_path)
    else:
        print("No synthetic dataset found. Using manual test cases.")
        from evaluation.test_dataset import get_test_cases

        manual_cases = get_test_cases()
        # Limit to 5 test cases to keep evaluation manageable
        test_data = [
            {"input": tc.input, "expected_output": tc.expected_output}
            for tc in manual_cases[:5]
        ]

    print(f"Loaded {len(test_data)} test cases")

    # Load the wco_hs_explanatory_notes project
    print("\nLoading wco_hs_explanatory_notes project...")
    project = Project.open("wco_hs_explanatory_notes")
    print(f"Project loaded: {project._config.name}")

    # Generate RAG responses
    # Use fewer chunks to keep DeepEval prompts within token limits
    print("\nGenerating RAG responses...")
    enriched_cases = await generate_rag_responses(
        project=project,
        test_cases=test_data,
        top_k=20,
        max_chunks=15,
    )

    # Save enriched cases for debugging
    output_dir = Path("evaluation/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "enriched_test_cases.json", "w", encoding="utf-8") as f:
        json.dump(enriched_cases, f, indent=2, ensure_ascii=False)
    print(f"Saved enriched test cases to {output_dir / 'enriched_test_cases.json'}")

    # Convert to DeepEval format
    deepeval_cases = create_deepeval_test_cases(enriched_cases)

    # Initialize custom model for evaluation
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

    print("\n" + "=" * 60)
    print("Evaluation Complete!")

    return results


if __name__ == "__main__":
    asyncio.run(main())
