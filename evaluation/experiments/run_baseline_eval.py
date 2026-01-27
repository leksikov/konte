"""Baseline RAG evaluation without reranking.

Evaluates raw retrieval (no context, no LLM reranking) for comparison
against contextual RAG with reranking.

Pipeline:
1. Load baseline project (built with skip_context=True)
2. For each test case:
   - Retrieve with project.query() - direct hybrid retrieval
   - Generate answer with generate_answer() using BackendAI model
3. Output in same format as llm_reranking.py for DeepEval compatibility
"""

import asyncio
import json
from pathlib import Path

import structlog
from tqdm import tqdm

from konte import Project
from konte.generator import generate_answer

logger = structlog.get_logger()


async def evaluate_baseline(
    project: Project,
    test_cases: list[dict],
    top_k: int = 15,
    max_cases: int | None = None,
) -> dict:
    """Evaluate baseline retrieval on test cases.

    Args:
        project: Konte project to query (baseline, no context).
        test_cases: List of test cases with 'input' and 'expected_output'.
        top_k: Number of chunks to retrieve and use for answer generation.
        max_cases: Limit number of test cases (for quick testing).

    Returns:
        Dict with evaluation results compatible with run_deepeval_full.py.
    """
    cases = test_cases[:max_cases] if max_cases else test_cases
    results = []

    for tc in tqdm(cases, desc="Baseline Eval"):
        query = tc["input"]
        expected = tc.get("expected_output", "")

        try:
            # Direct retrieval (no reranking)
            response = project.query(query, mode="hybrid", top_k=top_k)

            # Generate answer
            answer = await generate_answer(
                question=query,
                retrieval_response=response,
                max_chunks=top_k,
            )

            results.append({
                "input": query,
                "expected_output": expected,
                "actual_output": answer.answer,
                "retrieval_context": [r.content for r in response.results[:top_k]],
                "chunks_retrieved": len(response.results),
                "status": "success",
            })

        except Exception as e:
            logger.error("baseline_eval_failed", query=query[:50], error=str(e))
            results.append({
                "input": query,
                "expected_output": expected,
                "actual_output": f"Error: {e}",
                "retrieval_context": [],
                "status": "error",
            })

    success = sum(1 for r in results if r["status"] == "success")

    return {
        "method": "baseline",
        "initial_k": top_k,
        "final_k": top_k,
        "total_cases": len(cases),
        "success": success,
        "results": results,
    }


async def main():
    """Run baseline evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Baseline RAG Evaluation")
    parser.add_argument("--project", default="baseline_wco_korean")
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=Path("evaluation/data/synthetic/deepeval_goldens_korean_100.json"),
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=0, help="Limit cases (0=all)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/experiments/results/llm_rerank_baseline_diverse.json"),
    )
    args = parser.parse_args()

    # Load project
    print(f"Loading baseline project: {args.project}")
    project = Project.open(args.project)
    print(f"Project loaded with {len(project._chunks)} chunks")

    # Load test cases
    print(f"Loading test cases from {args.test_cases}")
    with open(args.test_cases) as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    # Run evaluation
    print(f"\nRunning baseline evaluation")
    print(f"  Top-K: {args.top_k}")
    print(f"  Max cases: {args.max_cases or 'all'}")

    max_cases = args.max_cases if args.max_cases > 0 else None

    results = await evaluate_baseline(
        project=project,
        test_cases=test_cases,
        top_k=args.top_k,
        max_cases=max_cases,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")
    print(f"Success: {results['success']}/{results['total_cases']}")


if __name__ == "__main__":
    asyncio.run(main())
