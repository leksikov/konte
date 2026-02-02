"""Ablation Case 1: Perfect Evidence → LLM Ceiling Test.

Tests if Konte LLM can produce correct answers when given perfect evidence directly.
Expected accuracy: ~99%

This test bypasses retrieval entirely - it creates a synthetic RetrievalResponse
with the retrieval_context as the only chunk, then calls generate_answer().

Evidence source: retrieval_context[0] from test dataset (the actual source chunk).
"""

import asyncio
import json
import random
from pathlib import Path

import structlog
from tqdm import tqdm

from konte.generator import generate_answer
from konte.models import RetrievalResponse, RetrievalResult

logger = structlog.get_logger()

# Paths
HS_CODE_DATASET = Path("evaluation/data/synthetic/synthetic_goldens_100.json")
DEEPEVAL_DATASET = Path("evaluation/data/synthetic/deepeval_goldens_korean_100.json")
RESULTS_DIR = Path("evaluation/experiments/results")

# Sampling
SAMPLE_SIZE = 15
RANDOM_SEED = 42


def load_test_cases(dataset_path: Path, sample_size: int = 15) -> list[dict]:
    """Load and sample test cases from dataset."""
    with open(dataset_path) as f:
        cases = json.load(f)

    random.seed(RANDOM_SEED)
    if len(cases) > sample_size:
        cases = random.sample(cases, sample_size)

    return cases


def get_evidence_from_case(tc: dict) -> str:
    """Extract evidence from test case.

    Uses retrieval_context[0] as the evidence (actual source chunk).
    """
    retrieval_context = tc.get("retrieval_context", [])
    if retrieval_context:
        return retrieval_context[0]
    # Fallback to expected_output if no retrieval_context
    return tc.get("expected_output", "")


def create_perfect_evidence_response(
    question: str,
    evidence: str,
) -> RetrievalResponse:
    """Create synthetic RetrievalResponse with perfect evidence.

    Args:
        question: The user's question.
        evidence: The source context chunk.

    Returns:
        RetrievalResponse with evidence as the only chunk.
    """
    perfect_result = RetrievalResult(
        content=evidence,
        context="[Ablation study: perfect evidence from retrieval_context]",
        score=1.0,
        source="ablation_study",
        chunk_id="perfect_evidence",
        metadata={"ablation_case": "case1_perfect_evidence"},
    )

    return RetrievalResponse(
        results=[perfect_result],
        query=question,
        total_found=1,
        top_score=1.0,
        score_spread=0.0,
        has_high_confidence=True,
        suggested_action="deliver",
    )


async def evaluate_single_case(
    question: str,
    expected_output: str,
    evidence: str,
    case_idx: int,
) -> dict:
    """Evaluate single test case with perfect evidence.

    Args:
        question: Input question.
        expected_output: Expected answer (for comparison).
        evidence: Source context chunk (from retrieval_context).
        case_idx: Case index for logging.

    Returns:
        Evaluation result dict.
    """
    try:
        # Create perfect evidence response
        fake_response = create_perfect_evidence_response(question, evidence)

        # Generate answer
        answer = await generate_answer(
            question=question,
            retrieval_response=fake_response,
            max_chunks=1,
        )

        return {
            "case_idx": case_idx,
            "input": question,
            "expected_output": expected_output,
            "evidence_used": evidence[:500] + "..." if len(evidence) > 500 else evidence,
            "actual_output": answer.answer,
            "model": answer.model,
            "status": "success",
        }

    except Exception as e:
        logger.error("case_failed", case_idx=case_idx, error=str(e))
        return {
            "case_idx": case_idx,
            "input": question,
            "expected_output": expected_output,
            "evidence_used": evidence[:500] if evidence else "",
            "actual_output": f"Error: {e}",
            "model": "",
            "status": "error",
        }


async def run_ablation_case1(
    hs_code_cases: list[dict],
    deepeval_cases: list[dict],
) -> dict:
    """Run Case 1 ablation study.

    Args:
        hs_code_cases: HS code dataset test cases.
        deepeval_cases: DeepEval dataset test cases.

    Returns:
        Complete results dict.
    """
    results = []

    # Process HS code cases
    print("\n=== Processing HS Code Dataset ===")
    for i, tc in enumerate(tqdm(hs_code_cases, desc="HS Code")):
        evidence = get_evidence_from_case(tc)
        result = await evaluate_single_case(
            question=tc["input"],
            expected_output=tc["expected_output"],
            evidence=evidence,
            case_idx=i,
        )
        result["dataset"] = "hs_code"
        results.append(result)

    # Process DeepEval cases
    print("\n=== Processing DeepEval Dataset ===")
    for i, tc in enumerate(tqdm(deepeval_cases, desc="DeepEval")):
        evidence = get_evidence_from_case(tc)
        result = await evaluate_single_case(
            question=tc["input"],
            expected_output=tc["expected_output"],
            evidence=evidence,
            case_idx=len(hs_code_cases) + i,
        )
        result["dataset"] = "deepeval"
        results.append(result)

    # Summary statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    hs_success = sum(1 for r in results if r["status"] == "success" and r["dataset"] == "hs_code")
    deepeval_success = sum(1 for r in results if r["status"] == "success" and r["dataset"] == "deepeval")

    return {
        "ablation_case": "case1_perfect_evidence",
        "description": "LLM ceiling test - perfect evidence (retrieval_context) provided",
        "evidence_source": "retrieval_context[0]",
        "expected_accuracy": "~99%",
        "total_cases": len(results),
        "success_count": success_count,
        "hs_code_cases": len(hs_code_cases),
        "hs_code_success": hs_success,
        "deepeval_cases": len(deepeval_cases),
        "deepeval_success": deepeval_success,
        "results": results,
    }


async def main():
    """Run Case 1 ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Case 1: Perfect Evidence Test")
    parser.add_argument("--hs-code-dataset", type=Path, default=HS_CODE_DATASET)
    parser.add_argument("--deepeval-dataset", type=Path, default=DEEPEVAL_DATASET)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ablation_case1_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION CASE 1: Perfect Evidence → LLM Ceiling Test")
    print("=" * 60)
    print(f"\nEvidence source: retrieval_context[0] (actual source chunks)")
    print(f"\nLoading datasets...")
    print(f"  HS Code: {args.hs_code_dataset}")
    print(f"  DeepEval: {args.deepeval_dataset}")
    print(f"  Sample size: {args.sample_size} per dataset")

    # Load and sample test cases
    hs_code_cases = load_test_cases(args.hs_code_dataset, args.sample_size)
    deepeval_cases = load_test_cases(args.deepeval_dataset, args.sample_size)

    print(f"\nLoaded {len(hs_code_cases)} HS code cases")
    print(f"Loaded {len(deepeval_cases)} DeepEval cases")
    print(f"Total: {len(hs_code_cases) + len(deepeval_cases)} cases")

    # Run evaluation
    results = await run_ablation_case1(hs_code_cases, deepeval_cases)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total cases: {results['total_cases']}")
    print(f"Success: {results['success_count']}/{results['total_cases']}")
    print(f"  HS Code: {results['hs_code_success']}/{results['hs_code_cases']}")
    print(f"  DeepEval: {results['deepeval_success']}/{results['deepeval_cases']}")
    print(f"\nResults saved to: {args.output}")
    print("\nNote: Accuracy evaluation requires running DeepEval GEval metric.")
    print("Use run_deepeval_full.py to evaluate actual vs expected outputs.")


if __name__ == "__main__":
    asyncio.run(main())
