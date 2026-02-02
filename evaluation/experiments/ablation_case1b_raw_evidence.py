"""Ablation Case 1b: Raw Evidence → LLM Ceiling Test (No Generated Context).

Tests if Konte LLM can produce correct answers when given RAW evidence directly.
This is a comparison to Case 1 which uses contextual evidence (with generated context).

Case 1: Uses full retrieval_context (with 맥락 설명, 문서 위치, 관련 HS 코드)
Case 1b: Uses only raw document text (stripped of generated context)

Expected accuracy: Lower than Case 1 (context helps LLM understand evidence).

Evidence source: Raw text extracted from retrieval_context[0].
"""

import asyncio
import json
import random
from pathlib import Path

import structlog
from tqdm import tqdm

from evaluation.experiments.evidence_utils import extract_raw_evidence
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


def get_raw_evidence_from_case(tc: dict) -> tuple[str, str]:
    """Extract raw evidence from test case.

    Uses retrieval_context[0] as the source, but strips generated context.

    Returns:
        Tuple of (raw_evidence, original_evidence) for comparison.
    """
    retrieval_context = tc.get("retrieval_context", [])
    if retrieval_context:
        original = retrieval_context[0]
        raw = extract_raw_evidence(original)
        return raw, original
    # Fallback to expected_output if no retrieval_context
    expected = tc.get("expected_output", "")
    return expected, expected


def create_raw_evidence_response(
    question: str,
    raw_evidence: str,
) -> RetrievalResponse:
    """Create synthetic RetrievalResponse with raw evidence only.

    Args:
        question: The user's question.
        raw_evidence: The raw document text (without generated context).

    Returns:
        RetrievalResponse with raw evidence as the only chunk.
    """
    raw_result = RetrievalResult(
        content=raw_evidence,
        context="[Ablation study: raw evidence - no generated context]",
        score=1.0,
        source="ablation_study",
        chunk_id="raw_evidence",
        metadata={"ablation_case": "case1b_raw_evidence"},
    )

    return RetrievalResponse(
        results=[raw_result],
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
    raw_evidence: str,
    original_evidence: str,
    case_idx: int,
) -> dict:
    """Evaluate single test case with raw evidence.

    Args:
        question: Input question.
        expected_output: Expected answer (for comparison).
        raw_evidence: Raw document text (stripped of context).
        original_evidence: Original evidence (with context) for comparison.
        case_idx: Case index for logging.

    Returns:
        Evaluation result dict.
    """
    try:
        # Create raw evidence response
        fake_response = create_raw_evidence_response(question, raw_evidence)

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
            "evidence_type": "raw",
            "raw_evidence_len": len(raw_evidence),
            "original_evidence_len": len(original_evidence),
            "evidence_reduction_pct": round(
                (1 - len(raw_evidence) / len(original_evidence)) * 100, 1
            )
            if original_evidence
            else 0,
            "evidence_used": raw_evidence[:500] + "..."
            if len(raw_evidence) > 500
            else raw_evidence,
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
            "evidence_type": "raw",
            "raw_evidence_len": len(raw_evidence) if raw_evidence else 0,
            "original_evidence_len": len(original_evidence) if original_evidence else 0,
            "evidence_reduction_pct": 0,
            "evidence_used": raw_evidence[:500] if raw_evidence else "",
            "actual_output": f"Error: {e}",
            "model": "",
            "status": "error",
        }


async def run_ablation_case1b(
    hs_code_cases: list[dict],
    deepeval_cases: list[dict],
) -> dict:
    """Run Case 1b ablation study.

    Args:
        hs_code_cases: HS code dataset test cases.
        deepeval_cases: DeepEval dataset test cases.

    Returns:
        Complete results dict.
    """
    results = []
    total_raw_len = 0
    total_original_len = 0

    # Process HS code cases
    print("\n=== Processing HS Code Dataset (Raw Evidence) ===")
    for i, tc in enumerate(tqdm(hs_code_cases, desc="HS Code")):
        raw_evidence, original_evidence = get_raw_evidence_from_case(tc)
        total_raw_len += len(raw_evidence)
        total_original_len += len(original_evidence)

        result = await evaluate_single_case(
            question=tc["input"],
            expected_output=tc["expected_output"],
            raw_evidence=raw_evidence,
            original_evidence=original_evidence,
            case_idx=i,
        )
        result["dataset"] = "hs_code"
        results.append(result)

    # Process DeepEval cases
    print("\n=== Processing DeepEval Dataset (Raw Evidence) ===")
    for i, tc in enumerate(tqdm(deepeval_cases, desc="DeepEval")):
        raw_evidence, original_evidence = get_raw_evidence_from_case(tc)
        total_raw_len += len(raw_evidence)
        total_original_len += len(original_evidence)

        result = await evaluate_single_case(
            question=tc["input"],
            expected_output=tc["expected_output"],
            raw_evidence=raw_evidence,
            original_evidence=original_evidence,
            case_idx=len(hs_code_cases) + i,
        )
        result["dataset"] = "deepeval"
        results.append(result)

    # Summary statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    hs_success = sum(
        1 for r in results if r["status"] == "success" and r["dataset"] == "hs_code"
    )
    deepeval_success = sum(
        1 for r in results if r["status"] == "success" and r["dataset"] == "deepeval"
    )
    avg_reduction = (
        round((1 - total_raw_len / total_original_len) * 100, 1)
        if total_original_len
        else 0
    )

    return {
        "ablation_case": "case1b_raw_evidence",
        "description": "LLM ceiling test - RAW evidence only (no generated context)",
        "evidence_source": "raw text extracted from retrieval_context[0]",
        "evidence_type": "raw (stripped of 맥락 설명, 문서 위치, 관련 HS 코드)",
        "comparison_to": "case1_perfect_evidence (contextual evidence)",
        "expected_finding": "Raw < Contextual (context helps LLM understand evidence)",
        "total_cases": len(results),
        "success_count": success_count,
        "hs_code_cases": len(hs_code_cases),
        "hs_code_success": hs_success,
        "deepeval_cases": len(deepeval_cases),
        "deepeval_success": deepeval_success,
        "avg_evidence_reduction_pct": avg_reduction,
        "results": results,
    }


async def main():
    """Run Case 1b ablation study."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ablation Case 1b: Raw Evidence Test (No Context)"
    )
    parser.add_argument("--hs-code-dataset", type=Path, default=HS_CODE_DATASET)
    parser.add_argument("--deepeval-dataset", type=Path, default=DEEPEVAL_DATASET)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "ablation_case1b_results.json"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION CASE 1b: Raw Evidence → LLM Ceiling Test")
    print("=" * 60)
    print("\nEvidence: RAW document text only (no generated context)")
    print("Comparison: Case 1 uses contextual evidence (with metadata)")
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
    results = await run_ablation_case1b(hs_code_cases, deepeval_cases)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Evidence type: {results['evidence_type']}")
    print(f"Avg evidence reduction: {results['avg_evidence_reduction_pct']}%")
    print(f"Total cases: {results['total_cases']}")
    print(f"Success: {results['success_count']}/{results['total_cases']}")
    print(f"  HS Code: {results['hs_code_success']}/{results['hs_code_cases']}")
    print(f"  DeepEval: {results['deepeval_success']}/{results['deepeval_cases']}")
    print(f"\nResults saved to: {args.output}")
    print("\nNote: Accuracy evaluation requires running DeepEval GEval metric.")
    print("Use: python -m evaluation.experiments.run_ablation_eval --case 1b")


if __name__ == "__main__":
    asyncio.run(main())
