"""Ablation Case 2: Evidence Injection → Answer Generation with Guaranteed Evidence.

Tests Konte answer generation when evidence is guaranteed in top-k retrieval.
Expected accuracy: ~97%

This test does real hybrid retrieval (top-k), then injects the retrieval_context
at a RANDOM position, then calls generate_answer().

Evidence source: retrieval_context[0] from test dataset (the actual source chunk).
Injection position: Random within retrieval results.
"""

import asyncio
import json
import random
from pathlib import Path

import structlog
from tqdm import tqdm

from konte import Project
from konte.generator import generate_answer

logger = structlog.get_logger()

# Paths
HS_CODE_DATASET = Path("evaluation/data/synthetic/synthetic_goldens_100.json")
DEEPEVAL_DATASET = Path("evaluation/data/synthetic/deepeval_goldens_korean_100.json")
RESULTS_DIR = Path("evaluation/experiments/results")

# Default project
DEFAULT_PROJECT = "wco_hs_explanatory_notes_korean"

# Sampling
SAMPLE_SIZE = 15
RANDOM_SEED = 42

# Retrieval settings
RETRIEVAL_TOP_K = 15


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


def find_injected_position(results: list) -> int:
    """Find position of injected evidence in results."""
    for i, r in enumerate(results):
        if r.metadata.get("injected", False):
            return i
    return -1


async def evaluate_single_case(
    project: Project,
    question: str,
    expected_output: str,
    evidence: str,
    case_idx: int,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict:
    """Evaluate single test case with evidence injection at random position.

    Args:
        project: Konte project to query.
        question: Input question.
        expected_output: Expected answer (for comparison).
        evidence: Source context chunk (from retrieval_context).
        case_idx: Case index for logging.
        top_k: Number of chunks to retrieve.

    Returns:
        Evaluation result dict.
    """
    try:
        # Do real hybrid retrieval with evidence injection at RANDOM position
        response = project.query(
            query=question,
            mode="hybrid",
            top_k=top_k,
            inject_evidence=evidence,
            inject_position=None,  # None = random position
        )

        # Find where evidence was injected
        injected_position = find_injected_position(response.results)

        # Generate answer with injected evidence at random position
        answer = await generate_answer(
            question=question,
            retrieval_response=response,
            max_chunks=top_k,
        )

        return {
            "case_idx": case_idx,
            "input": question,
            "expected_output": expected_output,
            "evidence_used": evidence[:500] + "..." if len(evidence) > 500 else evidence,
            "actual_output": answer.answer,
            "model": answer.model,
            "total_chunks": len(response.results),
            "injected_position": injected_position,
            "retrieval_context": [r.content[:200] for r in response.results[:3]],
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
            "total_chunks": 0,
            "injected_position": -1,
            "retrieval_context": [],
            "status": "error",
        }


async def run_ablation_case2(
    project: Project,
    hs_code_cases: list[dict],
    deepeval_cases: list[dict],
) -> dict:
    """Run Case 2 ablation study.

    Args:
        project: Konte project.
        hs_code_cases: HS code dataset test cases.
        deepeval_cases: DeepEval dataset test cases.

    Returns:
        Complete results dict.
    """
    results = []
    injection_positions = []

    # Process HS code cases
    print("\n=== Processing HS Code Dataset ===")
    for i, tc in enumerate(tqdm(hs_code_cases, desc="HS Code")):
        evidence = get_evidence_from_case(tc)
        result = await evaluate_single_case(
            project=project,
            question=tc["input"],
            expected_output=tc["expected_output"],
            evidence=evidence,
            case_idx=i,
        )
        result["dataset"] = "hs_code"
        results.append(result)
        if result["injected_position"] >= 0:
            injection_positions.append(result["injected_position"])

    # Process DeepEval cases
    print("\n=== Processing DeepEval Dataset ===")
    for i, tc in enumerate(tqdm(deepeval_cases, desc="DeepEval")):
        evidence = get_evidence_from_case(tc)
        result = await evaluate_single_case(
            project=project,
            question=tc["input"],
            expected_output=tc["expected_output"],
            evidence=evidence,
            case_idx=len(hs_code_cases) + i,
        )
        result["dataset"] = "deepeval"
        results.append(result)
        if result["injected_position"] >= 0:
            injection_positions.append(result["injected_position"])

    # Summary statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    hs_success = sum(1 for r in results if r["status"] == "success" and r["dataset"] == "hs_code")
    deepeval_success = sum(1 for r in results if r["status"] == "success" and r["dataset"] == "deepeval")
    avg_position = sum(injection_positions) / len(injection_positions) if injection_positions else 0

    return {
        "ablation_case": "case2_evidence_injection",
        "description": "Answer generation with evidence injected at RANDOM position",
        "evidence_source": "retrieval_context[0]",
        "injection_method": "random_position",
        "expected_accuracy": "~97%",
        "project": project.config.name,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "total_cases": len(results),
        "success_count": success_count,
        "injection_positions": injection_positions,
        "avg_injection_position": avg_position,
        "hs_code_cases": len(hs_code_cases),
        "hs_code_success": hs_success,
        "deepeval_cases": len(deepeval_cases),
        "deepeval_success": deepeval_success,
        "results": results,
    }


async def main():
    """Run Case 2 ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Case 2: Evidence Injection Test")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--hs-code-dataset", type=Path, default=HS_CODE_DATASET)
    parser.add_argument("--deepeval-dataset", type=Path, default=DEEPEVAL_DATASET)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ablation_case2_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION CASE 2: Evidence Injection → Answer Generation Test")
    print("=" * 60)
    print(f"\nEvidence source: retrieval_context[0] (actual source chunks)")
    print(f"Injection method: RANDOM position within retrieval results")

    # Load project
    print(f"\nLoading project: {args.project}")
    project = Project.open(args.project)
    print(f"Project loaded with {len(project._chunks)} chunks")

    print(f"\nLoading datasets...")
    print(f"  HS Code: {args.hs_code_dataset}")
    print(f"  DeepEval: {args.deepeval_dataset}")
    print(f"  Sample size: {args.sample_size} per dataset")
    print(f"  Retrieval top-k: {args.top_k}")

    # Load and sample test cases
    hs_code_cases = load_test_cases(args.hs_code_dataset, args.sample_size)
    deepeval_cases = load_test_cases(args.deepeval_dataset, args.sample_size)

    print(f"\nLoaded {len(hs_code_cases)} HS code cases")
    print(f"Loaded {len(deepeval_cases)} DeepEval cases")
    print(f"Total: {len(hs_code_cases) + len(deepeval_cases)} cases")

    # Run evaluation
    results = await run_ablation_case2(project, hs_code_cases, deepeval_cases)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Project: {results['project']}")
    print(f"Injection method: {results['injection_method']}")
    print(f"Total cases: {results['total_cases']}")
    print(f"Success: {results['success_count']}/{results['total_cases']}")
    print(f"Average injection position: {results['avg_injection_position']:.1f}")
    print(f"  HS Code: {results['hs_code_success']}/{results['hs_code_cases']}")
    print(f"  DeepEval: {results['deepeval_success']}/{results['deepeval_cases']}")
    print(f"\nResults saved to: {args.output}")
    print("\nNote: Accuracy evaluation requires running DeepEval GEval metric.")
    print("Use run_deepeval_full.py to evaluate actual vs expected outputs.")


if __name__ == "__main__":
    asyncio.run(main())
