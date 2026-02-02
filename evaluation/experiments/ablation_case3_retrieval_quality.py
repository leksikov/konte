"""Ablation Case 3: Retrieval Quality Check.

Measures how often evidence appears in top-k retrieved chunks.
Metric: Hit rate (% of cases where evidence found in top-k)

This test does real hybrid retrieval (top-k=50) without any injection,
then checks if the retrieval_context text appears in any retrieved chunk
using multiple matching strategies.

Evidence source: retrieval_context[0] from test dataset (the actual source chunk).
"""

import json
import random
import re
from pathlib import Path

import structlog
from tqdm import tqdm

from konte import Project

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

# Retrieval settings - expanded to 50 for better coverage
RETRIEVAL_TOP_K = 50


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


def extract_hs_codes(text: str) -> list[str]:
    """Extract HS codes from text.

    Matches patterns like:
    - 8301.20
    - 제8301호
    - 제2523.21호
    - 2523
    - HS 코드 7112.30
    """
    patterns = [
        r'\d{4}\.\d{2}',  # 8301.20
        r'\d{4}',  # 8301
        r'제(\d{4})호',  # 제8301호
        r'제(\d{4}\.\d{2})호',  # 제8301.20호
    ]

    codes = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Normalize: remove "제" prefix and "호" suffix
            code = match.strip()
            codes.add(code)

    return list(codes)


def check_evidence_in_chunks(
    evidence: str,
    chunks: list,
) -> dict:
    """Check if evidence appears in retrieved chunks.

    Uses multiple matching strategies:
    1. Substring match (key sentences from evidence)
    2. HS code match
    3. Key phrase overlap

    Args:
        evidence: Source context chunk (from retrieval_context).
        chunks: Retrieved chunks (RetrievalResult objects).

    Returns:
        Dict with matching details.
    """
    evidence_lower = evidence.lower()

    # Extract HS codes from evidence
    evidence_hs_codes = extract_hs_codes(evidence)

    # Track matches
    sentence_match = False
    hs_code_match = False
    matched_chunk_indices = []
    match_details = []

    for i, chunk in enumerate(chunks):
        chunk_text = f"{chunk.context} {chunk.content}".lower()
        chunk_hs_codes = extract_hs_codes(chunk.content)

        # Strategy 1: Sentence match
        # Check if key sentences from evidence appear in chunk
        sentences = evidence.split('.')
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30 and sent.lower() in chunk_text:
                sentence_match = True
                if i not in matched_chunk_indices:
                    matched_chunk_indices.append(i)
                match_details.append({
                    "chunk_idx": i,
                    "match_type": "sentence",
                    "matched_text": sent[:100],
                })
                break

        # Strategy 2: HS code match
        for code in evidence_hs_codes:
            if code in chunk.content or code in chunk.context:
                hs_code_match = True
                if i not in matched_chunk_indices:
                    matched_chunk_indices.append(i)
                match_details.append({
                    "chunk_idx": i,
                    "match_type": "hs_code",
                    "matched_code": code,
                })
                break

    return {
        "sentence_match": sentence_match,
        "hs_code_match": hs_code_match,
        "any_match": sentence_match or hs_code_match,
        "matched_chunk_indices": matched_chunk_indices,
        "evidence_hs_codes": evidence_hs_codes,
        "match_details": match_details[:5],  # Limit details
    }


def evaluate_single_case(
    project: Project,
    question: str,
    expected_output: str,
    evidence: str,
    case_idx: int,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict:
    """Evaluate single test case for retrieval quality.

    Args:
        project: Konte project to query.
        question: Input question.
        expected_output: Expected answer (for reference).
        evidence: Source context chunk (from retrieval_context).
        case_idx: Case index for logging.
        top_k: Number of chunks to retrieve.

    Returns:
        Evaluation result dict.
    """
    try:
        # Do real hybrid retrieval (no injection)
        response = project.query(
            query=question,
            mode="hybrid",
            top_k=top_k,
        )

        # Check if evidence is in retrieved chunks
        match_result = check_evidence_in_chunks(evidence, response.results)

        return {
            "case_idx": case_idx,
            "input": question,
            "expected_output": expected_output[:200] + "..." if len(expected_output) > 200 else expected_output,
            "evidence_preview": evidence[:300] + "..." if len(evidence) > 300 else evidence,
            "total_chunks": len(response.results),
            "top_score": response.top_score,
            "sentence_match": match_result["sentence_match"],
            "hs_code_match": match_result["hs_code_match"],
            "any_match": match_result["any_match"],
            "matched_chunk_indices": match_result["matched_chunk_indices"],
            "evidence_hs_codes": match_result["evidence_hs_codes"],
            "match_details": match_result["match_details"],
            "top_chunks_preview": [r.content[:150] for r in response.results[:3]],
            "status": "success",
        }

    except Exception as e:
        logger.error("case_failed", case_idx=case_idx, error=str(e))
        return {
            "case_idx": case_idx,
            "input": question,
            "expected_output": expected_output[:200],
            "evidence_preview": evidence[:300] if evidence else "",
            "total_chunks": 0,
            "top_score": 0.0,
            "sentence_match": False,
            "hs_code_match": False,
            "any_match": False,
            "matched_chunk_indices": [],
            "evidence_hs_codes": [],
            "match_details": [],
            "top_chunks_preview": [],
            "status": "error",
        }


def run_ablation_case3(
    project: Project,
    hs_code_cases: list[dict],
    deepeval_cases: list[dict],
    top_k: int = RETRIEVAL_TOP_K,
) -> dict:
    """Run Case 3 ablation study.

    Args:
        project: Konte project.
        hs_code_cases: HS code dataset test cases.
        deepeval_cases: DeepEval dataset test cases.
        top_k: Number of chunks to retrieve.

    Returns:
        Complete results dict.
    """
    results = []

    # Process HS code cases
    print("\n=== Processing HS Code Dataset ===")
    for i, tc in enumerate(tqdm(hs_code_cases, desc="HS Code")):
        evidence = get_evidence_from_case(tc)
        result = evaluate_single_case(
            project=project,
            question=tc["input"],
            expected_output=tc["expected_output"],
            evidence=evidence,
            case_idx=i,
            top_k=top_k,
        )
        result["dataset"] = "hs_code"
        results.append(result)

    # Process DeepEval cases
    print("\n=== Processing DeepEval Dataset ===")
    for i, tc in enumerate(tqdm(deepeval_cases, desc="DeepEval")):
        evidence = get_evidence_from_case(tc)
        result = evaluate_single_case(
            project=project,
            question=tc["input"],
            expected_output=tc["expected_output"],
            evidence=evidence,
            case_idx=len(hs_code_cases) + i,
            top_k=top_k,
        )
        result["dataset"] = "deepeval"
        results.append(result)

    # Summary statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    sentence_match_count = sum(1 for r in results if r.get("sentence_match", False))
    hs_code_match_count = sum(1 for r in results if r.get("hs_code_match", False))
    any_match_count = sum(1 for r in results if r.get("any_match", False))

    # By dataset
    hs_any_match = sum(1 for r in results if r.get("any_match", False) and r["dataset"] == "hs_code")
    deepeval_any_match = sum(1 for r in results if r.get("any_match", False) and r["dataset"] == "deepeval")

    # Calculate hit rates
    total = len(results)
    hit_rate = any_match_count / total if total > 0 else 0.0
    sentence_hit_rate = sentence_match_count / total if total > 0 else 0.0
    hs_code_hit_rate = hs_code_match_count / total if total > 0 else 0.0

    return {
        "ablation_case": "case3_retrieval_quality",
        "description": "Retrieval quality check - how often evidence (retrieval_context) appears in top-k",
        "evidence_source": "retrieval_context[0]",
        "project": project.config.name,
        "retrieval_top_k": top_k,
        "total_cases": len(results),
        "success_count": success_count,
        # Hit rates
        "hit_rate": hit_rate,
        "sentence_hit_rate": sentence_hit_rate,
        "hs_code_hit_rate": hs_code_hit_rate,
        # Counts
        "any_match_count": any_match_count,
        "sentence_match_count": sentence_match_count,
        "hs_code_match_count": hs_code_match_count,
        # By dataset
        "hs_code_cases": len(hs_code_cases),
        "hs_code_any_match": hs_any_match,
        "hs_code_hit_rate": hs_any_match / len(hs_code_cases) if hs_code_cases else 0.0,
        "deepeval_cases": len(deepeval_cases),
        "deepeval_any_match": deepeval_any_match,
        "deepeval_hit_rate": deepeval_any_match / len(deepeval_cases) if deepeval_cases else 0.0,
        "results": results,
    }


def main():
    """Run Case 3 ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Case 3: Retrieval Quality Check")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--hs-code-dataset", type=Path, default=HS_CODE_DATASET)
    parser.add_argument("--deepeval-dataset", type=Path, default=DEEPEVAL_DATASET)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ablation_case3_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION CASE 3: Retrieval Quality Check")
    print("=" * 60)
    print(f"\nEvidence source: retrieval_context[0] (actual source chunks)")

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
    results = run_ablation_case3(project, hs_code_cases, deepeval_cases, top_k=args.top_k)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Project: {results['project']}")
    print(f"Retrieval top-k: {results['retrieval_top_k']}")
    print(f"Total cases: {results['total_cases']}")
    print(f"\n--- Hit Rates ---")
    print(f"Overall Hit Rate: {results['hit_rate']:.1%} ({results['any_match_count']}/{results['total_cases']})")
    print(f"  Sentence Match: {results['sentence_hit_rate']:.1%} ({results['sentence_match_count']}/{results['total_cases']})")
    print(f"  HS Code Match: {results['hs_code_hit_rate']:.1%} ({results['hs_code_match_count']}/{results['total_cases']})")
    print(f"\n--- By Dataset ---")
    print(f"HS Code: {results['hs_code_hit_rate']:.1%} ({results['hs_code_any_match']}/{results['hs_code_cases']})")
    print(f"DeepEval: {results['deepeval_hit_rate']:.1%} ({results['deepeval_any_match']}/{results['deepeval_cases']})")
    print(f"\nResults saved to: {args.output}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if results['hit_rate'] < 0.7:
        print("Low hit rate suggests RETRIEVAL is the bottleneck.")
        print("Consider: improving embeddings, adding more chunks, tuning fusion weights.")
    elif results['hit_rate'] >= 0.9:
        print("High hit rate suggests retrieval is working well.")
        print("If overall accuracy is still low, LLM reasoning may be the bottleneck.")
    else:
        print("Moderate hit rate suggests room for improvement in retrieval.")


if __name__ == "__main__":
    main()
