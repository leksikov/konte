"""Deep analysis of failures - check if expected HS codes exist in retrieved chunks."""

import json
import re
from pathlib import Path


def extract_hs_codes(text: str) -> set[str]:
    """Extract HS codes from text."""
    patterns = [
        r'\b(\d{4}\.\d{2}(?:\.\d{2})?)\b',
        r'\b제?(\d{4})호\b',
        r'\b(\d{4})(?:\s|$|,|\)|\.)',
    ]
    codes = set()
    for pattern in patterns:
        for match in re.findall(pattern, text):
            code = match.replace('.', '')[:4]
            if code.isdigit() and len(code) == 4:
                codes.add(code)
    return codes


def analyze_failure_deeply(result: dict) -> dict:
    """Analyze if expected codes exist in retrieved context."""
    query = result["input"]
    expected = result["expected_output"]
    actual = result["actual_output"]
    context_chunks = result.get("retrieval_context", [])

    expected_codes = extract_hs_codes(expected)
    actual_codes = extract_hs_codes(actual)

    # Check if expected codes appear in ANY context chunk
    codes_in_context = set()
    chunk_with_expected = []
    for i, chunk in enumerate(context_chunks):
        chunk_codes = extract_hs_codes(chunk)
        codes_in_context.update(chunk_codes)
        matching = expected_codes & chunk_codes
        if matching:
            chunk_with_expected.append((i, matching, chunk[:300]))

    expected_in_context = expected_codes & codes_in_context
    missing_from_context = expected_codes - codes_in_context

    # Determine failure type
    if len(missing_from_context) == len(expected_codes):
        failure_type = "RETRIEVAL_FAILURE"  # Expected codes not in ANY chunk
    elif len(missing_from_context) > 0:
        failure_type = "PARTIAL_RETRIEVAL"  # Some codes missing from context
    elif expected_in_context and not (expected_codes & actual_codes):
        failure_type = "LLM_GENERATION_ERROR"  # Codes in context but LLM didn't include them
    else:
        failure_type = "UNKNOWN"

    return {
        "query": query,
        "expected_codes": sorted(expected_codes),
        "actual_codes": sorted(actual_codes),
        "codes_in_context": sorted(codes_in_context),
        "expected_in_context": sorted(expected_in_context),
        "missing_from_context": sorted(missing_from_context),
        "failure_type": failure_type,
        "num_chunks": len(context_chunks),
        "chunks_with_expected": [(i, list(c), t[:200]) for i, c, t in chunk_with_expected],
        "expected_preview": expected[:400],
        "actual_preview": actual[:400],
    }


def main():
    results_dir = Path("evaluation/experiments/results")

    # Analyze both methods
    for method in ["binary", "batch"]:
        path = results_dir / f"llm_rerank_{method}.json"
        with open(path) as f:
            data = json.load(f)

        results = data["results"]
        print(f"\n{'='*80}")
        print(f"DEEP ANALYSIS: {method.upper()} METHOD")
        print(f"{'='*80}")

        failures_by_type = {
            "RETRIEVAL_FAILURE": [],
            "PARTIAL_RETRIEVAL": [],
            "LLM_GENERATION_ERROR": [],
            "UNKNOWN": [],
        }

        for r in results:
            if r.get("status") != "success":
                continue

            expected_codes = extract_hs_codes(r["expected_output"])
            actual_codes = extract_hs_codes(r["actual_output"])

            # Check if this is a failure
            if expected_codes and not (expected_codes & actual_codes):
                analysis = analyze_failure_deeply(r)
                failures_by_type[analysis["failure_type"]].append(analysis)

        print(f"\n--- Failure Type Breakdown ---")
        for ftype, failures in failures_by_type.items():
            print(f"  {ftype}: {len(failures)}")

        print(f"\n--- RETRIEVAL FAILURES (codes not in ANY chunk) ---")
        for f in failures_by_type["RETRIEVAL_FAILURE"][:5]:
            print(f"\n  Query: {f['query'][:80]}...")
            print(f"  Expected: {f['expected_codes']}")
            print(f"  In Context: {f['codes_in_context'][:10]}...")  # First 10
            print(f"  Missing: {f['missing_from_context']}")
            print(f"  Num chunks: {f['num_chunks']}")

        print(f"\n--- LLM GENERATION ERRORS (codes in context but not in answer) ---")
        for f in failures_by_type["LLM_GENERATION_ERROR"][:5]:
            print(f"\n  Query: {f['query'][:80]}...")
            print(f"  Expected: {f['expected_codes']}")
            print(f"  Actual: {f['actual_codes']}")
            print(f"  In Context: {f['expected_in_context']}")
            if f["chunks_with_expected"]:
                idx, codes, text = f["chunks_with_expected"][0]
                print(f"  Chunk #{idx} with expected codes: {text[:150]}...")

        # Check for BAD TEST DATA pattern
        print(f"\n\n--- BAD TEST DATA ANALYSIS ---")
        bad_test_patterns = []
        for f in failures_by_type["RETRIEVAL_FAILURE"]:
            # Check if expected mentions codes NOT in the expected answer's logic
            exp = f["expected_preview"]
            # Pattern: expected output mentions a code the answer should NOT have
            if "제외" in exp or "분류되지 않" in exp or "아닌" in exp:
                bad_test_patterns.append(f)

        print(f"  Potential bad test cases (expected mentions exclusions): {len(bad_test_patterns)}")
        for f in bad_test_patterns[:3]:
            print(f"\n  Query: {f['query'][:60]}...")
            print(f"  Expected codes: {f['expected_codes']}")
            print(f"  Expected (preview): {f['expected_preview'][:200]}...")

    # Summary recommendations
    print(f"\n\n{'='*80}")
    print("PATH TO 99% ACCURACY")
    print(f"{'='*80}")

    print("""
Based on the analysis:

1. RETRIEVAL FAILURES: Expected HS codes don't exist in ANY retrieved chunk.
   - This is NOT a reranking problem - it's a retrieval problem
   - Solution: Increase initial_k from 100 to 200+ candidates
   - Solution: Use query expansion for specific HS code mentions

2. BAD TEST DATA: Some expected outputs contain codes for "what it's NOT"
   - Example: Query asks about X, expected mentions Y (the exclusion)
   - Our HS code matching treats Y as "expected" when it shouldn't be
   - Solution: Fix test data OR use semantic similarity instead of code matching

3. LLM GENERATION ERRORS: Codes exist in context but LLM didn't include them
   - This is the LLM answer generation failing
   - Solution: Improve answer generation prompt to cite specific HS codes

REALISTIC ACCURACY CEILING:
- Current: ~93% (binary), ~90% (batch)
- With fixed test data: ~95-96%
- With query expansion + more chunks: ~97-98%
- 99% requires: Perfect retrieval + perfect LLM generation + perfect test data
""")


if __name__ == "__main__":
    main()
