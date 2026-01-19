"""Analyze the 12 failures in Contextual RAG v2 (90% accuracy).

Categorize failures by root cause:
1. Retrieval failure - didn't retrieve relevant chunks
2. LLM hallucination - LLM generated wrong HS code despite having correct info
3. Test case issue - expected answer is incorrect
4. Judge metric issue - DeepEval scored incorrectly
"""

import json
from pathlib import Path

# Paths
DEEPEVAL_RESULTS = Path("evaluation/experiments/results/binary_v2_improved_deepeval_correctness.json")
LLM_RERANK_RESULTS = Path("evaluation/experiments/results/llm_rerank_binary_v2_improved.json")
TEST_CASES = Path("evaluation/data/synthetic/synthetic_goldens_korean_v2.json")


def load_data():
    """Load all relevant data."""
    with open(DEEPEVAL_RESULTS) as f:
        deepeval = json.load(f)
    with open(LLM_RERANK_RESULTS) as f:
        rerank = json.load(f)
    with open(TEST_CASES) as f:
        test_cases = json.load(f)
    return deepeval, rerank, test_cases


def analyze_single_failure(failure: dict, rerank_result: dict, test_case: dict):
    """Analyze a single failure case."""
    query = failure['input']
    score = failure['score']
    reason = failure['reason']

    # Get the actual and expected outputs
    expected = test_case.get('expected_output', '')
    actual = rerank_result.get('actual_output', '')
    retrieval_context = rerank_result.get('retrieval_context', [])

    print(f"\nQuery: {query}")
    print(f"Score: {score}")
    print(f"\nExpected output (from test case):")
    print(f"  {expected[:500]}...")
    print(f"\nActual output (from LLM):")
    print(f"  {actual[:500]}...")
    print(f"\nRetrieval context (first 2 chunks):")
    for i, chunk in enumerate(retrieval_context[:2]):
        print(f"  [{i+1}] {chunk[:200]}...")
    print(f"\nDeepEval reason: {reason}")

    return {
        "query": query,
        "score": score,
        "expected": expected,
        "actual": actual,
        "context_preview": [c[:200] for c in retrieval_context[:3]],
        "reason": reason,
    }


def main():
    """Analyze all failures."""
    deepeval, rerank, test_cases = load_data()

    # Get failed cases from deepeval
    failures = [r for r in deepeval['detailed_results'] if not r['passed']]

    # Map test cases by input
    tc_map = {tc['input']: tc for tc in test_cases}

    # Map rerank results by input
    rerank_map = {r['input']: r for r in rerank['results']}

    print("="*80)
    print(f"DETAILED FAILURE ANALYSIS: {len(failures)} failures")
    print("="*80)

    analyses = []
    for i, f in enumerate(failures):
        print(f"\n{'='*80}")
        print(f"FAILURE {i+1}/{len(failures)}")
        print("="*80)

        query = f['input']
        tc = tc_map.get(query, {})
        rr = rerank_map.get(query, {})

        analysis = analyze_single_failure(f, rr, tc)
        analyses.append(analysis)

    # Summary categorization
    print("\n" + "="*80)
    print("FAILURE CATEGORIZATION")
    print("="*80)

    # Manual categorization based on the analysis
    categories = {
        "retrieval_failure": [],  # Didn't retrieve relevant chunks
        "llm_hallucination": [],  # LLM gave wrong answer despite correct context
        "test_case_issue": [],    # Expected answer might be wrong
        "judge_issue": [],        # DeepEval scored incorrectly
        "ambiguous": [],          # Multiple valid interpretations
    }

    print("""
Based on the analysis above, failures should be manually categorized into:

1. RETRIEVAL FAILURE: Context doesn't contain the answer
   - Check if retrieval_context contains relevant HS codes/info

2. LLM HALLUCINATION: Context has correct info, LLM output is wrong
   - Context has HS code X, but LLM says HS code Y

3. TEST CASE ISSUE: Expected answer from test generation is incorrect
   - When source document says X, but expected output says Y

4. JUDGE ISSUE: DeepEval metric scored incorrectly
   - When actual matches expected but scored as fail

5. AMBIGUOUS: Multiple valid answers, question is unclear
   - Domain-specific edge cases
""")


if __name__ == "__main__":
    main()
