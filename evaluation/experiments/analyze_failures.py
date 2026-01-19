"""Analyze misclassifications from LLM reranking results.

Identifies patterns in failed cases and proposes improvements.
"""

import json
import re
from pathlib import Path
from collections import Counter


def extract_hs_codes(text: str) -> set[str]:
    """Extract HS codes from text (4-10 digit codes like 8540, 8540.11)."""
    # Match patterns: 4 digits, 4.2 digits, 6 digits, etc.
    patterns = [
        r'\b(\d{4}\.\d{2}(?:\.\d{2})?)\b',  # 8540.11 or 8540.11.00
        r'\b제?(\d{4})호\b',                 # 제8540호
        r'\b(\d{4})(?:\s|$|,|\))',           # 8540 followed by space/end/comma
    ]
    codes = set()
    for pattern in patterns:
        for match in re.findall(pattern, text):
            code = match.replace('.', '')[:4]  # Normalize to 4 digits
            if code.isdigit():
                codes.add(code)
    return codes


def analyze_single_result(result: dict) -> dict:
    """Analyze a single test case result."""
    query = result["input"]
    expected = result["expected_output"]
    actual = result["actual_output"]

    expected_codes = extract_hs_codes(expected)
    actual_codes = extract_hs_codes(actual)

    # Check if expected codes appear in actual
    matched_codes = expected_codes & actual_codes
    missing_codes = expected_codes - actual_codes
    extra_codes = actual_codes - expected_codes

    is_pass = len(expected_codes) == 0 or len(matched_codes) > 0

    return {
        "query": query,
        "expected_codes": list(expected_codes),
        "actual_codes": list(actual_codes),
        "matched_codes": list(matched_codes),
        "missing_codes": list(missing_codes),
        "extra_codes": list(extra_codes),
        "is_pass": is_pass,
        "expected_output": expected[:500],
        "actual_output": actual[:500],
        "retrieval_context_count": len(result.get("retrieval_context", [])),
        "retrieval_context_preview": [c[:200] for c in result.get("retrieval_context", [])[:3]],
    }


def analyze_results_file(path: Path) -> dict:
    """Analyze all results in a file."""
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    method = data["method"]

    analyses = []
    failures = []

    for r in results:
        if r.get("status") != "success":
            continue

        analysis = analyze_single_result(r)
        analyses.append(analysis)

        if not analysis["is_pass"]:
            failures.append(analysis)

    # Categorize failures
    failure_categories = {
        "no_expected_codes_found": [],
        "wrong_codes": [],
        "missing_some_codes": [],
        "insufficient_context": [],
    }

    for f in failures:
        if len(f["expected_codes"]) > 0 and len(f["matched_codes"]) == 0:
            if len(f["actual_codes"]) > 0:
                failure_categories["wrong_codes"].append(f)
            else:
                failure_categories["no_expected_codes_found"].append(f)
        elif len(f["missing_codes"]) > 0:
            failure_categories["missing_some_codes"].append(f)

        if f["retrieval_context_count"] < 5:
            failure_categories["insufficient_context"].append(f)

    # Stats
    total = len(analyses)
    passed = sum(1 for a in analyses if a["is_pass"])

    return {
        "method": method,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total if total > 0 else 0,
        "failure_categories": {k: len(v) for k, v in failure_categories.items()},
        "failures": failures,
        "failure_details": failure_categories,
    }


def print_failure_analysis(analysis: dict):
    """Print detailed failure analysis."""
    print(f"\n{'='*80}")
    print(f"Method: {analysis['method']}")
    print(f"{'='*80}")
    print(f"Total: {analysis['total']}")
    print(f"Passed: {analysis['passed']} ({analysis['accuracy']:.1%})")
    print(f"Failed: {analysis['failed']}")

    print(f"\n--- Failure Categories ---")
    for cat, count in analysis["failure_categories"].items():
        print(f"  {cat}: {count}")

    print(f"\n--- Detailed Failures ---")
    for i, f in enumerate(analysis["failures"][:15], 1):  # Show first 15
        print(f"\n[Failure {i}]")
        print(f"  Query: {f['query'][:100]}...")
        print(f"  Expected codes: {f['expected_codes']}")
        print(f"  Actual codes: {f['actual_codes']}")
        print(f"  Missing: {f['missing_codes']}")
        print(f"  Context chunks: {f['retrieval_context_count']}")
        print(f"  Expected (preview): {f['expected_output'][:150]}...")
        print(f"  Actual (preview): {f['actual_output'][:150]}...")

        if f["retrieval_context_preview"]:
            print(f"  Context #1: {f['retrieval_context_preview'][0][:100]}...")


def compare_methods(binary_analysis: dict, batch_analysis: dict):
    """Compare failures between binary and batch methods."""
    print(f"\n{'='*80}")
    print("COMPARISON: Binary Filter vs Batch Scoring")
    print(f"{'='*80}")

    binary_failed_queries = {f["query"] for f in binary_analysis["failures"]}
    batch_failed_queries = {f["query"] for f in batch_analysis["failures"]}

    only_binary_failed = binary_failed_queries - batch_failed_queries
    only_batch_failed = batch_failed_queries - binary_failed_queries
    both_failed = binary_failed_queries & batch_failed_queries

    print(f"\nBinary accuracy: {binary_analysis['accuracy']:.1%}")
    print(f"Batch accuracy: {batch_analysis['accuracy']:.1%}")
    print(f"\nFailures unique to Binary: {len(only_binary_failed)}")
    print(f"Failures unique to Batch: {len(only_batch_failed)}")
    print(f"Both failed: {len(both_failed)}")

    if both_failed:
        print(f"\n--- Queries that BOTH methods failed ---")
        for q in list(both_failed)[:5]:
            print(f"  - {q[:80]}...")

    if only_binary_failed:
        print(f"\n--- Queries ONLY Binary failed (Batch succeeded) ---")
        for q in list(only_binary_failed)[:5]:
            print(f"  - {q[:80]}...")


def suggest_improvements(analysis: dict):
    """Suggest improvements based on failure patterns."""
    print(f"\n{'='*80}")
    print("SUGGESTED IMPROVEMENTS")
    print(f"{'='*80}")

    cats = analysis["failure_categories"]
    failures = analysis["failures"]

    suggestions = []

    if cats.get("insufficient_context", 0) > 0:
        suggestions.append(
            "1. INCREASE CONTEXT: Some failures have <5 context chunks. "
            "Consider keeping more chunks after filtering (top-20 instead of top-15)."
        )

    if cats.get("wrong_codes", 0) > 0:
        suggestions.append(
            "2. IMPROVE BINARY FILTER PROMPT: LLM is selecting wrong chunks. "
            "Add domain-specific instructions about HS code relevance."
        )

    if cats.get("no_expected_codes_found", 0) > 0:
        suggestions.append(
            "3. CHECK CHUNK CONTENT: Expected codes may not be in any retrieved chunks. "
            "This indicates retrieval failure, not reranking failure."
        )

    # Check for pattern: queries about specific subheadings
    subheading_failures = [f for f in failures if "소호" in f["query"] or "하위" in f["query"]]
    if subheading_failures:
        suggestions.append(
            f"4. SUBHEADING QUERIES: {len(subheading_failures)} failures involve subheading (소호) questions. "
            "These require very specific chunk matching. Consider query expansion."
        )

    # Check for pattern: queries about classification rules
    rule_failures = [f for f in failures if "분류" in f["query"] and len(f["expected_codes"]) > 1]
    if rule_failures:
        suggestions.append(
            f"5. MULTI-CODE QUERIES: {len(rule_failures)} failures expect multiple HS codes. "
            "May need more context chunks to cover all relevant codes."
        )

    for s in suggestions:
        print(f"\n{s}")

    return suggestions


def main():
    results_dir = Path("evaluation/experiments/results")

    binary_path = results_dir / "llm_rerank_binary.json"
    batch_path = results_dir / "llm_rerank_batch.json"

    print("Analyzing Binary Filter results...")
    binary_analysis = analyze_results_file(binary_path)
    print_failure_analysis(binary_analysis)

    print("\n\nAnalyzing Batch Scoring results...")
    batch_analysis = analyze_results_file(batch_path)
    print_failure_analysis(batch_analysis)

    compare_methods(binary_analysis, batch_analysis)

    # Use batch (better) for improvement suggestions
    suggest_improvements(batch_analysis)

    # Save analysis
    output = {
        "binary": {
            "accuracy": binary_analysis["accuracy"],
            "failed": binary_analysis["failed"],
            "failure_categories": binary_analysis["failure_categories"],
        },
        "batch": {
            "accuracy": batch_analysis["accuracy"],
            "failed": batch_analysis["failed"],
            "failure_categories": batch_analysis["failure_categories"],
        },
        "failures_binary": binary_analysis["failures"],
        "failures_batch": batch_analysis["failures"],
    }

    output_path = results_dir / "failure_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
