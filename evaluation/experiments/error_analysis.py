"""
Error Analysis for Contextual RAG Failures

Analyzes failing cases from evaluation results to identify:
1. Failure type (retrieval miss, context quality, LLM hallucination, etc.)
2. Whether evidence was present in retrieved chunks
3. Recommendations for improvement

Goal: Identify 2+ recoverable failures to reach 96% accuracy from current 94%.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


FailureType = Literal[
    "retrieval_miss",      # Evidence not in top-k chunks
    "context_quality",     # Evidence present but poorly contextualized
    "llm_hallucination",   # LLM ignored or misinterpreted evidence
    "ambiguous_gt",        # Expected answer is questionable/ambiguous
    "knowledge_gap",       # Information not in corpus
    "hs_code_confusion",   # Similar HS codes confused (e.g., 5511.10 vs 5511.20)
]


@dataclass
class FailureAnalysis:
    """Analysis result for a failing case."""
    query: str
    expected: str
    actual: str
    score: float
    reason: str
    failure_type: FailureType
    evidence_in_chunks: bool
    recoverable: bool
    recommendation: str


def load_json(path: Path) -> dict:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_hs_code_failures() -> list[FailureAnalysis]:
    """Analyze failures from HS code evaluation dataset."""
    results_path = Path("evaluation/experiments/results/binary_100_deepeval_correctness.json")
    retrieval_path = Path("evaluation/experiments/results/llm_rerank_binary_100.json")

    eval_results = load_json(results_path)
    retrieval_data = load_json(retrieval_path)

    # Build retrieval lookup
    retrieval_lookup = {r["input"]: r for r in retrieval_data["results"]}

    failures = []
    for result in eval_results["detailed_results"]:
        if not result["passed"]:
            query = result["input"]
            retrieval = retrieval_lookup.get(query, {})

            analysis = analyze_single_case(
                query=query,
                expected=result.get("expected_output", retrieval.get("expected_output", "")),
                actual=retrieval.get("actual_output", ""),
                score=result["score"],
                reason=result["reason"],
                chunks=retrieval.get("retrieval_context", []),
                dataset="hs_code"
            )
            failures.append(analysis)

    return failures


def analyze_diverse_failures() -> list[FailureAnalysis]:
    """Analyze failures from diverse evaluation dataset."""
    results_path = Path("evaluation/experiments/results/binary_deepeval_diverse_deepeval_correctness.json")
    retrieval_path = Path("evaluation/experiments/results/llm_rerank_binary_deepeval_diverse.json")

    eval_results = load_json(results_path)
    retrieval_data = load_json(retrieval_path)

    # Build retrieval lookup
    retrieval_lookup = {r["input"]: r for r in retrieval_data["results"]}

    failures = []
    for result in eval_results["detailed_results"]:
        if not result["passed"]:
            query = result["input"]
            retrieval = retrieval_lookup.get(query, {})

            analysis = analyze_single_case(
                query=query,
                expected=result.get("expected_output", retrieval.get("expected_output", "")),
                actual=retrieval.get("actual_output", ""),
                score=result["score"],
                reason=result["reason"],
                chunks=retrieval.get("retrieval_context", []),
                dataset="diverse"
            )
            failures.append(analysis)

    return failures


def analyze_single_case(
    query: str,
    expected: str,
    actual: str,
    score: float,
    reason: str,
    chunks: list[str],
    dataset: str
) -> FailureAnalysis:
    """Analyze a single failing case to determine failure type."""

    # Check if key evidence from expected answer appears in chunks
    chunks_text = " ".join(chunks)

    # Extract HS codes from expected output
    import re
    expected_codes = re.findall(r'\d{4}\.\d{2}', expected)
    actual_codes = re.findall(r'\d{4}\.\d{2}', actual)

    # Check evidence presence
    evidence_found = False
    for code in expected_codes:
        if code in chunks_text:
            evidence_found = True
            break

    # Determine failure type
    failure_type: FailureType = "knowledge_gap"
    recoverable = False
    recommendation = ""

    if dataset == "hs_code":
        # HS code specific analysis
        if expected_codes and actual_codes:
            exp_prefix = expected_codes[0][:4] if expected_codes else ""
            act_prefix = actual_codes[0][:4] if actual_codes else ""

            if exp_prefix == act_prefix:
                # Same heading, different subheading - likely HS code confusion
                failure_type = "hs_code_confusion"
                recoverable = True
                recommendation = f"LLM confused similar codes: expected {expected_codes[0]}, got {actual_codes[0]}. " \
                               f"Improve context prompt to emphasize subheading distinctions or add few-shot examples."
            elif evidence_found:
                failure_type = "llm_hallucination"
                recoverable = True
                recommendation = "Evidence present but LLM selected wrong code. " \
                               "Consider adding chain-of-thought reasoning or verification step."
            else:
                failure_type = "retrieval_miss"
                recoverable = True
                recommendation = f"Expected HS code {expected_codes[0] if expected_codes else 'N/A'} not found in chunks. " \
                               "Increase initial_k or add hybrid search weight adjustment."
        else:
            if evidence_found:
                failure_type = "context_quality"
                recoverable = True
                recommendation = "Evidence present but context quality insufficient. " \
                               "Review context generation prompt."
            else:
                failure_type = "knowledge_gap"
                recoverable = False
                recommendation = "Information may not exist in corpus. " \
                               "Verify if the expected HS code is covered in source documents."

    else:  # diverse dataset
        if evidence_found:
            if "incorrect" in reason.lower() or "reverses" in reason.lower():
                failure_type = "llm_hallucination"
                recoverable = True
                recommendation = "LLM misinterpreted or reversed facts from evidence. " \
                               "Add verification prompting or chain-of-thought."
            elif "lacks" in reason.lower() or "missing" in reason.lower():
                failure_type = "context_quality"
                recoverable = True
                recommendation = "Context present but key details missing. " \
                               "Improve context generation to capture more specific details."
            else:
                failure_type = "llm_hallucination"
                recoverable = True
                recommendation = "Evidence in chunks but answer incorrect. " \
                               "Review answer generation prompt."
        else:
            if "no specific information" in reason.lower() or "does not contain" in reason.lower():
                failure_type = "knowledge_gap"
                recoverable = False
                recommendation = "Required information not in corpus. " \
                               "Would need additional source documents."
            else:
                failure_type = "retrieval_miss"
                recoverable = True
                recommendation = "Relevant information not retrieved. " \
                               "Adjust retrieval parameters or query expansion."

    return FailureAnalysis(
        query=query,
        expected=expected,
        actual=actual,
        score=score,
        reason=reason,
        failure_type=failure_type,
        evidence_in_chunks=evidence_found,
        recoverable=recoverable,
        recommendation=recommendation
    )


def generate_report(hs_failures: list[FailureAnalysis], diverse_failures: list[FailureAnalysis]) -> str:
    """Generate markdown report of error analysis."""

    # Count by type
    hs_type_counts = {}
    diverse_type_counts = {}

    for f in hs_failures:
        hs_type_counts[f.failure_type] = hs_type_counts.get(f.failure_type, 0) + 1
    for f in diverse_failures:
        diverse_type_counts[f.failure_type] = diverse_type_counts.get(f.failure_type, 0) + 1

    hs_recoverable = sum(1 for f in hs_failures if f.recoverable)
    diverse_recoverable = sum(1 for f in diverse_failures if f.recoverable)

    report = f"""# Error Analysis Report

## Summary

| Dataset | Total Failures | Recoverable | Not Recoverable |
|---------|---------------|-------------|-----------------|
| HS Code | {len(hs_failures)} | {hs_recoverable} | {len(hs_failures) - hs_recoverable} |
| Diverse | {len(diverse_failures)} | {diverse_recoverable} | {len(diverse_failures) - diverse_recoverable} |
| **Total** | **{len(hs_failures) + len(diverse_failures)}** | **{hs_recoverable + diverse_recoverable}** | **{len(hs_failures) + len(diverse_failures) - hs_recoverable - diverse_recoverable}** |

### Failure Type Distribution

**HS Code Dataset:**
"""
    for ftype, count in sorted(hs_type_counts.items()):
        report += f"- {ftype}: {count}\n"

    report += "\n**Diverse Dataset:**\n"
    for ftype, count in sorted(diverse_type_counts.items()):
        report += f"- {ftype}: {count}\n"

    report += "\n---\n\n## HS Code Failures (Detailed)\n"

    for i, f in enumerate(hs_failures, 1):
        report += f"""
### Case {i}: {f.query[:50]}...

| Field | Value |
|-------|-------|
| **Score** | {f.score} |
| **Failure Type** | {f.failure_type} |
| **Evidence in Chunks** | {f.evidence_in_chunks} |
| **Recoverable** | {f.recoverable} |

**Expected:** {f.expected[:200]}...

**Actual:** {f.actual[:200]}...

**Evaluation Reason:** {f.reason[:300]}...

**Recommendation:** {f.recommendation}

---
"""

    report += "\n## Diverse Failures (Detailed)\n"

    for i, f in enumerate(diverse_failures, 1):
        report += f"""
### Case {i}: {f.query[:50]}...

| Field | Value |
|-------|-------|
| **Score** | {f.score} |
| **Failure Type** | {f.failure_type} |
| **Evidence in Chunks** | {f.evidence_in_chunks} |
| **Recoverable** | {f.recoverable} |

**Expected:** {f.expected[:200]}...

**Actual:** {f.actual[:200]}...

**Evaluation Reason:** {f.reason[:300]}...

**Recommendation:** {f.recommendation}

---
"""

    # Recommendations summary
    report += """
## Recommendations to Reach 96% Accuracy

Based on the analysis, here are the prioritized recommendations:

### High Priority (Quick Wins)

"""
    # Collect unique recommendations for recoverable failures
    hs_recs = set(f.recommendation for f in hs_failures if f.recoverable)
    diverse_recs = set(f.recommendation for f in diverse_failures if f.recoverable)

    all_recs = hs_recs.union(diverse_recs)
    for i, rec in enumerate(all_recs, 1):
        report += f"{i}. {rec}\n\n"

    report += f"""
### Expected Impact

- **HS Code Dataset:** {hs_recoverable}/{len(hs_failures)} failures are recoverable
  - Current: 94% → Potential: {94 + (hs_recoverable * 1)}% (if all recoverable fixed)

- **Diverse Dataset:** {diverse_recoverable}/{len(diverse_failures)} failures are recoverable
  - Current: 94% → Potential: {94 + (diverse_recoverable * 1)}% (if all recoverable fixed)

### Key Patterns Identified

1. **HS Code Confusion**: Similar subheadings within same heading cause confusion
2. **Evidence Present but Misinterpreted**: LLM sometimes reverses or misunderstands facts
3. **Knowledge Gap**: Some questions require information not in the corpus

### Next Steps

1. Fix top 2 recoverable issues to reach 96% target
2. Consider adding few-shot examples for HS code disambiguation
3. Evaluate chain-of-thought prompting for complex questions
"""

    return report


def main():
    """Run error analysis and generate report."""
    print("Analyzing HS Code failures...")
    hs_failures = analyze_hs_code_failures()

    print("Analyzing Diverse failures...")
    diverse_failures = analyze_diverse_failures()

    print("\n" + "="*60)
    print("FAILURE ANALYSIS RESULTS")
    print("="*60)

    print(f"\nHS Code Failures: {len(hs_failures)}")
    for i, f in enumerate(hs_failures, 1):
        print(f"\n  {i}. {f.query[:60]}...")
        print(f"     Score: {f.score} | Type: {f.failure_type}")
        print(f"     Evidence in chunks: {f.evidence_in_chunks} | Recoverable: {f.recoverable}")

    print(f"\nDiverse Failures: {len(diverse_failures)}")
    for i, f in enumerate(diverse_failures, 1):
        print(f"\n  {i}. {f.query[:60]}...")
        print(f"     Score: {f.score} | Type: {f.failure_type}")
        print(f"     Evidence in chunks: {f.evidence_in_chunks} | Recoverable: {f.recoverable}")

    # Generate report
    report = generate_report(hs_failures, diverse_failures)

    # Save report
    output_path = Path("evaluation/experiments/results/ERROR_ANALYSIS.md")
    output_path.write_text(report, encoding="utf-8")
    print(f"\n\nReport saved to: {output_path}")

    # Summary
    hs_recoverable = sum(1 for f in hs_failures if f.recoverable)
    diverse_recoverable = sum(1 for f in diverse_failures if f.recoverable)
    total_recoverable = hs_recoverable + diverse_recoverable

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total failures: {len(hs_failures) + len(diverse_failures)}")
    print(f"Recoverable: {total_recoverable}")
    print(f"Not recoverable: {len(hs_failures) + len(diverse_failures) - total_recoverable}")
    print(f"\nTo reach 96%: Need to fix 2 failures per dataset")
    print(f"Recoverable available: HS={hs_recoverable}, Diverse={diverse_recoverable}")

    if hs_recoverable >= 2 and diverse_recoverable >= 2:
        print("\n✓ 96% target is ACHIEVABLE based on recoverable failures")
    else:
        print("\n⚠ May need additional improvements to reach 96% target")


if __name__ == "__main__":
    main()
