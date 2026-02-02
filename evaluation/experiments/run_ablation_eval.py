"""Run DeepEval correctness evaluation on ablation study results.

Evaluates actual_output vs expected_output for each ablation case.

Supports:
- Case 1: Perfect evidence (contextual)
- Case 1b: Perfect evidence (raw - no generated context)
- Case 2: Evidence injection (contextual)
- Case 2b: Evidence injection (raw - no generated context)
"""

import json
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tqdm import tqdm

from evaluation.custom_llm import BackendAIModel
from evaluation.prompts.eval_prompts import (
    ANSWER_CORRECTNESS_CRITERIA,
    ANSWER_CORRECTNESS_STEPS,
    HS_CODE_CRITERIA,
    HS_CODE_STEPS,
)


def evaluate_ablation_case(case_id: str = "1"):
    """Run DeepEval on ablation case results.

    Args:
        case_id: Ablation case identifier (1, 1b, 2, 2b, or 3).
    """
    results_path = Path(f"evaluation/experiments/results/ablation_case{case_id}_results.json")

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r.get("status") == "success"]
    print(f"Evaluating {len(results)} cases from ablation case {case_id}")
    print(f"Description: {data.get('description', 'N/A')}")

    custom_model = BackendAIModel()
    print(f"Model: {custom_model.model_name}")
    print(f"Endpoint: {custom_model.base_url}")

    # Separate by dataset and evaluate with appropriate metric
    hs_code_results = [r for r in results if r.get("dataset") == "hs_code"]
    deepeval_results = [r for r in results if r.get("dataset") == "deepeval"]

    all_scores = []
    all_detailed = []

    # Evaluate HS code cases
    if hs_code_results:
        print(f"\n--- Evaluating {len(hs_code_results)} HS Code cases ---")
        hs_metric = GEval(
            name="HSCodeCorrectness",
            criteria=HS_CODE_CRITERIA,
            evaluation_steps=HS_CODE_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=custom_model,
            threshold=0.5,
        )

        for r in tqdm(hs_code_results, desc="HS Code"):
            tc = LLMTestCase(
                input=r["input"],
                actual_output=r["actual_output"],
                expected_output=r["expected_output"],
            )
            try:
                hs_metric.measure(tc)
                score = hs_metric.score if hs_metric.score is not None else 0.0
                reason = hs_metric.reason or "unknown"
            except Exception as e:
                score = 0.0
                reason = f"Error: {e}"

            all_scores.append(score)
            all_detailed.append({
                "input": r["input"][:100],
                "dataset": "hs_code",
                "score": score,
                "reason": reason[:200],
                "passed": score >= 0.5,
            })

    # Evaluate DeepEval cases
    if deepeval_results:
        print(f"\n--- Evaluating {len(deepeval_results)} DeepEval cases ---")
        answer_metric = GEval(
            name="AnswerCorrectness",
            criteria=ANSWER_CORRECTNESS_CRITERIA,
            evaluation_steps=ANSWER_CORRECTNESS_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=custom_model,
            threshold=0.5,
        )

        for r in tqdm(deepeval_results, desc="DeepEval"):
            tc = LLMTestCase(
                input=r["input"],
                actual_output=r["actual_output"],
                expected_output=r["expected_output"],
            )
            try:
                answer_metric.measure(tc)
                score = answer_metric.score if answer_metric.score is not None else 0.0
                reason = answer_metric.reason or "unknown"
            except Exception as e:
                score = 0.0
                reason = f"Error: {e}"

            all_scores.append(score)
            all_detailed.append({
                "input": r["input"][:100],
                "dataset": "deepeval",
                "score": score,
                "reason": reason[:200],
                "passed": score >= 0.5,
            })

    # Calculate stats
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    pass_rate = sum(1 for s in all_scores if s >= 0.5) / len(all_scores) if all_scores else 0.0
    passed = sum(1 for s in all_scores if s >= 0.5)

    # By dataset
    hs_detailed = [d for d in all_detailed if d["dataset"] == "hs_code"]
    de_detailed = [d for d in all_detailed if d["dataset"] == "deepeval"]

    hs_passed = sum(1 for d in hs_detailed if d["passed"])
    hs_avg = sum(d["score"] for d in hs_detailed) / len(hs_detailed) if hs_detailed else 0.0

    de_passed = sum(1 for d in de_detailed if d["passed"])
    de_avg = sum(d["score"] for d in de_detailed) / len(de_detailed) if de_detailed else 0.0

    result = {
        "ablation_case": f"case{case_id}",
        "description": data.get("description", ""),
        "evidence_source": data.get("evidence_source", ""),
        "evidence_type": data.get("evidence_type", "contextual"),
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "passed": passed,
        "failed": len(all_scores) - passed,
        "total": len(all_scores),
        "hs_code_passed": hs_passed,
        "hs_code_total": len(hs_detailed),
        "hs_code_avg_score": hs_avg,
        "deepeval_passed": de_passed,
        "deepeval_total": len(de_detailed),
        "deepeval_avg_score": de_avg,
        "detailed_results": all_detailed,
    }

    # Save results
    output_path = Path(f"evaluation/experiments/results/ablation_case{case_id}_deepeval.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"ABLATION CASE {case_id} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Description: {data.get('description', '')}")
    print(f"Evidence source: {data.get('evidence_source', '')}")
    print(f"\n--- Overall ---")
    print(f"  Avg Score: {avg_score:.3f}")
    print(f"  Pass Rate: {pass_rate:.1%}")
    print(f"  Passed: {passed}/{len(all_scores)}")
    print(f"\n--- By Dataset ---")
    print(f"  HS Code: {hs_passed}/{len(hs_detailed)} ({hs_avg:.3f})")
    print(f"  DeepEval: {de_passed}/{len(de_detailed)} ({de_avg:.3f})")
    print(f"\nSaved to: {output_path}")

    # Show failures
    failures = [r for r in all_detailed if not r["passed"]]
    if failures:
        print(f"\n--- FAILURES ({len(failures)}) ---")
        for f in failures[:10]:
            print(f"  [{f['dataset']}] Q: {f['input'][:50]}...")
            print(f"  Score: {f['score']:.2f}")
            print(f"  Reason: {f['reason'][:80]}...")
            print()

    return result


def main():
    """Run evaluation for all ablation cases."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ablation study results")
    parser.add_argument(
        "--case",
        type=str,
        default="0",
        help="Case identifier (1, 1b, 2, 2b, or 3). 0 = all cases including 1b/2b",
    )
    args = parser.parse_args()

    if args.case == "0":
        # Run all cases (including raw evidence variants)
        all_results = []
        # Case 3 has no actual_output (retrieval only)
        for case_id in ["1", "1b", "2", "2b"]:
            results_path = Path(
                f"evaluation/experiments/results/ablation_case{case_id}_results.json"
            )
            if not results_path.exists():
                print(f"\nSkipping case {case_id} - results file not found")
                continue

            print(f"\n{'#'*60}")
            print(f"# ABLATION CASE {case_id}")
            print(f"{'#'*60}")
            result = evaluate_ablation_case(case_id)
            if result:
                all_results.append(result)

        # Summary
        if all_results:
            print(f"\n{'='*60}")
            print("ABLATION STUDY SUMMARY")
            print(f"{'='*60}")
            print(f"\n{'Case':<10} {'Type':<15} {'Pass Rate':<12} {'Avg Score':<10}")
            print("-" * 50)
            for r in all_results:
                evidence_type = r.get("evidence_type", "contextual")
                print(
                    f"{r['ablation_case']:<10} {evidence_type:<15} "
                    f"{r['pass_rate']:.1%}        {r['avg_score']:.3f}"
                )

            # Show comparison if both contextual and raw versions exist
            contextual_cases = {
                r["ablation_case"]: r for r in all_results if "b" not in r["ablation_case"]
            }
            raw_cases = {
                r["ablation_case"]: r for r in all_results if "b" in r["ablation_case"]
            }

            if contextual_cases and raw_cases:
                print(f"\n{'='*60}")
                print("CONTEXTUAL vs RAW EVIDENCE COMPARISON")
                print(f"{'='*60}")
                for case_id, contextual in contextual_cases.items():
                    raw_id = f"{case_id}b"
                    if raw_id in {r["ablation_case"] for r in raw_cases.values()}:
                        raw = next(
                            r for r in raw_cases.values() if r["ablation_case"] == raw_id
                        )
                        delta_rate = contextual["pass_rate"] - raw["pass_rate"]
                        delta_score = contextual["avg_score"] - raw["avg_score"]
                        print(f"\nCase {case_id} vs {raw_id}:")
                        print(
                            f"  Contextual: {contextual['pass_rate']:.1%} ({contextual['avg_score']:.3f})"
                        )
                        print(f"  Raw:        {raw['pass_rate']:.1%} ({raw['avg_score']:.3f})")
                        print(f"  Delta:      {delta_rate:+.1%} ({delta_score:+.3f})")
    else:
        evaluate_ablation_case(args.case)


if __name__ == "__main__":
    main()
