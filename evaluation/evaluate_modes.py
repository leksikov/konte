"""RAG Evaluation with checkpoints.

Step 1: Generate all RAG answers for all modes (saved to checkpoints)
Step 2: Run metrics on saved answers
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from konte import Project


def load_test_cases(json_path: Path) -> list[dict]:
    """Load test cases from JSON file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(data: dict, path: Path) -> None:
    """Save checkpoint to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: Path) -> dict | None:
    """Load checkpoint if exists."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


async def process_single_case(
    project: Project,
    tc: dict,
    mode: str,
    top_k: int,
    max_chunks: int,
    rerank: bool = False,
    rerank_initial_k: int = 50,
) -> dict:
    """Process a single test case."""
    try:
        response, answer = await project.query_with_answer(
            query=tc["input"],
            mode=mode,
            top_k=top_k,
            max_chunks=max_chunks,
            rerank=rerank,
            rerank_initial_k=rerank_initial_k,
        )
        # Store chunks for evaluation (match max_chunks to avoid truncation)
        retrieval_context = [result.content for result in response.results[:max_chunks]]
        return {
            "input": tc["input"],
            "expected_output": tc.get("expected_output", ""),
            "actual_output": answer.answer,
            "retrieval_context": retrieval_context,
            "top_score": response.top_score,
            "suggested_action": response.suggested_action,
            "status": "success",
        }
    except Exception as e:
        return {
            "input": tc["input"],
            "expected_output": tc.get("expected_output", ""),
            "actual_output": f"Error: {e}",
            "retrieval_context": [],
            "top_score": 0.0,
            "suggested_action": "error",
            "status": "error",
            "error": str(e),
        }


async def generate_all_answers(
    project: Project,
    test_cases: list[dict],
    modes: list[str],
    top_k: int,
    max_chunks: int,
    checkpoint_dir: Path,
    batch_size: int = 16,
    rerank: bool = False,
    rerank_initial_k: int = 50,
) -> dict[str, list[dict]]:
    """Generate RAG answers for all modes with checkpoints.

    Returns dict mapping mode -> list of enriched test cases.
    """
    all_results = {}

    # Add suffix for rerank checkpoints
    suffix = "_rerank" if rerank else ""

    for mode in modes:
        checkpoint_path = checkpoint_dir / f"answers_{mode}{suffix}.json"

        # Check for existing checkpoint
        existing = load_checkpoint(checkpoint_path)
        if existing and len(existing) == len(test_cases):
            print(f"\n[{mode.upper()}] Loading from checkpoint ({len(existing)} answers)")
            all_results[mode] = existing
            continue

        # Start from scratch or resume
        start_idx = len(existing) if existing else 0
        results = existing if existing else []

        rerank_label = " +rerank" if rerank else ""
        print(f"\n[{mode.upper()}{rerank_label}] Generating answers ({start_idx}/{len(test_cases)} done)")

        # Process in batches with progress bar
        remaining = test_cases[start_idx:]
        pbar = tqdm(total=len(test_cases), initial=start_idx, desc=f"  {mode}{rerank_label}")

        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]

            # Process batch in parallel
            tasks = [
                process_single_case(project, tc, mode, top_k, max_chunks, rerank, rerank_initial_k)
                for tc in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "input": batch[j]["input"],
                        "expected_output": batch[j].get("expected_output", ""),
                        "actual_output": f"Error: {result}",
                        "retrieval_context": [],
                        "top_score": 0.0,
                        "suggested_action": "error",
                        "status": "error",
                        "error": str(result),
                    })
                else:
                    results.append(result)

            pbar.update(len(batch))

            # Save checkpoint after each batch
            save_checkpoint(results, checkpoint_path)

        pbar.close()
        all_results[mode] = results

        # Summary
        success = sum(1 for r in results if r.get("status") == "success")
        print(f"  Done: {success}/{len(results)} success")

    return all_results


def run_metrics(
    answers: dict[str, list[dict]],
    metrics_to_run: list[str],
    checkpoint_dir: Path,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Run metrics on pre-generated answers using DeepEval's native batch processing."""
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from evaluation.custom_llm import BackendAIModel

    custom_model = BackendAIModel()

    all_metrics = {
        "answer_relevancy": ("Answer Relevancy", lambda: AnswerRelevancyMetric(
            threshold=threshold, model=custom_model, include_reason=False,
        )),
        "faithfulness": ("Faithfulness", lambda: FaithfulnessMetric(
            threshold=threshold, model=custom_model, include_reason=False,
        )),
        "contextual_recall": ("Contextual Recall", lambda: ContextualRecallMetric(
            threshold=threshold, model=custom_model, include_reason=False,
        )),
        "contextual_precision": ("Contextual Precision", lambda: ContextualPrecisionMetric(
            threshold=threshold, model=custom_model, include_reason=False,
        )),
        "contextual_relevancy": ("Contextual Relevancy", lambda: ContextualRelevancyMetric(
            threshold=threshold, model=custom_model, include_reason=False,
        )),
        "correctness": ("FactualCorrectness", lambda: GEval(
            name="FactualCorrectness",
            criteria="""Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.

Focus on:
1. Are the main HS codes or classification codes mentioned in expected output ALSO present in actual output?
2. Are the key facts, categories, or classifications from expected output covered in actual output?
3. Is the core answer semantically equivalent, even if worded differently or more detailed?

IGNORE these differences:
- Length differences (actual may be longer with more detail - this is OK)
- Format differences (bullet points vs paragraphs, etc.)
- Additional context, explanations, or citations in actual output
- Language mixing (Korean text alongside English)
- Different ordering of information

Score 1.0 if: Actual contains all key facts from expected (even if it has more)
Score 0.7-0.9 if: Actual contains most key facts with minor omissions
Score 0.4-0.6 if: Actual contains some key facts but misses important ones
Score 0.0-0.3 if: Actual is missing most key facts or contains wrong information""",
            evaluation_steps=[
                "Extract the key HS codes, classification codes, or category names from the expected output",
                "Check if these same codes/categories appear in the actual output",
                "Extract the key factual claims from the expected output",
                "Verify these facts are present in the actual output (can be worded differently)",
                "Ignore length, format, and stylistic differences",
                "Score based on factual coverage, not format matching",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=custom_model, threshold=threshold,
        )),
    }

    results = {}

    for mode, mode_answers in answers.items():
        print(f"\n{'='*60}")
        print(f"METRICS: {mode.upper()}")
        print(f"{'='*60}")

        # Convert to DeepEval test cases
        test_cases = [
            LLMTestCase(
                input=tc["input"],
                actual_output=tc["actual_output"],
                expected_output=tc.get("expected_output"),
                retrieval_context=tc["retrieval_context"],
            )
            for tc in mode_answers
        ]

        mode_results = {
            "mode": mode,
            "test_cases": len(test_cases),
            "avg_retrieval_score": sum(tc["top_score"] for tc in mode_answers) / len(mode_answers),
            "metrics": {},
        }

        for metric_key in metrics_to_run:
            if metric_key not in all_metrics:
                print(f"  Unknown metric: {metric_key}")
                continue

            metric_name, metric_factory = all_metrics[metric_key]
            metric = metric_factory()

            # Check for checkpoint
            metric_checkpoint = checkpoint_dir / f"metric_{mode}_{metric_key}.json"
            existing = load_checkpoint(metric_checkpoint)
            if existing:
                print(f"  {metric_name}: loaded from checkpoint")
                mode_results["metrics"][metric_key] = existing
                continue

            print(f"  {metric_name}: evaluating {len(test_cases)} test cases...", flush=True)

            # Evaluate test-case-by-test-case to filter errors
            scores = []
            errors = 0
            for i, tc in enumerate(test_cases):
                try:
                    metric.measure(tc)
                    scores.append(metric.score)
                except Exception:
                    errors += 1

                # Progress every 10 cases
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i+1}/{len(test_cases)} (errors: {errors})", flush=True)

            valid_scores = [s for s in scores if s is not None]

            if valid_scores:
                metric_result = {
                    "name": metric_name,
                    "avg_score": sum(valid_scores) / len(valid_scores),
                    "pass_rate": sum(1 for s in valid_scores if s >= threshold) / len(valid_scores),
                    "passed": sum(1 for s in valid_scores if s >= threshold),
                    "failed": sum(1 for s in valid_scores if s < threshold),
                    "total": len(valid_scores),
                    "errors": errors,
                }
                mode_results["metrics"][metric_key] = metric_result
                save_checkpoint(metric_result, metric_checkpoint)

                print(f"    Score: {metric_result['avg_score']:.3f} | Pass: {metric_result['pass_rate']:.1%} | Errors: {errors}")
            else:
                print(f"    All {len(test_cases)} test cases failed")
                mode_results["metrics"][metric_key] = {"name": metric_name, "error": f"All {len(test_cases)} test cases failed"}

        results[mode] = mode_results

    return results


def print_comparison(results: dict) -> None:
    """Print comparison table."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    modes = list(results.keys())
    first_mode = results[modes[0]]
    metric_keys = list(first_mode["metrics"].keys())

    # Header
    header = f"{'Metric':<25}" + "".join(f"{m.upper():^18}" for m in modes)
    print(header)
    print("-" * len(header))

    # Retrieval score
    row = f"{'Avg Retrieval Score':<25}"
    for mode in modes:
        score = results[mode]["avg_retrieval_score"]
        row += f"{score:^18.3f}"
    print(row)

    # Metrics
    for metric_key in metric_keys:
        row = f"{metric_key:<25}"
        for mode in modes:
            metric_data = results[mode]["metrics"].get(metric_key, {})
            if "error" in metric_data:
                row += f"{'ERROR':^18}"
            else:
                score = metric_data.get("avg_score", 0)
                pass_rate = metric_data.get("pass_rate", 0)
                row += f"{score:.3f} ({pass_rate:.0%})".center(18)
        print(row)

    print("="*80)


async def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation with Checkpoints")
    parser.add_argument("--modes", nargs="+", default=["hybrid", "semantic", "lexical"])
    parser.add_argument("--project", default="wco_hs_explanatory_notes")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-chunks", type=int, default=25)
    parser.add_argument("--metrics", nargs="+",
                        default=["answer_relevancy", "faithfulness", "contextual_recall",
                                 "contextual_precision", "contextual_relevancy", "correctness"])
    parser.add_argument("--test-cases", type=Path, default=Path("evaluation/data/synthetic/synthetic_goldens.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--skip-answers", action="store_true", help="Skip answer generation, use existing checkpoints")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics, only generate answers")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking with Qwen3-Reranker-8B")
    parser.add_argument("--rerank-initial-k", type=int, default=50, help="Initial candidates for reranking")
    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load test cases
    print(f"Loading test cases from {args.test_cases}")
    test_cases = load_test_cases(args.test_cases)
    print(f"Loaded {len(test_cases)} test cases")

    # STEP 1: Generate all answers
    if not args.skip_answers:
        print(f"\nLoading project: {args.project}")
        project = Project.open(args.project)
        print(f"Project loaded with {len(project._chunks)} chunks")

        rerank_label = " +rerank" if args.rerank else ""
        print("\n" + "="*60)
        print(f"STEP 1: GENERATING RAG ANSWERS{rerank_label}")
        print("="*60)

        answers = await generate_all_answers(
            project=project,
            test_cases=test_cases,
            modes=args.modes,
            top_k=args.top_k,
            max_chunks=args.max_chunks,
            checkpoint_dir=checkpoint_dir,
            rerank=args.rerank,
            rerank_initial_k=args.rerank_initial_k,
        )
    else:
        # Load from checkpoints
        suffix = "_rerank" if args.rerank else ""
        print(f"\nLoading answers from checkpoints{' (rerank)' if args.rerank else ''}...")
        answers = {}
        for mode in args.modes:
            checkpoint_path = checkpoint_dir / f"answers_{mode}{suffix}.json"
            data = load_checkpoint(checkpoint_path)
            if data:
                answers[mode] = data
                print(f"  {mode}: {len(data)} answers")
            else:
                print(f"  {mode}: NO CHECKPOINT FOUND")

    if args.skip_metrics:
        print("\nSkipping metrics (--skip-metrics)")
        return

    # STEP 2: Run metrics
    print("\n" + "="*60)
    print("STEP 2: RUNNING METRICS")
    print("="*60)

    results = run_metrics(
        answers=answers,
        metrics_to_run=args.metrics,
        checkpoint_dir=checkpoint_dir,
    )

    # Save final results
    results_path = args.output_dir / f"comparison_{timestamp}.json"
    save_checkpoint(results, results_path)

    # Print comparison
    print_comparison(results)

    print("\nEvaluation Complete!")


if __name__ == "__main__":
    asyncio.run(main())
