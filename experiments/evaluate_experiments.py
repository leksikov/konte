"""Evaluate experimental project variants and compare correctness.

Compares:
1) wco_hs_explanatory_notes - 8000/800 with context embedded (baseline)
2) exp_8000_800_context_meta - 8000/800 with context in metadata
3) exp_4000_400_context_meta - 4000/400 with context in metadata
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


def save_checkpoint(data: dict | list, path: Path) -> None:
    """Save checkpoint to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Checkpoint saved: {path}", flush=True)


def load_checkpoint(path: Path) -> dict | list | None:
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
    eval_chunks: int,
    timeout: float = 120.0,
) -> dict:
    """Process a single test case with timeout."""
    try:
        # Wrap with timeout
        response, answer = await asyncio.wait_for(
            project.query_with_answer(
                query=tc["input"],
                mode=mode,
                top_k=top_k,
                max_chunks=max_chunks,
            ),
            timeout=timeout,
        )
        retrieval_context = [result.content for result in response.results[:eval_chunks]]
        return {
            "input": tc["input"],
            "expected_output": tc.get("expected_output", ""),
            "actual_output": answer.answer,
            "retrieval_context": retrieval_context,
            "top_score": response.top_score,
            "suggested_action": response.suggested_action,
            "status": "success",
        }
    except asyncio.TimeoutError:
        return {
            "input": tc["input"],
            "expected_output": tc.get("expected_output", ""),
            "actual_output": "Error: Timeout after 120 seconds",
            "retrieval_context": [],
            "top_score": 0.0,
            "suggested_action": "error",
            "status": "timeout",
            "error": "Timeout",
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


async def generate_answers_for_project(
    project_name: str,
    test_cases: list[dict],
    checkpoint_dir: Path,
    mode: str = "hybrid",
    top_k: int = 20,
    max_chunks: int = 15,
    eval_chunks: int = 10,
    batch_size: int = 8,
) -> list[dict]:
    """Generate RAG answers for a single project.

    Returns list of enriched test cases with answers and context.
    """
    checkpoint_path = checkpoint_dir / f"answers_{project_name}_{mode}.json"

    # Check for existing checkpoint
    existing = load_checkpoint(checkpoint_path)
    if existing and len(existing) == len(test_cases):
        print(f"  [{project_name}] Loading from checkpoint ({len(existing)} answers)", flush=True)
        return existing

    # Load project
    print(f"  [{project_name}] Loading project...", flush=True)
    project = Project.open(project_name)
    print(f"    Loaded with {len(project._chunks)} chunks", flush=True)

    # Start from scratch or resume
    start_idx = len(existing) if existing else 0
    results = existing if existing else []

    print(f"  [{project_name}] Generating answers ({start_idx}/{len(test_cases)} done)", flush=True)

    # Process in batches with progress bar
    remaining = test_cases[start_idx:]
    pbar = tqdm(total=len(test_cases), initial=start_idx, desc=f"  {project_name}")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]

        # Process batch in parallel
        tasks = [
            process_single_case(project, tc, mode, top_k, max_chunks, eval_chunks)
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

    # Summary
    success = sum(1 for r in results if r.get("status") == "success")
    print(f"    Done: {success}/{len(results)} success", flush=True)

    return results


def run_correctness_metric(
    answers: dict[str, list[dict]],
    checkpoint_dir: Path,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Run correctness metric on pre-generated answers."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    from evaluation.custom_llm import BackendAIModel

    custom_model = BackendAIModel()

    results = {}

    for project_name, project_answers in answers.items():
        print(f"\n{'='*60}")
        print(f"CORRECTNESS: {project_name}")
        print(f"{'='*60}")

        # Check for checkpoint
        metric_checkpoint = checkpoint_dir / f"correctness_{project_name}.json"
        existing = load_checkpoint(metric_checkpoint)
        if existing:
            print(f"  Loaded from checkpoint", flush=True)
            results[project_name] = existing
            continue

        # Convert to DeepEval test cases
        test_cases = [
            LLMTestCase(
                input=tc["input"],
                actual_output=tc["actual_output"],
                expected_output=tc.get("expected_output"),
                retrieval_context=tc["retrieval_context"],
            )
            for tc in project_answers
        ]

        # Create metric
        metric = GEval(
            name="Correctness",
            criteria="Determine if the actual output is factually correct and semantically similar to the expected output.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=custom_model,
            threshold=threshold,
        )

        print(f"  Evaluating {len(test_cases)} test cases...", flush=True)

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
                "project": project_name,
                "avg_score": sum(valid_scores) / len(valid_scores),
                "pass_rate": sum(1 for s in valid_scores if s >= threshold) / len(valid_scores),
                "passed": sum(1 for s in valid_scores if s >= threshold),
                "failed": sum(1 for s in valid_scores if s < threshold),
                "total": len(valid_scores),
                "errors": errors,
                "avg_retrieval_score": sum(tc["top_score"] for tc in project_answers) / len(project_answers),
            }
            results[project_name] = metric_result
            save_checkpoint(metric_result, metric_checkpoint)

            print(f"    Score: {metric_result['avg_score']:.3f} | Pass: {metric_result['pass_rate']:.1%} | Errors: {errors}", flush=True)
        else:
            print(f"    All {len(test_cases)} test cases failed", flush=True)
            results[project_name] = {"project": project_name, "error": f"All {len(test_cases)} test cases failed"}

    return results


def print_comparison(results: dict) -> None:
    """Print comparison table."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print()
    print(f"{'Configuration':<35} {'Correctness':^12} {'Pass Rate':^12} {'Retrieval':^12}")
    print("-" * 80)

    for project_name, data in results.items():
        if "error" in data:
            print(f"{project_name:<35} {'ERROR':^12}")
        else:
            score = data["avg_score"]
            pass_rate = data["pass_rate"]
            retrieval = data.get("avg_retrieval_score", 0)
            print(f"{project_name:<35} {score:^12.3f} {pass_rate:^12.1%} {retrieval:^12.3f}")

    print("="*80)

    # Explanation
    print("\nConfigurations:")
    print("  1) wco_hs_explanatory_notes    - 8000/800 with context EMBEDDED (baseline)")
    print("  2) exp_8000_800_context_meta   - 8000/800 with context in METADATA")
    print("  3) exp_4000_400_context_meta   - 4000/400 with context in METADATA")


async def main():
    parser = argparse.ArgumentParser(description="Experiment Evaluation")
    parser.add_argument("--projects", nargs="+",
                        default=["wco_hs_explanatory_notes", "exp_8000_800_context_meta", "exp_4000_400_context_meta"])
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-chunks", type=int, default=15)
    parser.add_argument("--eval-chunks", type=int, default=10)
    parser.add_argument("--test-cases", type=Path, default=Path("evaluation/data/synthetic/synthetic_goldens.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--skip-answers", action="store_true", help="Skip answer generation, use existing checkpoints")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics, only generate answers")
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
    print(f"Mode: {args.mode}, top_k: {args.top_k}, max_chunks: {args.max_chunks}")

    # STEP 1: Generate all answers for each project
    all_answers = {}

    if not args.skip_answers:
        print("\n" + "="*60)
        print("STEP 1: GENERATING RAG ANSWERS")
        print("="*60)

        for project_name in args.projects:
            answers = await generate_answers_for_project(
                project_name=project_name,
                test_cases=test_cases,
                checkpoint_dir=checkpoint_dir,
                mode=args.mode,
                top_k=args.top_k,
                max_chunks=args.max_chunks,
                eval_chunks=args.eval_chunks,
            )
            all_answers[project_name] = answers
    else:
        # Load from checkpoints
        print("\nLoading answers from checkpoints...")
        for project_name in args.projects:
            checkpoint_path = checkpoint_dir / f"answers_{project_name}_{args.mode}.json"
            data = load_checkpoint(checkpoint_path)
            if data:
                all_answers[project_name] = data
                print(f"  {project_name}: {len(data)} answers")
            else:
                print(f"  {project_name}: NO CHECKPOINT FOUND")

    if args.skip_metrics:
        print("\nSkipping metrics (--skip-metrics)")
        return

    # STEP 2: Run correctness metric
    print("\n" + "="*60)
    print("STEP 2: RUNNING CORRECTNESS METRIC")
    print("="*60)

    results = run_correctness_metric(
        answers=all_answers,
        checkpoint_dir=checkpoint_dir,
    )

    # Save final results
    results_path = args.output_dir / f"experiment_comparison_{timestamp}.json"
    save_checkpoint(results, results_path)

    # Print comparison
    print_comparison(results)

    print("\nExperiment Evaluation Complete!")


if __name__ == "__main__":
    asyncio.run(main())
